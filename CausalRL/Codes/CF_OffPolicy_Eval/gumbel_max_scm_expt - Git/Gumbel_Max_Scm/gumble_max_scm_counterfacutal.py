
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random
from collections import deque, namedtuple
from collections import deque
import torch
import argparse
import glob
import random
#from agent import CQLAgent
import pandas as pd
from collections import defaultdict
import math

class gumble_max_scm:
    def __init__(self):
        print('**** Main posterior Class Gumbel SCM ***')
    def truncated_gumbel(self,logit, truncation):
        """truncated_gumbel
    
        :param logit: Location of the Gumbel variable (e.g., log probability)
        :param truncation: Value of Maximum Gumbel
        """
        # Note: In our code, -inf shows up for zero-probability events, which is
        # handled in the topdown function
        assert not np.isneginf(logit)
    
        gumbel = np.random.gumbel(size=(truncation.shape[0])) + logit
        trunc_g = -np.log(np.exp(-gumbel) + np.exp(-truncation))
        return trunc_g
    
    def topdown(self,logits, k, nsamp=1):
        """topdown
    
        Top-down sampling from the Gumbel posterior
    
        :param logits: log probabilities of each outcome
        :param k: Index of observed maximum
        :param nsamp: Number of samples from gumbel posterior
        """
        np.testing.assert_approx_equal(np.sum(np.exp(logits)), 1), "Probabilities do not sum to 1"
        ncat = logits.shape[0]
    
        gumbels = np.zeros((nsamp, ncat))
    
        # Sample top gumbels
        topgumbel = np.random.gumbel(size=(nsamp))
    
        for i in range(ncat):
            # This is the observed outcome
            if i == k:
                gumbels[:, k] = topgumbel - logits[i]
            # These were the other feasible options (p > 0)
            elif not(np.isneginf(logits[i])):
                gumbels[:, i] = self.truncated_gumbel(logits[i], topgumbel) - logits[i]
            # These have zero probability to start with, so are unconstrained
            else:
                gumbels[:, i] = np.random.gumbel(size=nsamp)
    
        return gumbels
    
    
    def tx_posterior(self,p_c, p_t, obs=0, n_samp=1000):
        """tx_posterior
    
        Get a posterior over counterfactual transitions
    
        :param p_c: "Control" probabilities, under observed action
        :param p_t: "Treatment" probabilities, under different action
        :param obs: Observed outcome under observed action
        :param n_samp: Number of Monte Carlo samples from posterior
        """
        assert isinstance(p_c, list), "Pass probabilities in as a list!"
        assert isinstance(p_t, list), "Pass probabilities in as a list!"
    
        n_cat = len(p_c)
        assert len(p_c) == len(p_t)
        assert obs in range(n_cat), "Obs is {}, not valid!".format(obs)
        np.testing.assert_approx_equal(np.sum(p_c), 1)
        np.testing.assert_approx_equal(np.sum(p_t), 1)
    
        # Define our categorical logits
        with np.errstate(divide='ignore'):
            logits_control = np.log(np.array(p_c))
            logits_treat = np.log(np.array(p_t))
    
        assert p_c[obs] != 0, "Probability of observed event was zero!"
    
        # Note:  These are the Gumbel values (just g), not log p + g
        posterior_samp = self.topdown(logits_control, obs, n_samp)
    
        # The posterior under control should give us the same result as the obs
        assert ((posterior_samp + logits_control).argmax(axis=1) == obs).sum() == n_samp
    
        # Counterfactual distribution
        # This throws a RunTimeWarning because logits_treat includes some -inf, but
        # that is expected
        posterior_sum = posterior_samp + logits_treat
    
        # Because some logits are -inf, some entries of posterior_sum will be nan,
        # but this is OK - these correspond to zero-probability transitions.  We
        # just assert here that at least one of the entries for each sample is an
        # actual number (read the assert below as:  Make sure that none of the
        # samples have all NaNs)
        assert not np.any(np.all(np.isnan(posterior_sum), axis=1))
        posterior_treat = posterior_sum.argmax(axis=1)
    
        # Reshape posterior argmax into a 1-D one-hot encoding for each sample
        mask = np.zeros(posterior_sum.shape)
        mask[np.arange(len(posterior_sum)), posterior_treat] = 1
        posterior_prob = mask.sum(axis=0) / mask.shape[0]
   
        return posterior_prob
        
class gumble_max_scm_process:
    def __init__(self,cql_agent,obs_arr,matrix_to_seq_dct,r_mat_full,tx_mat_full,NUM_FULL_STATES,n_cf_samps=5):
        self.gmcm=gumble_max_scm()
        self.n_cf_samps=n_cf_samps
        self.obs_arr=obs_arr
        self.matrix_to_seq_dct=matrix_to_seq_dct
        self.r_mat_full=r_mat_full
        self.tx_mat_full=tx_mat_full
        self.cql_agent=cql_agent
        self.NUM_FULL_STATES=NUM_FULL_STATES 
    def cf_trajectories_generation(self,n_draws=1000):
        print(f"batch trajectories shape: {self.obs_arr.shape}")
        #n_cf_samps=1
        tqdm_desc='CF OPE'
        #batch==self.obs_arr
        obs_samps_proj=self.obs_arr
        batch=self.obs_arr
        n_obs_eps = batch.shape[0]
        n_obs_steps = batch.shape[1]
        #print(n_obs_eps,n_obs_steps)
        print(f"batch trajectories: {n_obs_eps}, batch steps: {n_obs_steps}")

        mx_posterior = np.ones((n_obs_eps, 1))
        #print(n_obs_eps,n_obs_steps,mx_posterior[0])
        
        result = np.zeros((n_obs_eps, self.n_cf_samps, n_obs_steps, 5))
        result[:, :, :, 0] = np.arange(n_obs_steps)
        result[:, :, :, 1:4] = -1  # Placeholders for end of sequence
        #for obs_samp_idx in tqdm(range(n_obs_eps), disable=not(use_tqdm), desc=tqdm_desc):
        cf_policy='CQL'
        eps=0.01
        for obs_samp_idx in range(n_obs_eps):
        
            for cf_samp_idx in range(self.n_cf_samps):
                obs_actions = batch[obs_samp_idx, :, 1].astype(int).squeeze().tolist()
                obs_from_states = batch[obs_samp_idx, :, 2].astype(int).squeeze().tolist()
                obs_to_states = batch[obs_samp_idx, :, 3].astype(int).squeeze().tolist()
                # Same initial state
                current_state = obs_from_states[0]
                # Infer / Sample from the mixture posterior
                this_mx_posterior = mx_posterior[obs_samp_idx].tolist()
                # component = np.random.choice(
                #     self.mdp.n_components, size=1, p=this_mx_posterior)
                for time_idx in range(n_obs_steps):
                    obs_action = obs_actions[time_idx]
                    if cf_policy is None:  # Random Policy
                        cf_action = np.random.randint(self.mdp.n_actions)
                    else:
                        stateNN=self.matrix_to_seq_dct[current_state]
                        cf_action =self.cql_agent.get_action(stateNN, epsilon=eps)[0]
                        #print(cf_action,current_state)
                        #action = agent.get_action(stateNN, epsilon=eps)
                        # cf_action = np.random.choice(
                        #     self.mdp.n_actions, size=1,
                        #     p=cf_policy[current_state, :].squeeze().tolist())[0]
        
                    # Interventional probabilities under new action
                    new_interv_probs = self.tx_mat_full[cf_action, current_state,:].squeeze().tolist()
                    total_sum = sum(new_interv_probs)
                    new_interv_probs = [x / total_sum for x in new_interv_probs]
                    #print(new_interv_probs)
                        # self.mdp.tx_mat[component,
                        #                 cf_action, current_state,
                        #                 :].squeeze().tolist()
        
                    # If observed sequence did not terminate, then infer cf
                    # probabilities;  Otherwise treat this as an interventional
                    # query (once we're past the final time-step of the
                    # observed sequence, there is no posterior over latents)
                    if obs_action == -1:
                        cf_probs = new_interv_probs
                    else:
                        # Old and new interventional probabilities
                        prev_interv_probs =  self.tx_mat_full[ obs_action, obs_from_states[time_idx],
                                            :].squeeze().tolist()
                        total_sum = sum(prev_interv_probs)
                        prev_interv_probs = [x / total_sum for x in prev_interv_probs]
                        #print(prev_interv_probs)
                            # self.mdp.tx_mat[component,
                            #                 obs_action, obs_from_states[time_idx],
                            #                 :].squeeze().tolist()
        
                        assert prev_interv_probs[obs_to_states[time_idx]] != 0
                        # Infer counterfactual probabilities
                        #print(prev_interv_probs,new_interv_probs)
                        #print(obs_to_states[time_idx])
                        cf_probs = self.gmcm.tx_posterior(
                            prev_interv_probs, new_interv_probs,
                            obs=obs_to_states[time_idx],
                            n_samp=n_draws).tolist()
                    next_state = np.random.choice(
                        self.NUM_FULL_STATES, size=1, p=cf_probs)[0]
                    this_reward = self.r_mat_full[
                         cf_action, current_state, next_state]
                    # Record result
                    result[obs_samp_idx, cf_samp_idx, time_idx] = (
                        time_idx,
                        cf_action,
                        current_state,
                        next_state,
                        this_reward)
        
                    if this_reward != 0 and time_idx != n_obs_steps - 1:
                        # Fill in next state, convention in obs_samps
                        result[obs_samp_idx, cf_samp_idx, time_idx + 1] = (
                            time_idx + 1,
                            -1,
                            next_state,
                            -1,
                            0)
                        break
        
                    current_state = next_state
        #print(result.shape)
        print(f"CF trajectories generated: {result.shape}")
        return result

         
    def process_cf_trajs(self):
        result =self.cf_trajectories_generation()
        traj_arr_ls=[]
        for indx in range(result.shape[0]):
            random_indexes_cfs = range(0, self.n_cf_samps)
            temp=[]
            for idx in random_indexes_cfs:
              data = result[indx,idx,:,:]
              first_index_arr = np.where(data == -1)[0]
              if first_index_arr.shape[0]:
                first_index = first_index_arr[0]
                res=result[indx,idx,:first_index,1:]
              else:
                res=result[indx,idx,:,1:]
              resr=res[:,3]
              #donnes=[0 if x <=0 else 1 for x in resr]
              resrdonnes=np.array([0 if x <=0 else 1 for x in resr]).reshape(len(res), 1)
              new_data = np.hstack((res, resrdonnes))
              #print(new_data.shape)
              temp.append((new_data,resr[-1]))
            traj_arr_ls.append(temp)
        return traj_arr_ls

    def process_cf_trajs(self):
      result =self.cf_trajectories_generation()
      traj_arr_ls=[]
      for indx in range(result.shape[0]):
        random_indexes_cfs = range(0, self.n_cf_samps)
        temp=[]
        for idx in random_indexes_cfs:
          data = result[indx,idx,:,:]
          first_index_arr = np.where(data == -1)[0]
          if first_index_arr.shape[0]:
            first_index = first_index_arr[0]
            res=result[indx,idx,:first_index,1:]
          else:
            res=result[indx,idx,:,1:]
          resr=res[:,3]
          #donnes=[0 if x <=0 else 1 for x in resr]
          resrdonnes=np.array([0 if x <=0 else 1 for x in resr]).reshape(len(res), 1)
          new_data = np.hstack((res, resrdonnes))
          #print(new_data.shape)
          #temp.append((new_data,resr[-1]))
          temp.append(new_data)
        traj_arr_ls.append(temp)
      #print(len(traj_arr_ls))
      #traj_cf=np.array(traj_arr_ls)
      return traj_arr_ls

    def process_obs_for_traj(self):
      obj_process=[]
      for i in range(self.obs_arr.shape[0]):
        #vals=obs_samps_proj[i,:,:]
        vals=self.obs_arr[i,:,:]
        temp=[]
        for tx in range(obs_arr.shape[1]):
          exps=vals[tx,:][1:]
          exps=[int(exps[index]) if index in [0,1,2,4] else float(exps[index])  for index in range(len(exps))]
          #print(exps)]
          rewd=exps[len(exps)-2]   
          if  exps[0]==-1:
              break
          temp.append((exps,rewd))
        obj_process.append(temp)
      return obj_process
        
def cf_traj_comaprision(obj_process,traj_cf):
    #print(len(obj_process),len(traj_cf))
    print(f"base Trajectories: {len(obj_process)}, CF Trajectories: {len(traj_cf)}")
    
    lsx=[]
    for index in range(len(obj_process)):
        obj_traj=obj_process[index]
        obj_traj_arr=np.array(obj_traj)
        n,m=obj_traj_arr.shape
        obj_traj_reward=obj_traj_arr[n-1,m-2]
        #print(obj_traj_arr,obj_traj_reward)
        cf_trajindx_lsts=traj_cf[index]
        cfrwds=[]
        for cf_indx in range(len(cf_trajindx_lsts)):
            cf_traj=cf_trajindx_lsts[cf_indx]
            cf_traj_arr=np.array(cf_traj)
            n,m=cf_traj_arr.shape
            cf_traj_reward=cf_traj_arr[n-1,m-2]
            #print(cf_indx,cf_traj_arr,cf_traj_reward)
            cfrwds.append(cf_traj_reward)
      #print(index,cfrwds,np.mean(cfrwds),obj_traj_reward)
        print("index: {} | basepolicy_reward: {} | conterfactual_mean: {} | conterfactual_traj_rewards: {}".format(index, obj_traj_reward, np.round(np.mean(cfrwds),3), cfrwds))
        lsx.append((index,obj_traj_reward,np.round(np.mean(cfrwds),3),cfrwds))
    df=pd.DataFrame(lsx)
    df.columns=['Trajectory-Id', 'Base_Policy_Reward' ,'Mean_Cf_Reward', 'Cf_rewards']
    return df

#from Gumbel_Max_Scm import gumble_max_scm_counterfacutal as gmsc_cf 
#len(obs_arr)