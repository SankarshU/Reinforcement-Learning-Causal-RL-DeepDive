import gym
from collections import defaultdict
from gym.envs.toy_text.frozen_lake import generate_random_map
import pandas as pd
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import tqdm

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
import time
from matplotlib.patches import Patch
import seaborn as sns
def moving_average(x, span=25):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values

def make_env( map_name='4x4',is_slippery=True):
    #return gym.make('FrozenLake-v1',desc=generate_random_map(size=8)).env  # .env unwraps the TimeLimit wrapper
    return gym.make('FrozenLake-v1',is_slippery=True,map_name=map_name).env  # .env unwraps the TimeLimit wrapper


def plot_rewards(rewards,window=1000):
    moving_avg_reward = []
    #window = 1000
    for i in range(window, len(rewards)):
        moving_avg_reward.append(100*sum(rewards[i-window:i])/window)
    
    fig, axes = plt.subplots(figsize=(8, 8))
    plt.plot(range(window, len(rewards)), moving_avg_reward)
    axes.set(xlabel='Episode Idx', ylabel='Success Rate', title='Expected reward with a moving average with window size = {}'.format(window))
    plt.show()

def plot_all_rewards(batch_policy_reward, offline_cql_reward, window=1000):
    def moving_average(rewards, window):
        moving_avg_reward = []
        for i in range(window, len(rewards)):
            moving_avg_reward.append(100 * sum(rewards[i-window:i]) / window)
        return moving_avg_reward

    # Compute moving averages
    batch_policy_moving_avg = moving_average(batch_policy_reward, window)
    offline_cql_moving_avg = moving_average(offline_cql_reward, window)

    # Create x-axis indices
    x1 = np.arange(window, len(batch_policy_reward))
    x2 = np.arange(window, len(offline_cql_reward))

    # Plotting
    fig, axes = plt.subplots(figsize=(10, 6))
    plt.plot(x1, batch_policy_moving_avg, label='Base Policy Reward', color='blue')
    plt.plot(x2, offline_cql_moving_avg, label=' Target Agent(CQL) Reward(trained on base Trajectories)', color='orange')

    # Add labels, title, and legend
    axes.set(xlabel='Time Index', ylabel='Success Rate', title='Expected Reward with Moving Average (Window Size = {})'.format(window))
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def  histogram(df):
    plt.figure(figsize=(12, 4))
    
    # Create a histogram
    n, bins, patches = plt.hist(df['Deviation'], bins=10, edgecolor='black')
    
    # Set the colors of the bars based on the bin values
    for patch in patches:
        if patch.get_x() >= 0 and patch.get_x() + patch.get_width() > 0:
            patch.set_facecolor('red')
        elif patch.get_x() < 0 and patch.get_x() + patch.get_width() <= 0:
            patch.set_facecolor('blue')
        else:
            patch.set_facecolor('green')
    
    # Adding titles and labels
    plt.title('Histogram of Deviations with Color Coding')
    plt.xlabel('Deviation')
    plt.ylabel('Frequency')
    
    # Create custom legend
    legend_patches = [Patch(color='green', label='base = Target Agent(CQL) mean Cf rewards'),
                      Patch(color='blue', label='base <  Target Agent(CQL) mean Cf rewards'),
                      Patch(color='red', label='base >  Target Agent(CQL) mean Cf rewards')]
    
    
    # Add legend to plot
    plt.legend(handles=legend_patches)
    
    # Displaying the plot
    plt.show()
def boxplot(df):
    # Set up the matplotlib figure
    plt.figure(figsize=(4, 4))
    
    # Plot Base_Policy_Reward vertically
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    sns.boxplot(y=df['Base_Policy_Reward'])
    plt.title('Box Plot of Base_Policy_Reward')
    
    # Plot Mean_Cf_Reward vertically
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    sns.boxplot(y=df['Target Agent(CQL)Mean_Cf_Reward'])
    plt.title('Box Plot of Target Agent Mean_Cf_Reward')
    
    # Show plots
    plt.tight_layout()
    plt.show()
def Qlearning(env, n=25000, max_steps=100,epsilon = 0.25,hole_states=[]):
    #all_states = range(n_states)
   
    num_actions = env.action_space.n
    num_states = env.observation_space.n
    print("All possible states:", list(range(num_states)))
    q_table = np.zeros((num_states, num_actions))
    # Q-table update parameters
    alpha = 0.8
    gamma = 0.95
    # Exploration paramters
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.001
    rewards = []
    epsilon = 0.25
    #n, max_steps = 25000, 100
    done=False
    for episode in range(n):
        s = env.reset()
        total_reward = 0
        for i in range(max_steps):
        #while not done:
            if random.uniform(0, 1) < epsilon:
                a = env.action_space.sample()
            else:
                a = np.argmax(q_table[s, :])
    
            s_new, r, done, info = env.step(a)
            if s_new in hole_states:
              r=-0.15
    
            q_table[s, a] = q_table[s, a] + alpha*(r + gamma*np.max(q_table[s_new, :]) - q_table[s, a])
            s, total_reward = s_new, total_reward+r
            if done:
                rewards.append(total_reward)
                epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
                #print("Episode: {} | Reward: {} | Epsilon: {}".format(episode, total_reward, epsilon))
                break
    #env.close()
    return rewards,q_table,epsilon
    
def process_obs_for_traj(obs_arr):
  obj_process,obj_exp=[],[]
  for i in range(obs_arr.shape[0]):
    #vals=obs_samps_proj[i,:,:]
    vals=obs_arr[i,:,:]
    temp,temp2=[],[]
    for tx in range(obs_arr.shape[1]):
      exps=vals[tx,:][1:]
      exps=[int(exps[index]) if index in [0,1,2,4] else float(exps[index])  for index in range(len(exps))]
      #print(exps)]
      rewd=exps[len(exps)-2]   
      if  exps[0]==-1:
          break
      temp.append((exps,rewd))
      temp2.append(exps)
    
    obj_process.append(temp)
    obj_exp.append(temp2)
  return obj_process,obj_exp
class expt_cf_utility_frozen_lake:
    def __init__(self,env,agent,hole_states,matrix_to_seq_dct,q_table,max_steps=100,epsilon=0.10):
        self.env=env
        self.hole_states=hole_states
        self.q_table=q_table
        self.max_steps=max_steps
        self.epsilon=epsilon
        self.agent=agent
        self.matrix_to_seq_dct=matrix_to_seq_dct
        self.sample_store=[]
    def collect_random_fronzen_lake( self,dataset,obj_exp):
    
        #n, max_steps = num_samples, 50
        #epsilon=0.10#0.01best #0.25
        for episode in obj_exp:
            for exp in episode:
                a,s,s_new,r,done=exp
                if s_new in self.hole_states:
                  r=-0.15
                #exp.append([s,a,r,s_new,done])
                stateNN=self.matrix_to_seq_dct[s]
                s_newNN=self.matrix_to_seq_dct[s_new]
                dataset.add(stateNN,a,r,s_newNN,done)
    def base_trajectories_fronzen_lake(self,n=1000,epsilon=0.10):
        ### Trajectories from base policy
        #epsilon=0.10 #0.25 # small room to explore
        obs=[]
        for episode in range(n):
            s = self.env.reset()
            total_reward = 0
            exp=[]
            for i in range(self.max_steps):
                if random.uniform(0, 1) < epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = np.argmax(self.q_table[s, :])
        
                s_new, r, done, info = self.env.step(a)
                if s_new in self.hole_states:
                  r=-0.15
                exp.append([i,a,s,s_new,r,bool(done)])
                s=s_new
                total_reward += r
                if r==1:
                  break
                if done:
                  #rewards.append(total_reward)
                  #epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
                  break
            for j in range(self.max_steps-len(exp)):
              exp.append([i+j+1,-1,-1,-1,0,False])
            #print(len(exp))
            obs.append(exp)
        obs_arr=np.array(obs)
        self.env.close()
        #print(obs_arr.shape)
        print(f" base policy obeservations shape: {obs_arr.shape}")
        return obs_arr
    def base_trajectories_fronzen_lake_old(self,n=1000,epsilon=0.10):
        ### Trajectories from base policy
        #epsilon=0.10 #0.25 # small room to explore
        obs=[]
        for episode in range(n):
            s = self.env.reset()
            total_reward = 0
            exp=[]
            for i in range(max_steps):
                if random.uniform(0, 1) < epsilon:
                    a = env.action_space.sample()
                else:
                    a = np.argmax(q_table[s, :])
        
                s_new, r, done, info = env.step(a)
                if s_new in self.hole_states:
                  r=-0.15
                exp.append([i,a,s,s_new,r,bool(done)])
                s=s_new
                total_reward += r
                if r==1:
                  break
                if done:
                  #rewards.append(total_reward)
                  #epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
                  break
            for j in range(max_steps-len(exp)):
              exp.append([i+j+1,-1,-1,-1,0,False])
            #print(len(exp))
            obs.append(exp)
        obs_arr=np.array(obs)
        env.close()
        #print(obs_arr.shape)
        print(f" base policy obeservations shape: {obs_arr.shape}")
        return obs_arr
    def play_game_fronzen_lake(self,agent,eps=0.10,tries=200):
        frames=[]
        self.env.reset()
        frames.append(self.env.render(mode="rgb_array"))
        s = self.env.reset()
        #eps=0.145
        flag=False
        frames=[]
        for episode in range(tries):
            if episode%50==0:
              print(f'Tries: = {episode}')
            state = self.env.reset()
            steps=0
            frames.append(self.env.render(mode="rgb_array"))
            for i in range(self.max_steps):
                stateNN=self.matrix_to_seq_dct[state]
                action = self.agent.get_action(stateNN, epsilon=eps)
                steps += 1
                next_state, reward, done, _ = self.env.step(action[0])
                next_stateNN=self.matrix_to_seq_dct[next_state]
                state=next_state
                frames.append(self.env.render(mode="rgb_array"))
                time.sleep(0.05)
                if reward==1:
                  flag=True
                  print(f'success: episode = {episode}, steps = {steps}')
                  break
                if  done:
                  frames=[]
                  steps=0
                  break
            if flag is True:
              break
        self.env.close()
        if len(frames)==0:
            print('***** No game sucessful !!!!! ****')
        return frames    
    def train_cql_fronzen_lake(self,buffer,episodes=2000,min_eps=0.01,max_steps=100,log=False):
        seed=1
        buffer_size=100000
        rewardsl=[]
        eps = 1.
        d_eps = 1 - min_eps
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(max_epsilon,min_epsilon)
        max_epsilon = 1.0
        min_epsilon = 0.01
        decay_rate = 0.001
        steps = 0
        average10 = deque(maxlen=10)
        total_steps = 0
        eps_frames=1e4
        #print(len(buffer.memory),'experiences')
        print(f"experiences: {len(buffer.memory)}, max_epsilon: {max_epsilon}, min_epsilon: {min_epsilon}")
        
        for i in range(1, episodes+1):
          state = self.env.reset()
          episode_steps = 0
          rewards = 0
          #while True:
          for j in range(self.max_steps):
              #stateNN=np.array(matrix_to_seq[state])
              stateNN=self.matrix_to_seq_dct[state]
              action = self.agent.get_action(stateNN, epsilon=eps)
              steps += 1
              next_state, reward, done, _ = self.env.step(action[0])
              next_stateNN=self.matrix_to_seq_dct[next_state]    
              loss, cql_loss, bellmann_error = self.agent.learn(buffer.sample())
              state = next_state
              if next_state in self.hole_states:
                    reward=-0.15
              rewards += reward
              episode_steps += 1
              eps = max(1 - ((steps*d_eps)/eps_frames), min_eps)
              if done:
                  rewardsl.append(rewards)
                  eps = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*i)
                  break   
          #average10.append(rewards)
          #rewardsl.append(rewards)
          #total_steps += episode_steps
          if (i % 20 == 0 or (i%2==0 and reward==1)) and log is True :
              print("Episode: {} | Reward: {} | Q Loss: {} | Steps: {}|eps : {}".format(i, rewards, loss, episode_steps,np.round(eps,3)))
        return rewardsl,self.agent
    def collect_random_fronzen_lake(self, dataset,obj_exp):

        #n, max_steps = num_samples, 50
        #epsilon=0.10#0.01best #0.25
        for episode in obj_exp:
            for exp in episode:
                a,s,s_new,r,done=exp
                if s_new in self.hole_states:
                  r=-0.15
                #exp.append([s,a,r,s_new,done])
                stateNN=self.matrix_to_seq_dct[s]
                s_newNN=self.matrix_to_seq_dct[s_new]
                dataset.add(stateNN,a,r,s_newNN,done)
    
    def collect_random_fronzen_lake_old(self, dataset, num_samples=200,epsilon=0.10):
    
        #n, max_steps = num_samples, 50
        #epsilon=0.10#0.01best #0.25
        obs=[]
        for episode in range(num_samples):
            s = self.env.reset()
            total_reward = 0
            exp=[]
            for i in range(self.max_steps):
                if random.uniform(0, 1) < epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = np.argmax(self.q_table[s, :])
    
                s_new, r, done, info = self.env.step(a)
                #if s_new in [5,7,11,12]:
                if s_new in self.hole_states:
                  r=-0.15
                #exp.append([s,a,r,s_new,done])
                stateNN=self.matrix_to_seq_dct[s]
                s_newNN=self.matrix_to_seq_dct[s_new]
                dataset.add(stateNN,a,r,s_newNN,done)
                self.sample_store.append((stateNN,a,r,s_newNN,done))
                #dataset.add(state, action, reward, next_state, done)
                s=s_new
                if done:
                  #epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
                  break