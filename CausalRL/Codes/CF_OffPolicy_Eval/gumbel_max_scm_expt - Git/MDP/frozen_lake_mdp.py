
# def make_env():
#     #return gym.make('FrozenLake-v1',desc=generate_random_map(size=8)).env  # .env unwraps the TimeLimit wrapper
#     return gym.make('FrozenLake-v1',is_slippery=True).env  # .env unwraps the TimeLimit wrapper

import gym
# env = make_env()
import numpy as np
class Frozen_MDP():
    def __init__(self, env,envsize=4,hole_reward=-0.15):
        self.env=env
        #self.matrix_to_seq=matrix_to_seq
        #self.matrix_to_seq_rev=matrix_to_seq_rev
        self.envsize=envsize
        self.n_actions = self.env.action_space.n
        self.NUM_FULL_STATES = self.env.observation_space.n
        self.tx_mat_full = np.zeros((self.n_actions, self.NUM_FULL_STATES, self.NUM_FULL_STATES))
        self.r_mat_full = np.zeros((self.n_actions, self.NUM_FULL_STATES, self.NUM_FULL_STATES))
        self.hrwrd=-0.15
        #print(self.tx_mat_full.shape,self.r_mat_full.shape)
        print(f"tx_mat_full shape: {self.tx_mat_full.shape}, r_mat_full shape: {self.r_mat_full.shape}")
        all_states = range(self.NUM_FULL_STATES)
        print("All possible states:", list(all_states))
        all_actions = range(self.n_actions)
        print("All possible actions:", list(all_actions))
        self.hole_states=[]
        self.matrix_to_seq=None
        self.matrix_to_seq_rev=None
        self.matrix_to_seq_dct=None
        self.create_encoding_dict()
        _,_=self.extract_states_info()
        self.mdp()
    def create_encoding_dict(self):
        n=self.envsize
        encode = {i: [1 if i == j else 0 for j in range(n)] for i in range(n)}
        matrix_to_seq={}
        matrix_to_seq_dct={}
        for r  in range(n):
          for c in range(n):
              f= n*r+c
              matrix_to_seq[f]=(r,c)
              #print(r,c, f)
        matrix_to_seq_rev={v:k for k,v in matrix_to_seq.items()}
        for key in matrix_to_seq:
            matrix_to_seq_dct[key]=np.array(encode[matrix_to_seq[key][0]]+encode[matrix_to_seq[key][1]])
        self.matrix_to_seq=matrix_to_seq
        self.matrix_to_seq_rev=matrix_to_seq_rev
        self.matrix_to_seq_dct=matrix_to_seq_dct
    
        
    def extract_states_info(self):
        desc = self.env.desc
        for row in range(desc.shape[0]):
            for col in range(desc.shape[1]):
                if desc[row, col] == b'G':
                    goal_state = row * desc.shape[1] + col
                elif desc[row, col] == b'H':
                    self.hole_states.append(row * desc.shape[1] + col)
        print("Goal state:", goal_state)
        print("Hole states:", self.hole_states)
        #self.hole_states=hole_states
        return goal_state,self.hole_states
    #def mad
    
    def mdp(self):
        # 0: LEFT ;# 1: DOWN ;# 2: RIGHT ;# 3: UP
        #goal_state,hole_states=self.extract_states_info()
        reward_dict={15:1}
        for key in self.hole_states:
            reward_dict[key]=self.hrwrd
        action_dct={
        0: (0,-1),
        1: (1,0),
        2: (0,1),
        3: (-1,0)
        }
        print(reward_dict)
        prob_dir={ 0: [0,1,3],1:[0,1,2],2:[1,2,3],3:[0,2,3]}
        ### Lets build the MDP
        for action in range(self.n_actions):# [0]:
          for state in range(self.NUM_FULL_STATES): #[1]:
            curr_pos=self.matrix_to_seq[state]
            possible_actions=prob_dir[action]
            for action_s in possible_actions:
              nex_matrix_pos=(curr_pos[0]+action_dct[action_s][0],curr_pos[1]+action_dct[action_s][1])
              if nex_matrix_pos in self.matrix_to_seq_rev.keys():
                next_state=self.matrix_to_seq_rev[nex_matrix_pos]
                self.tx_mat_full[action,state,next_state]=1/3
                if next_state in reward_dict:
                  self.r_mat_full[action,state,next_state]=reward_dict[next_state]
              else:
                self.tx_mat_full[action,state,state]=1/3
        
