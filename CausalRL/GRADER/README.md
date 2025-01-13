### Main Summary

GRADER is a itertiave learning framework  that learns Causal graph from sampled buffer, learns transition model , transition model acts as proxy to environment
which helps drawing trajectories ( N time steps) and Trajectories are used in planning to learn the policy, planning aim is to reach towards goal

 <img width="324" alt="image" src="https://github.com/user-attachments/assets/4d14cc7d-44b5-4538-8323-5d84ecab937c" />

- This method application requires problem to be defined as  graph ( for eg: KPI is in a telecommunication network) & goals can be intents on KPIs that a polciy should achieve, in Goal conditional set up the paper observes RL agorithms such as SAC perform poorely in genralizing to different goals , i should be noted that invariance captured Causal graph and SCM that models it  ,  if graphical structre  applicable to a problem domian this work can be leveraged     

- It also requires to understand one of environments on which experiments are conducted , which will be breiefly described below 

**Published Forum:** NeurIPS, 2022 ,[paper](https://arxiv.org/abs/1905.05824](https://arxiv.org/abs/2207.09081)][)  
**Experiment Setup (Publication):** Tailored environments

#### Key Ideas from the Publication:

- **Causal Graph is critical :** Causal graph can be learned as input/learned.
- **Genralization:** Genralization is variation in Goal (G) with  states (S) , G ∈ S, train for a Goal G and model should peform for G`.
- **Transition Model** Unlike in Model based RL Causal Graph is levaerged to model environment and SCM ( here its a GNN) becomes proxy to enironment for drawing trajectories
- **Planning:** RL policy is planning model whose aim is to reach Goal

<img width="357" alt="image" src="https://github.com/user-attachments/assets/16cc1d15-07f6-49ae-90e0-5935369aa4c8" /> 

  - Training with delta of S’ & S as the label from the Buffer obtained from Policy as well environment 
  - Embeddings are learned to predict next State
  - Causal Graph makes it robust than Model based 
    
<img width="373" alt="image" src="https://github.com/user-attachments/assets/b92dea42-73f9-4ed8-9bd2-b57908260761" />

  - The policy aims to optimize an action-state value function can be obtained by unrolling the transition model with a horizon of H steps and discount factor 
  - Model predictive control (MPC) with random shooting which selects the first action in the fixed-horizon trajectory that has the highest action-state value






#### Key Theoretical Aspects (Main points to Pay Attention to While Reading the Publication):



#### Code from [Github]() that captures main contribution of the paper 


#### To be noted 


#### My notes on Theoretical aspects  
- < to be added>
