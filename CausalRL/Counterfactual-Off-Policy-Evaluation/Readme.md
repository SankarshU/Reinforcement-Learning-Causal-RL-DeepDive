### Main Summary

A diagnostic method for evaluating an offline-trained RL agent intended for real-world deployment without environmental exploration. Essentially, this involves debugging the target model (without a simulator), given the following conditions: access to 

1. observation trajectories,
2. finite MDP  and
3. discrete states. 

This method elegantly generates counterfactual trajectory samples for each observed trajectory under the target policy. It provides a method for qualitative introspection and ‘debugging’ of RL models, in settings where a domain expert could plausibly examine individual trajectories. The diagram below intuitively captures this summary for Frozen lake Gym environment.The experiment with Frozen lake is for demonstration !!! (while actual experiment is from medical domain it has been set up to be as much faithful to paper's theory and assumptions 

![image](https://github.com/user-attachments/assets/f2d3119f-9f48-4d6c-b39b-f395bb7e90cb)


**Published Forum:** ICML, 2019 ,[paper](https://arxiv.org/abs/1905.05824)  
**Experiment Setup (Publication):** Medical Domain (Sepsis management)

#### Key Ideas from the Publication:

- **Non-Identifiability Challenge:** Given a fully-specified finite Markov Decision Process (MDP), there are multiple parameterizations of a structural causal model (SCM) which are equally compatible with the transition and reward distributions.
- **Gumbel-Max SCM:** Based on the Gumbel-Max trick for sampling from discrete distributions, demonstrate that it satisfies the counterfactual stability condition.
- **Reward Decomposition:** Decompose the expected difference in reward between the policies (observed and target) into differences on observed episodes over counterfactual trajectories.

#### Key Theoretical Aspects (Main points to Pay Attention to While Reading the Publication):

- **Reward Decomposition:** As per the equation provided in the paper.
 ![image](https://github.com/user-attachments/assets/2f8c495b-c107-4c11-b697-815d27a34b1f)

- **Causal Model Assumption:** Given a causal model of the environment (in the form of an SCM; SCM for a POMDP as taken from Buesing et al. (2019), with initial state Us1 = S1, states St, and histories Ht.
  ![image](https://github.com/user-attachments/assets/2334cbbe-1b8b-4508-b96c-411aee6196d5)

- **Gumbel-Max SCM:** Drawing counterfactual trajectories, in fact, we draw probabilities for the next state under the target policy action.
![image](https://github.com/user-attachments/assets/6a48350c-8c18-43e0-8b9e-141e0383cb92)

#### My demo,Experiments with Gym

- **Frozen Lake:** Include `is_slippery=True`, States: `range(0, 15)`, Actions: `{'L', 'U', 'D', 'R'}`

- **MDP:** MDP can be constructed from below for .is_slippery=True and below probabilities for each of action

$$
a \text{ is the intended action, } a^* \text{ is the action possibility.}
$$

$$
P(r, s' \mid s, a) \text{ can be modeled with the following probabilities and grid position}
$$

$$
P(a^* = \text{‘U’} \mid a = \text{‘L’}) = \frac{1}{3} ;
P(a^* = \text{‘D’} \mid a = \text{‘L’}) = \frac{1}{3} ;
P(a^* = \text{‘L’} \mid a = \text{‘L’}) = \frac{1}{3}
$$

- **Sample results:** This illustration is for demonstration only , for details please refer [experiment code](https://github.com/SankarshU/Reinforcement-Learning-DeepDive/blob/230fb446295f6aaf2f3319de4500a12e710ed0ae/CausalRL/Codes/CF_OffPolicy_Eval/gumbel_max_scm_expt%20-%20Git/FrozenLake_Gumbel_CF_EXPT.ipynb)  
![image](https://github.com/user-attachments/assets/8f5eddce-d631-4887-88cb-6e11812c830b) ![image](https://github.com/user-attachments/assets/330ed74b-af0a-42aa-bb3a-f727b485e532)


#### Code from [Github](https://github.com/clinicalml/gumbel-max-scm/tree/master) that captures main contribution of the paper 

![image](https://github.com/user-attachments/assets/eab6d744-e110-4a0c-8531-b271662e8d41)

#### To be noted 
- True MDP Requirements: Evaluate whether a learned MDP is sufficient, and consider the need for experimental validation and an in-depth theoretical analysis.
- Limitations of Discrete States: Acknowledge the constraints imposed by discrete states.
- Future Research Directions: Identify potential areas for follow-up research, a follow up wok is [here](https://arxiv.org/abs/2111.06888).
- I used Conservative Q-Learning (CQL) for Offline Reinforcement Learning in my experiments. A dedicated folder will cover CQL


#### My notes on Theoretical aspects  
- < to be added>
