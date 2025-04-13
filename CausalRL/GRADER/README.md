### Main Summary

# GRADER: Goal Conditioning RL for Generalization

A high-level framework for **generalizing goal-conditioned reinforcement learning** by leveraging **causal graph estimation** and model-based planning.


## 🔁 Overall Framework


- Iterative learning framework
- Improves generalization and robustness via causal modeling


GRADER is a itertiave learning framework  that learns Causal graph from sampled buffer, learns transition model , transition model acts as proxy to environment
which helps drawing trajectories ( N time steps) and Trajectories are used in planning to learn the policy, planning aim is to reach towards goal

 <img width="324" alt="image" src="https://github.com/user-attachments/assets/4d14cc7d-44b5-4538-8323-5d84ecab937c" />

- This method application requires problem to be defined as  graph ( for eg: KPI is in a telecommunication network) & goals can be intents on KPIs that a polciy should achieve, in Goal conditional set up the paper observes RL agorithms such as SAC perform poorely in genralizing to different goals , i should be noted that invariance captured Causal graph and SCM that models it  ,  if graphical structre  applicable to a problem domian this work can be leveraged     

- It also requires to understand one of environments on which experiments are conducted , which will be briefly described below 

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



# Environment 

<img width="977" alt="image" src="https://github.com/user-attachments/assets/d6235973-9c3e-438b-9ebc-c0a42c4dbc2e" />

## 🧪 Chemistry Environment (Causal Reasoning Benchmark)

The **Chemistry environment** is designed to evaluate causal generalization in structured RL tasks by simulating color-based interactions over objects governed by an underlying **causal graph**.

### Environment Overview

- **Objects**: 10 uniquely identifiable nodes (e.g., blocks), each with a discrete color.
- **Causal Graph**: Controls how the color of one object probabilistically influences others.
- **Action Space**: At each timestep, the agent **changes the color of one node**.
- **Goal**: Match a **target color configuration** across all nodes.

### Causal Mechanics

- Transitions follow **Conditional Probability Tables (CPTs)** defined by the graph.
- **Interventions** (e.g., forcing a node's color) sever its causal links from parent nodes.
- Graph structures (e.g., chain, collider) affect how color changes propagate.

### Evaluation Settings

- **Spuriousness**: All nodes are assigned the **same target color**, testing if the model relies on true causality or superficial patterns.
- **No Composition Setting**: Unlike other benchmarks, Chemistry does not test compositional generalization.

> This environment allows for flexible testing of causal models under varying **graph complexities**, **stochasticity**, and **number of objects/colors** — making it ideal for benchmarking causal reasoning in RL.

📄 *Referenced from:*  
N. R. Ke et al., [Systematic Evaluation of Causal Discovery in Visual Model-Based RL (arXiv:2107.00848)](https://arxiv.org/abs/2107.00848)


---

##  Key Ideas from RL Planning

### Value Function

$$
Q(s^t, a^t) = \mathbb{E} \left[ \sum_{t'=0}^H \gamma^{t'} r(s^{t'+t}, a^{t'+t}) \mid s^t, a^t \right]
$$

### Policy

$$
\hat{\pi}(s^t) = \arg\max_{a^t \in \mathcal{A}} Q^G_\theta(s^t, a^t)
$$



- Unroll the learned transition model for \( H \) steps into the future
- Use **Model Predictive Control (MPC)** with **random shooting**
- Selects the first action in the best-scoring trajectory (in terms of goal-conditioned Q-value)

---

##  Reference

**Wenhao Ding, et al.**  
*Generalizing Goal-Conditioned Reinforcement Learning with Variational Causal Reasoning*, [NeurIPS 2022](https://neurips.cc/Conferences/2022)

---

##  Components Summary

| Component                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Buffer**               | Stores transitions from policy & environment                                |
| **Causal Graph Estimation** | Learns causal structure of variables using GNNs                             |
| **Transition Model**     | Predicts next state embedding using GRU + causal parents                    |
| **Planning-based Policy**| Uses MPC and transition model to select goal-reaching actions               |


<img width="1006" alt="image" src="https://github.com/user-attachments/assets/36da4c64-31e8-4442-ba0e-3c206a6dd7dc" />


#### Key Theoretical Aspects (Main points to Pay Attention to While Reading the Publication):

- We will expand on how Graph is used as Latent
  ##  Latent Graph as a Variable in ELBO

In GRADER, the causal graph `G` is treated as a **latent variable** that governs transitions. This setup allows the model to reason about how objects interact causally in the environment.

---

### 1. **Generative Factorization**

The joint distribution over the next state and the graph is factorized as:

$$
p(s', G \mid s, a) = p(G \mid s, a) \cdot p(s' \mid s, a, G)
$$

 *Interpretation:*  
The model first samples a causal graph `G`, then generates the next state `s'` conditioned on both the current state-action pair and the sampled graph.

---

### 2. **Variational Posterior**

Since the graph `G` is unobserved, GRADER introduces a **variational posterior** to approximate its distribution:

$$
q(G \mid s, a, s')
$$

 *Interpretation:*  
This is the inference model. Given a transition `(s, a, s')`, it estimates a distribution over plausible causal graphs that could have caused this transition.

---

### 3. **Evidence Lower Bound (ELBO)**

GRADER then optimizes the ELBO:

$$
\log p(s' \mid s, a) \geq \mathbb{E}_{q(G \mid s, a, s')} \left[ \log p(s' \mid s, a, G) \right] - \mathrm{KL} \left[ q(G \mid s, a, s') \| p(G \mid s, a) \right]
$$

 *Interpretation:*  
- The first term maximizes how well the model predicts `s'` using the inferred graph `G`.  
- The second term penalizes the divergence between the inferred graph and the prior belief about graphs.  
- Together, this encourages learning **useful and generalizable causal structures** that explain transitions well.

---

>  This approach enables GRADER to combine the benefits of **model-based RL** and **causal inference**, resulting in improved generalization and modularity.

TBE


#### Code from [Github]() that captures main contribution of the paper 


#### To be noted 


#### My notes on Theoretical aspects  
- < to be added>
