# Fundamental RL Equations

1. Both policy and value-based methods rely on the expectation of reward. Naturally thus, reward is a significant aspect.
2. This expectation is key to RL, either done with Monte Carlo sampling or estimating using deep neural networks any which way analytically in scenarios applicable .
3. All algorithms in the below chart, especially value-based ones, revolve around addressing this expectation.
4. In a way, RL/Deep RL involves a lot of approximations.
5. With neural networks, since we use complex functions to estimate either estimate or make it  of the loss  such as value( state,state-action) , reward expection or auxillary like advantage  thus  derivatives over expecation , loss determination are key aspects . All cleverness goes into enabling the same.
6. Whether as a practitioner or a researcher, knowing these equations will be key to understanding DRL at fundamental level.
7. Bellman Equations , Expectations remain foundational aspect 

### Some nuts and bolts of Deep RL , details in the correspning sections of that paper summry & demo
- Value function can be paramterized by NN instead of Value estimation using Bellman Equations 
- Many times we may just need gradients , log deravative trick is often used (Policy gradients !!!) 
- KL divergence plays a key roles as we are dealing with probability distributions and policy drifts can be contorlled (TRPO,PPO)
- For loss required by a NN , as in supervised setting Targets are also estimated ( with a Target Nn !!! ( DDQN etc) )
  
### Discounted  and Expected Return that are maximized in any RL Setting  

$$
\Large
\begin{aligned}
R_t \triangleq \sum_{i=t}^{\infty} \gamma^{i} r_i \\
& J\left(\theta\right) \triangleq E_{{{s_t \text{tilda} \mathrm{~\ \rho} }^{\mathrm{\pi}_\theta},a}_t\mathrm{~\ } \mathrm{\pi}_\theta}[R_t]
\end{aligned}
$$



### Value Function

$$
\Large
\begin{aligned}
v_\pi(s) &= E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \mid S_t = s] \\
&= E[r + \gamma E[R_{t+1} \mid S_{t+1} = s']] \\
&= \sum_{a} \textcolor{green}{\pi(a \mid s)} \sum_{r, s'} \textcolor{red}{p(r, s' \mid s, a)} (r + \gamma v_\pi(s'))
\end{aligned}
$$

### Action-Value Function

$$
\Large
\begin{aligned}
Q_\pi(s, a) &= E[R_t \mid s, a] \\
&= E[r + \gamma E[R_{t+1} \mid S_{t+1} = s']] \\
&= \sum_{r, s'} \textcolor{red}{p(r, s' \mid s, a)} (r + \gamma v_\pi(s'))
\end{aligned}
$$

### Relation between Value and Action-Value

$$
\Large
\begin{aligned}
v_\pi(s) &= \sum_{a} \textcolor{green}{\pi(a \mid s)} Q_\pi(s, a)
\end{aligned}
$$

### Action-Value in a Recurrent Form

$$
\Large
\begin{aligned}
Q_\pi(s, a) &= \sum_{r, s'} \textcolor{red}{p(r, s' \mid s, a)} \left( r + \gamma \sum_{a'} \textcolor{green}{\pi(a' \mid s')} Q_\pi(s', a') \right)
\end{aligned}
$$

### Legend

$$
\begin{aligned}
\textcolor{red}{p(r, s' \mid s, a)} &\text{ : Transition probabilities due to the environment, often unknown.} \\
\textcolor{green}{\pi(a \mid s)} &\text{ : Policy probabilities due to the agent's policy.}
\end{aligned}
$$

![image](https://github.com/user-attachments/assets/4f10af22-9d21-42a2-9378-99ec16e31a4e)

