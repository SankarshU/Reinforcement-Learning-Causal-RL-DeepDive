# Counterfactual RL (CTRL) on Noisy CartPole  
Reproduction & Extension of NeurIPS 2020 Paper  
**“Sample-Efficient Reinforcement Learning via Counterfactual-Based Data Augmentation”**  
PDF: https://arxiv.org/pdf/2012.09092

---

## 1. Overview

This repository implements the **full CTRL pipeline** on a modified noisy CartPole system:

- Noisy CartPole environment with 11 discrete actions  
- SD dataset generation (250 trials × 20 steps)  
- BiCoGAN training (Generator, Encoder, Discriminator)  
- Counterfactual transition generation  
- Offline D3QN + CQL policy learning  
- Evaluation on **clean CartPole-v1**

The structure and methods closely follow the CTRL paper.

---

## 2. Repository Structure

```
.
├── ctrl_env.py                 # Noisy CartPole SD environment
├── ctrl_data.py                # Dataset creation + CF generation
├── ctrl_models.py              # G, E, D, Q-net architectures
├── ctrl_trainer.py             # BiCoGAN + D3QN+CQL training loops
├── ctrl_utilities.py           # Evaluation + plotting utilities
├── counterfactual_rl_expts.ipynb
├── SD_dataset_clean.pt
├── README.md
```

---

## 3. Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

Optional Conda:

```bash
conda create -n ctrl python=3.10
conda activate ctrl
pip install -r requirements.txt
```

---

## 4. SD Dataset Creation

Implements the paper’s SD dataset:

- 11 discrete actions mapped to continuous \[0,1\]  
- Noise injected in action execution  
- Noise added to observations  
- Termination checked on the clean state  
- 250 trials × 20 steps each

```python
from ctrl_data import make_SD_dataset
make_SD_dataset("SD_dataset_clean.pt")
```

---

## 5. BiCoGAN (CTRL-g) Training

```python
from ctrl_trainer import train_bicogan
from ctrl_data import CTRLTransitionDataset

ds = CTRLTransitionDataset("SD_dataset_clean.pt")
G, E, D, logs = train_bicogan(ds)
```

Implements:

- G(s, a, u) → next state
- E(s') → latent u
- Adversarial loss
- Consistency loss `E(G(s,a,u)) = u`
- Training schedule from CTRL

---

## 6. Counterfactual Transition Generation

```python
from ctrl_data import generate_cf_dataset

generate_cf_dataset(
    G, E, S_real, A_real, SP_real,
    cf_per_state=5,
    save_path="cf_dataset.pt"
)
```

For each transition:

- Encode latent `u = E(s')`
- Sample random counterfactual action `a_cf`
- Predict counterfactual next state `s'_cf = G(...)`
- Recompute reward and termination
- Optionally mix with real transitions

---

## 7. Offline RL with D3QN + CQL

Includes:

- Dueling network  
- Double Q-learning  
- LayerNorm  
- Advantage normalization  
- CQL penalty  
- Smooth L1 TD loss  
- Soft-target update  

```python
from ctrl_trainer import train_offline_d3qn

q_net, logs = train_offline_d3qn(
    S, A, R, SP, D,
    hyper,
    S_mean, S_std,
    eval_every=20
)
```

---

## 8. Evaluation on Clean CartPole-v1

```python
from ctrl_utilities import evaluate_policy

returns = evaluate_policy(q_net, S_mean, S_std, episodes=50)
print("Mean return:", returns.mean())
```

---

## 9. Analysis Tools

```python
from ctrl_utilities import (
    plot_d3qn_training,
    remove_outliers,
    diagnostic_plots
)
```

Includes:

- Histograms  
- KDE curves  
- Boxplots  
- Mean return tracking  
- Outlier removal  
- Comparison of **Real-only vs CF-augmented training**

---

## 10. Citation

```
@inproceedings{lu2020ctrl,
  title={Sample-Efficient Reinforcement Learning via Counterfactual-Based Data Augmentation},
  author={Lu, Cheryl and Tucker, George and others},
  booktitle={NeurIPS},
  year={2020}
}
```

---

## 11. License

MIT License (or choose your own)

---

## 12. Acknowledgements

This project re-implements the CTRL pipeline faithfully and adds diagnostics, visualization tools, and reproducible evaluation scripts.

