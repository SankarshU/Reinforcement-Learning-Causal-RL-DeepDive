# Reinforcement-Learning-DeepDive
Look at the RL paradigm at a fundamental level with relevant notes, courses, materials, and experiment notebooks, digging into RL a level deeper.

#### RL and Deep RL have a wealth of material available; this is one such attempt to delve deeply into the subject to help study and understand state-of-the-art research
- The following will be covered:
  - RL & Deep RL [[fundamentals](https://github.com/SankarshU/Reinforcement-Learning-Causal-RL-DeepDive/blob/2d9192dde95a1002dead4d40656e168cd9906b4e/Basics/Introduction.md)]:
    - Concepts and illustrations from the following courses, additional experiments beyond assignments:
      - [Berkeley Deep RL][[link](http://rll.berkeley.edu/deeprlcourse/)]
      - [Practical RL][[link](https://github.com/yandexdataschool/Practical_RL)]
  - Causal Reinforcement Learning research [exploration](https://github.com/SankarshU/Reinforcement-Learning-DeepDive/tree/71c25d91abea2bc68876c736fae8632b5143061a/CausalRL):
    - Summarizing or deep diving into research
  - Deep RL literature deep dive to explore the developments in this field-[current-research](https://github.com/SankarshU/Reinforcement-Learning-DeepDive/blob/c6bd0d27c9566eb535946a8243cfb0e765a20c72/Latest-Research/Readme.md):

- Other Interesting References (will be added continually)
  - Grokking Deep RL [Notebooks](https://github.com/mimoralea/gdrl)
  - MIT 6.S191: Reinforcement Learning [link](https://www.youtube.com/watch?v=8JVRbHAVCws)

- Must read/not to be missed
  - Reinforcement Learning, An Introduction By Richard S. Sutton and Andrew G. Barto


## Quickstart (CartPole CTRL experiments)

### Setup
1. Create/activate a Python 3.10+ venv and install deps:
   ```bash
   pip install torch gymnasium numpy matplotlib seaborn tqdm scikit-learn pandas
   ```
2. From repo root, create the SD dataset:
   ```bash
   python scripts/run_ctrl.py dataset --episodes 250 --horizon 200 --output data/SD_dataset_clean.pt
   ```

### Train CTRL components
- Train BiCoGAN (for counterfactual generation):
  ```bash
  python scripts/run_ctrl.py train-bicogan --dataset-path data/SD_dataset_clean.pt --output-dir results/cartpole/bicogan
  ```
- Train offline D3QN+CQL on real-only:
  ```bash
  python scripts/run_ctrl.py train-d3qn --dataset-path data/SD_dataset_clean.pt --output-dir results/cartpole/d3qn_real
  ```
- Train offline D3QN+CQL with counterfactual augmentation:
  ```bash
  python scripts/run_ctrl.py train-d3qn --dataset-path data/SD_dataset_clean.pt --use-cf --cf-k 1 --bicogan-dir results/cartpole/bicogan --output-dir results/cartpole/d3qn_cf
  ```

### Alternative baselines
- Soft Actor-Critic (offline):
  ```bash
  python scripts/run_alt_algos.py sac --dataset-path data/SD_dataset_clean.pt --output-dir results/cartpole/sac
  ```
- Rainbow DQN (C51, offline):
  ```bash
  python scripts/run_alt_algos.py rainbow --dataset-path data/SD_dataset_clean.pt --output-dir results/cartpole/rainbow
  ```

### Demo / recording
Render or record any trained policy (SAC/Rainbow/D3QN) on CartPole:
```bash
python scripts/demo_policy.py --algo sac --model-path results/cartpole/sac/actor.pt --dataset-path data/SD_dataset_clean.pt --episodes 1
# or record video frames:
python scripts/demo_policy.py --algo rainbow --model-path results/cartpole/rainbow/q_net.pt --dataset-path data/SD_dataset_clean.pt --episodes 2 --record-dir results/cartpole/rainbow/videos
```
