import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACTIONS = 11  # discrete actions 0..10


class CTRL_CartPoleSD_CLEAN(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, seed=None):
        super().__init__()

        self.action_space = Discrete(11)
        self.action_set = np.linspace(0.0, 1.0, 11)

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = 1.1
        self.length = 0.5
        self.polemass_length = 0.05
        self.tau = 0.02
        self.max_force = 10.0

        # termination thresholds
        self.theta_threshold_radians = 12 * np.pi / 180
        self.x_threshold = 2.4

        high = np.array([
            4.8,
            np.finfo(np.float32).max,
            0.418,
            np.finfo(np.float32).max,
        ], dtype=np.float32)

        self.observation_space = Box(-high, high, dtype=np.float32)

        self.np_random = np.random.default_rng(seed)
        self.state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state.astype(np.float32), {}

    # ----------------------------------------
    # FIXED STEP: termination on CLEAN state
    # ----------------------------------------
    def step(self, a_noisy):
        # (1) convert noisy a in [0,1]
        force = (2.0 * a_noisy - 1.0) * self.max_force

        # (2) unpack
        x, x_dot, theta, theta_dot = self.state

        costheta, sintheta = np.cos(theta), np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4/3 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # (3) integrate
        x_n = x + self.tau * x_dot
        x_dot_n = x_dot + self.tau * xacc
        theta_n = theta + self.tau * theta_dot
        theta_dot_n = theta_dot + self.tau * thetaacc

        # CLEAN next state
        sp_clean = np.array([x_n, x_dot_n, theta_n, theta_dot_n])

        # (4) termination BEFORE noise
        terminated = (
            (sp_clean[0] < -self.x_threshold) or
            (sp_clean[0] >  self.x_threshold) or
            (sp_clean[2] < -self.theta_threshold_radians) or
            (sp_clean[2] >  self.theta_threshold_radians)
        )

        # (5) ADD NOISE
        sp_noisy = sp_clean + 0.05 * self.np_random.normal(size=4)

        # only clip after termination check
        sp_noisy[0] = np.clip(sp_noisy[0], -4.8, 4.8)
        sp_noisy[2] = np.clip(sp_noisy[2], -0.418, 0.418)

        self.state = sp_noisy
        return sp_noisy.astype(np.float32), 1.0, terminated, False, {}
def make_SD_dataset(num_eps=250, horizon=200, seed=0, save_path="SD_dataset_clean.pt"):

    env = CTRL_CartPoleSD_CLEAN(seed=seed)
    rng = np.random.default_rng(seed)

    S = []
    A = []
    Acont = []
    R = []
    SP = []

    for ep in range(num_eps):
        s, _ = env.reset()

        for t in range(horizon):

            # 1. choose action idx
            a_idx = rng.integers(0, 11)

            # 2. generate same noisy a used in dataset
            a_val = a_idx / 10.0
            a_noisy = a_val + 0.05 * rng.normal()

            # 3. feed a_noisy into step()
            sp, r, done, trunc, _ = env.step(a_noisy)

            S.append(s)
            A.append(a_idx)
            Acont.append(a_noisy)
            R.append(r)
            SP.append(sp)

            s = sp
            if done:
                break

    dataset = {
        "s": torch.tensor(np.array(S), dtype=torch.float32),
        "a": torch.tensor(np.array(A), dtype=torch.long),
        "acont": torch.tensor(np.array(Acont), dtype=torch.float32),
        "r": torch.tensor(np.array(R), dtype=torch.float32),
        "sp": torch.tensor(np.array(SP), dtype=torch.float32)
    }

    torch.save(dataset, save_path)

    print(f"Saved CLEAN SD dataset → {save_path}")
    print({k: v.shape for k, v in dataset.items()})

    return dataset


class CTRLTransitionDataset(Dataset):
    def __init__(self, data):
        self.s      = data["s"].reshape(-1, 4).float()
        self.a_idx  = data["a"].reshape(-1).long()
        self.a_cont = data["acont"].reshape(-1).float()
        self.r      = data["r"].reshape(-1).float()
        self.sp     = data["sp"].reshape(-1, 4).float()

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        return (
            self.s[idx],        # (4,)
            self.a_idx[idx],    # int
            self.a_cont[idx],   # float (continuous)
            self.r[idx],        # float
            self.sp[idx]        # (4,)
        )

# ============================================================
# COUNTERFACTUAL DATA GENERATION (using trained G, E)
# ============================================================
def generate_cf_dataset(
        G: nn.Module,
        E: nn.Module,
        S_raw: torch.Tensor,
        A_raw: torch.Tensor,
        SP_raw: torch.Tensor,
        cf_k: int = 5,
        device: str = "cpu",
):
    """
    Generate a CF dataset with SAME keys as SD_dataset_clean.pt:
        {'s','a','r','sp'}

    Uses:
      - s (current state)
      - latent U = E(sp)
      - random new discrete actions
      - G(s, a_cont, U) -> sp_cf
      - termination and reward defined from sp_cf (clean thresholds)
    """
    G.eval()
    E.eval()

    N = S_raw.shape[0]

    S_cf = []
    A_cf = []
    R_cf = []
    SP_cf = []

    with torch.no_grad():
        s = S_raw.to(device)      # (N,4)
        sp = SP_raw.to(device)    # (N,4)

        # encode latent U from next state
        _, _, U = E(sp)           # (N, latent_dim)

        for _ in range(cf_k):
            # random discrete CF action
            a_idx = torch.randint(0, NUM_ACTIONS, (N,), device=device)
            a_cont = a_idx.float() / 10.0

            # generator input: [s, a_cont, U]
            x = torch.cat([s, a_cont.unsqueeze(1), U], dim=1)

            sp_hat = G(x)         # (N,4)

            # physical clipping (same as SD dataset)
            sp_hat[:, 0] = torch.clamp(sp_hat[:, 0], -4.8, 4.8)      # x
            sp_hat[:, 2] = torch.clamp(sp_hat[:, 2], -0.418, 0.418)  # θ

            # termination (using CLEAN thresholds)
            x_pos = sp_hat[:, 0]
            th    = sp_hat[:, 2]

            done = (
                (x_pos < -2.4) | (x_pos > 2.4) |
                (th < -0.2095) | (th > 0.2095)
            ).float()

            reward = 1.0 - done

            S_cf.append(s)
            A_cf.append(a_idx)
            SP_cf.append(sp_hat)
            R_cf.append(reward)

    # concatenate final CF dataset
    S_cf = torch.cat(S_cf, dim=0)
    A_cf = torch.cat(A_cf, dim=0)
    R_cf = torch.cat(R_cf, dim=0)
    SP_cf = torch.cat(SP_cf, dim=0)

    print("Generated CF transitions:", S_cf.shape[0])

    # return dictionary COMPATIBLE with your D3QN loader
    return {
        "s": S_cf.cpu(),
        "a": A_cf.cpu(),
        "r": R_cf.cpu(),
        "sp": SP_cf.cpu()
    }


def build_training_buffer(
    real_data: dict,
    use_cf: bool = False,
    G: Optional[nn.Module] = None,
    E: Optional[nn.Module] = None,
    cf_k: int = 5,
):
    """
    Returns *raw* (unnormalized) S_raw, A, R, SP_raw

    If use_cf=False:
        just returns real dataset.

    If use_cf=True:
        generates cf_k counterfactuals per real transition using BiCoGAN
        and concatenates with real data.
    """
    S_real = real_data["s"].float().reshape(-1, 4)
    A_real = real_data["a"].long().reshape(-1)
    R_real = real_data["r"].float().reshape(-1)
    SP_real = real_data["sp"].float().reshape(-1, 4)

    if not use_cf:
        print("Using REAL-ONLY dataset.")
        return S_real, A_real, R_real, SP_real

    if G is None or E is None:
        raise ValueError("USE_CF=True but G/E are None. Load BiCoGAN models first.")

    print(f"Generating {cf_k} CF transitions per real transition...")
    cf_data = generate_cf_dataset(
        G, E,
        S_real, A_real, SP_real,
        cf_k=cf_k,
        device=device
    )

    S_cf = cf_data["s"]
    A_cf = cf_data["a"]
    R_cf = cf_data["r"]
    SP_cf = cf_data["sp"]

    S_all  = torch.cat([S_real, S_cf], dim=0)
    A_all  = torch.cat([A_real, A_cf], dim=0)
    R_all  = torch.cat([R_real, R_cf], dim=0)
    SP_all = torch.cat([SP_real, SP_cf], dim=0)

    print("Total transitions (real + CF):", S_all.shape[0])

    return S_all, A_all, R_all, SP_all


