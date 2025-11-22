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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACTIONS = 11  # discrete actions 0..10


# ============================================================
# EVALUATION ON CLEAN CARTPOLE-V1
# ============================================================
def evaluate_policy(
    q_net: nn.Module,
    S_mean: torch.Tensor,
    S_std: torch.Tensor,
    episodes: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """
    Evaluate the learned Q-network on clean Gym CartPole-v1.
    Mapping:
        idx ∈ {0..10} → a_cont ∈ [0,1] → force ∈ [-10,10]
        force > 0 ⇒ env action 1, else 0
    """
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    q_net.eval()
    returns = []

    S_mean = S_mean.to(device)
    S_std = S_std.to(device)

    for ep in range(episodes):
        s_raw, _ = env.reset(seed=seed + ep)
        s = torch.tensor(s_raw, dtype=torch.float32, device=device)
        s = (s - S_mean[0]) / S_std[0]

        done = False
        trunc = False
        total_r = 0.0

        while not (done or trunc):
            with torch.no_grad():
                q_vals = q_net(s.unsqueeze(0))       # (1, NUM_ACTIONS)
                a_idx = q_vals.argmax(dim=1).item()  # 0..10

            a_cont = a_idx / 10.0                    # ∈ [0,1]
            force = (2.0 * a_cont - 1.0) * 10.0      # [-10, 10]
            a_bin = 1 if force > 0 else 0            # CartPole-v1 expects 0/1

            sp_raw, r, done, trunc, _ = env.step(a_bin)
            total_r += r

            s = torch.tensor(sp_raw, dtype=torch.float32, device=device)
            s = (s - S_mean[0]) / S_std[0]

        returns.append(total_r)

    env.close()
    return np.array(returns, dtype=np.float32)


def plot_losses(trainer):
    hist = trainer.history

    epochs = np.arange(1, len(hist["D"])+1)

    plt.figure(figsize=(14,8))

    # -------------------------------
    # 1. Discriminator Loss
    # -------------------------------
    plt.subplot(2,2,1)
    plt.plot(epochs, hist["D"], label="D-loss", color="red")
    plt.plot(epochs, hist["Adv"], label="Adv-loss (G)", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Discriminator Loss")
    plt.grid(True)

    # -------------------------------
    # 2. Generator Adversarial Loss
    # -------------------------------
    plt.subplot(2,2,2)
    plt.plot(epochs, hist["Adv"], label="Adv-loss (G)", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Generator Adversarial Loss")
    plt.grid(True)

    # -------------------------------
    # 3. Encoder Forward Loss
    # -------------------------------
    plt.subplot(2,2,3)
    plt.plot(epochs, hist["EFL"], label="EFL (encoder forward loss)", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Encoder Consistency Loss")
    plt.grid(True)

    # -------------------------------
    # 4. Gamma schedule
    # -------------------------------
    plt.subplot(2,2,4)
    plt.plot(epochs, hist["Gamma"], label="Gamma (EFL weight)", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("γ(t)")
    plt.title("Gamma Schedule Over Training")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def test_counterfactual_quality(G, E, loader, device):
    """
    Tests:
        • how well E reconstructs (s, a_cont)
        • how well G reconstructs sp using u_hat
    """
    G.eval()
    E.eval()

    batch = next(iter(loader))

    # unpack batch
    s, a_idx, a_cont, r, sp = batch
    s      = s.to(device)
    a_cont = a_cont.to(device).unsqueeze(-1)   # (B,1)
    sp     = sp.to(device)

    # -------------------------------------------------------
    # ENCODER: E(s') -> (s_hat, a_cont_hat, u_hat)
    # -------------------------------------------------------
    s_hat, acont_hat, u_hat = E(sp)

    # -------------------------------------------------------
    # GENERATOR RECONSTRUCTION:
    # sp_rec = G([s, a_cont, u_hat])
    # -------------------------------------------------------
    x = torch.cat([s, a_cont, u_hat], dim=1)
    sp_rec = G(x)

    # -------------------------------------------------------
    # Compute diagnostics
    # -------------------------------------------------------
    mse_sp = F.mse_loss(sp_rec, sp).item()
    mse_s  = F.mse_loss(s_hat, s).item()
    mse_a  = F.mse_loss(acont_hat, a_cont).item()

    print("\n===== Counterfactual Quality Check =====")
    print(f"Reconstruction of next-state  sp_hat vs sp: {mse_sp:.6f}")
    print(f"Encoder reconstruction        s_hat  vs s : {mse_s:.6f}")
    print(f"Encoder action prediction     a_hat  vs a : {mse_a:.6f}\n")

def plot_d3qn_training(
    total_losses,
    td_losses,
    cql_losses,
    Q_means,
    Q_stds,
    title: str = "Offline D3QN+CQL Training",
):
    x = np.arange(1, len(total_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(x, total_losses, label="Total")
    plt.plot(x, td_losses, label="TD")
    plt.plot(x, cql_losses, label="CQL")
    plt.title("Losses")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(x, Q_means, label="Q mean")
    plt.plot(x, Q_stds, label="Q std")
    plt.title("Q Statistics")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(x, total_losses, label="Total")
    plt.plot(x, Q_stds, label="Q std")
    plt.title("Loss vs Qstd")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------
def remove_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data >= lower) & (data <= upper)]



# --------------------------------------------------------
# 3. PRINT SUMMARY STATS
# --------------------------------------------------------
def summary(name, arr):
    print(f"\n=== {name} ===")
    print(f"Count: {len(arr)}")
    print(f"Mean : {np.mean(arr):.3f}")
    print(f"Std  : {np.std(arr):.3f}")
    print(f"Min  : {np.min(arr):.3f}")
    print(f"Max  : {np.max(arr):.3f}")

def diagnostic_plots(real_means_clean,cf_means_clean,real_means,cf_means):
    # --------------------------------------------------------
    # 4. PLOTS — HISTOGRAMS OF MEANS
    # --------------------------------------------------------
    plt.figure(figsize=(12,5))
    sns.histplot(real_means_clean, kde=True, color='blue', label='Real', bins=10, alpha=0.5)
    sns.histplot(cf_means_clean,   kde=True, color='red',  label='CF',   bins=10, alpha=0.5)
    plt.title("Distribution of Mean Returns")
    plt.xlabel("Mean Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    # --------------------------------------------------------
    # 5. BOXPLOTS FOR MEANS
    # --------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.boxplot([real_means_clean, cf_means_clean], labels=["Real", "CF"])
    plt.title("Boxplot: Mean Returns (Outliers Removed)")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.show()
    
    
    # --------------------------------------------------------
    # 6. BOXPLOTS FOR VARIANCES
    # --------------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.boxplot([real_vars_clean, cf_vars_clean], labels=["Real", "CF"])
    plt.title("Boxplot: Variance (Std²) of Returns")
    plt.ylabel("Variance")
    plt.grid(True)
    plt.show()
    
    
    # --------------------------------------------------------
    # 7. OVERLAYED MEAN CURVES
    # --------------------------------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(real_means, marker='o', label="Real Means")
    plt.plot(cf_means, marker='x', label="CF Means")
    plt.title("Mean Return per Evaluation Iteration")
    plt.xlabel("Evaluation Index")
    plt.ylabel("Mean Return")
    plt.legend()
    plt.grid(True)
    plt.show()    
