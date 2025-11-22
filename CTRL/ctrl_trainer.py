import sys, os
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

from tqdm import tqdm   # ✅ NEW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.abspath("/Users/pratyushuppuluri/Desktop/Summer/CARL"))
from ctrl_models import GeneratorCartPoleMLP,DiscriminatorCTRL,EncoderCTRL,QNetCTRL
from ctrl_utilities import evaluate_policy,plot_losses,test_counterfactual_quality,plot_d3qn_training,remove_outliers,summary,diagnostic_plots


class BiCoGAN:
    def __init__(self, G, E, D, args):
        self.G, self.E, self.D, self.args = G, E, D, args
        self.device = next(G.parameters()).device

        # --- Optimizers ---
        self.optG  = torch.optim.Adam(G.parameters(), lr=args.pre_train_lr)
        self.optD  = torch.optim.Adam(D.parameters(), lr=args.disc_lr)
        self.optGE = torch.optim.Adam(
            list(G.parameters()) + list(E.parameters()),
            lr=args.lr, betas=(args.beta1, args.beta2)
        )

        # --- Losses ---
        self.adv = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        # Gamma schedule
        self.history = {"D": [], "Adv": [], "EFL": [], "Gamma": []}
        self.gamma = lambda t: min(args.alpha * math.exp(args.rho * t), args.phi)

    # ============================================================
    # 1. PRETRAIN GENERATOR ON REAL DYNAMICS
    #    batch = (s, a_idx, a_cont, r, sp)
    # ============================================================
    def pretrain_forward(self, batch):
        s, a_idx, a_cont, r, sp = batch
        s       = s.to(self.device)
        a_cont  = a_cont.to(self.device).unsqueeze(-1)
        sp      = sp.to(self.device)

        bsz = s.size(0)
        u = torch.zeros(bsz, self.args.udim, device=self.device)

        x = torch.cat([s, a_cont, u], dim=1)

        sp_hat = self.G(x)
        loss   = self.mse(sp_hat, sp)

        self.optG.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
        self.optG.step()

        return loss.item()

    # ============================================================
    # 2. DISCRIMINATOR UPDATE
    # ============================================================
    def train_discriminator(self, batch):
        s, a_idx, a_cont, r, sp = batch
        s       = s.to(self.device)
        sp      = sp.to(self.device)
        a_cont  = a_cont.to(self.device).unsqueeze(-1)

        bsz = s.size(0)

        # ---- REAL ----
        real_logits = self.D(s, a_cont, sp)

        # ---- FAKE ----
        u = torch.randn(bsz, self.args.udim, device=self.device)
        sp_fake = self.G(torch.cat([s, a_cont, u], dim=1)).detach()
        fake_logits = self.D(s, a_cont, sp_fake)

        # ---- ADV LOSS ----
        logits = torch.cat([real_logits, fake_logits], dim=0)
        labels = torch.cat([
            torch.ones_like(real_logits),
            torch.zeros_like(fake_logits)
        ], dim=0)

        loss = self.adv(logits, labels)

        self.optD.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
        self.optD.step()

        return loss.item()

    # ============================================================
    # 3. GENERATOR + ENCODER UPDATE
    # ============================================================
    def train_generator_encoder(self, batch, gamma, lambda_fwd=0.05):
        s, a_idx, a_cont, r, sp = batch
        s       = s.to(self.device)
        sp      = sp.to(self.device)
        a_cont  = a_cont.to(self.device).unsqueeze(-1)

        bsz = s.size(0)

        # ---------------------------------------------------------
        # 1. Encoder reconstructs (s_hat, acont_hat, u_hat)
        # ---------------------------------------------------------
        s_hat, acont_hat, u_hat = self.E(sp)

        # ---------------------------------------------------------
        # 2. Generator forward pass with encoded u_hat
        # ---------------------------------------------------------
        sp_fake = self.G(torch.cat([s, a_cont, u_hat], dim=1))

        # ---------------------------------------------------------
        # 3. Adversarial loss
        # ---------------------------------------------------------
        d_fake = self.D(s, a_cont, sp_fake)
        adv_loss = self.adv(d_fake, torch.ones_like(d_fake))

        # ---------------------------------------------------------
        # 4. EFL: (s_hat ~ s) + (acont_hat ~ a_cont)
        # ---------------------------------------------------------
        efl_s = self.mse(s_hat, s)
        efl_a = self.mse(acont_hat, a_cont)

        efl = efl_s + 0.5 * efl_a   # weight action term lightly

        # ---------------------------------------------------------
        # 5. Forward reconstruction loss (cycle)
        # ---------------------------------------------------------
        fwd_recon = self.mse(sp_fake, sp)

        # ---------------------------------------------------------
        # TOTAL LOSS
        # ---------------------------------------------------------
        total = adv_loss + gamma * efl + lambda_fwd * fwd_recon

        self.optGE.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(
            list(self.G.parameters()) + list(self.E.parameters()),
            1.0
        )
        self.optGE.step()

        return adv_loss.item(), efl.item(), fwd_recon.item()

    # ============================================================
    # 4. TRAIN LOOP (with tqdm)
    # ============================================================
    def train(self, loader):
        print("\n[Stage 1] Pretraining G...")

        # --------------------------
        # PRETRAINING WITH TQDM
        # --------------------------
        for ep in tqdm(range(self.args.pre_train_epochs), desc="Pretrain Epochs"):
            losses = []
            #batch_bar = tqdm(loader, desc=f"Pretrain {ep+1}", leave=False)
            for batch in loader:
                mse_val = self.pretrain_forward(batch)
                losses.append(mse_val)
                #batch_bar.set_postfix(MSE=f"{mse_val:.4f}")
            print(f"  Epoch {ep+1}: MSE={np.mean(losses):.6f}")

        print("\n[Stage 2] BiCoGAN training...")
        global_step = 0

        # --------------------------
        # MAIN BiCoGAN TRAINING WITH TQDM
        # --------------------------
        epoch_bar = tqdm(range(self.args.num_epochs), desc="BiCoGAN Epochs")
        for ep in epoch_bar:
            Dl, Advl, EFLl = [], [], []

            #batch_bar = tqdm(loader, desc=f"Epoch {ep+1}", leave=False)
            for batch in loader:
                g_val = self.gamma(global_step)
                global_step += 1

                d_loss = self.train_discriminator(batch)
                adv_l, efl_l, _ = self.train_generator_encoder(batch, g_val)

                Dl.append(d_loss)
                Advl.append(adv_l)
                EFLl.append(efl_l)

                # batch_bar.set_postfix(
                #     D=f"{d_loss:.3f}",
                #     Adv=f"{adv_l:.3f}",
                #     EFL=f"{efl_l:.3f}",
                #     gamma=f"{g_val:.3f}"
                # )

            # Logging
            D_mean = float(np.mean(Dl))
            Adv_mean = float(np.mean(Advl))
            EFL_mean = float(np.mean(EFLl))

            self.history["D"].append(D_mean)
            self.history["Adv"].append(Adv_mean)
            self.history["EFL"].append(EFL_mean)
            self.history["Gamma"].append(g_val)

            epoch_bar.set_postfix(
                D=f"{D_mean:.3f}",
                Adv=f"{Adv_mean:.3f}",
                EFL=f"{EFL_mean:.3f}",
                gamma=f"{g_val:.3f}"
            )

            print(
                f"Epoch {ep+1}/{self.args.num_epochs} | "
                f"D={D_mean:.4f}  Adv={Adv_mean:.4f}  "
                f"EFL={EFL_mean:.4f}  γ={g_val:.3f}"
            )


# ============================================================
# TRAINING LOOP (OFFLINE D3QN + CQL)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACTIONS = 11  # discrete actions 0..10

@dataclass
class D3QNHyperParams:
    epochs: int = 1000
    gamma: float = 0.99
    batch_size: int = 512
    lr: float = 1.5e-4
    tau: float = 0.005                 # target update rate
    alpha_cql: float = 0.02            # CQL weight
    reward_clip: Optional[float] = None
    target_clip: Optional[float] = 20.0
    max_grad_norm: Optional[float] = 1.0

class QNetCTRL(nn.Module):
    def __init__(self, state_dim: int = 4, n_actions: int = NUM_ACTIONS):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, 512)
        self.ln1 = nn.LayerNorm(512)

        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)

        self.fc3 = nn.Linear(512, 512)
        self.ln3 = nn.LayerNorm(512)

        self.fc4 = nn.Linear(512, 512)
        self.ln4 = nn.LayerNorm(512)

        # Dueling heads
        self.val = nn.Linear(512, 1)
        self.adv = nn.Linear(512, n_actions)

        # Initialization
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        nn.init.uniform_(self.val.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.adv.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.val.bias)
        nn.init.zeros_(self.adv.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, state_dim)
        return: Q-values (B, NUM_ACTIONS)
        """
        h = F.relu(self.ln1(self.fc1(s)))
        h = F.relu(self.ln2(self.fc2(h)))
        h = F.relu(self.ln3(self.fc3(h)))
        h = F.relu(self.ln4(self.fc4(h)))

        V = self.val(h)          # (B,1)
        A = self.adv(h)          # (B,NUM_ACTIONS)

        # Advantage normalization to keep Qstd ~ O(1)
        A = A / (A.std(dim=1, keepdim=True) + 1e-6)

        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ACTIONS = 11  # discrete actions 0..10


def train_offline_d3qn(
    S: torch.Tensor,
    A: torch.Tensor,
    R: torch.Tensor,
    SP: torch.Tensor,
    D: torch.Tensor,
    hyper: D3QNHyperParams,
    S_mean: torch.Tensor,
    S_std: torch.Tensor,
    eval_every: int = 20,
    label: str = "Offline D3QN+CQL",
):
    print(f"\n========== OFFLINE D3QN TRAINING: {label} ==========")

    N, state_dim = S.shape
    valist = []

    # Move to device
    S  = S.to(device).float()
    SP = SP.to(device).float()
    A  = A.to(device).long()
    R  = R.to(device).float()
    D  = D.to(device).float()

    # These must be defined elsewhere in your code:
    # - QNetCTRL
    # - NUM_ACTIONS
    # - evaluate_policy
    q_net  = QNetCTRL(state_dim=state_dim, n_actions=NUM_ACTIONS).to(device)
    tgt_net = QNetCTRL(state_dim=state_dim, n_actions=NUM_ACTIONS).to(device)
    tgt_net.load_state_dict(q_net.state_dict())

    opt = torch.optim.Adam(q_net.parameters(), lr=hyper.lr)

    total_losses, td_losses, cql_losses = [], [], []
    Q_means, Q_stds = [], []

    # --------------------------
    # EPOCH LOOP WITH TQDM
    # --------------------------
    epoch_bar = tqdm(range(hyper.epochs), desc="D3QN Epochs")
    for ep in epoch_bar:

        perm = torch.randperm(N, device=device)
        S2, A2, R2, SP2, D2 = S[perm], A[perm], R[perm], SP[perm], D[perm]

        batch_tot, batch_td, batch_cql = [], [], []

        # --------------------------
        # BATCH LOOP WITH TQDM
        # --------------------------
        # batch_bar = tqdm(
        #     range(0, N, hyper.batch_size),
        #     desc=f"Epoch {ep+1}",
        #     leave=False
        # )

        for i in range(0, N, hyper.batch_size):
            s  = S2[i:i + hyper.batch_size]
            a  = A2[i:i + hyper.batch_size]
            r  = R2[i:i + hyper.batch_size]
            sp = SP2[i:i + hyper.batch_size]
            d  = D2[i:i + hyper.batch_size]

            if s.size(0) == 0:
                continue

            # Optional reward clipping
            if hyper.reward_clip is not None:
                r = torch.clamp(r, -hyper.reward_clip, hyper.reward_clip)

            # Q(s, ·)
            q_all = q_net(s)                            # (B, NUM_ACTIONS)
            q_all = q_all - q_all.mean(dim=1, keepdim=True)

            # Q(s,a)
            q_sa = q_all.gather(1, a.unsqueeze(1))      # (B,1)

            # Double DQN target
            with torch.no_grad():
                q_next_online = q_net(sp)
                next_a = q_next_online.argmax(dim=1, keepdim=True)
                q_next = tgt_net(sp).gather(1, next_a)  # (B,1)

                y = r.unsqueeze(1) + hyper.gamma * q_next * (1.0 - d.unsqueeze(1))

                if hyper.target_clip is not None:
                    y = torch.clamp(y, -hyper.target_clip, hyper.target_clip)

            # TD error
            td_loss = F.smooth_l1_loss(q_sa, y)

            # Conservative Q-Learning term
            logsum = torch.logsumexp(q_all, dim=1, keepdim=True)  # (B,1)
            conservative = (logsum - q_sa).mean()
            cql_loss = hyper.alpha_cql * conservative

            loss = td_loss + cql_loss

            opt.zero_grad()
            loss.backward()
            if hyper.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(q_net.parameters(), hyper.max_grad_norm)
            opt.step()

            # Soft target update
            with torch.no_grad():
                for p, tp in zip(q_net.parameters(), tgt_net.parameters()):
                    tp.data.mul_(1.0 - hyper.tau).add_(hyper.tau * p.data)

            batch_tot.append(loss.item())
            batch_td.append(td_loss.item())
            batch_cql.append(cql_loss.item())

            # batch_bar.set_postfix(
            #     TD=f"{td_loss.item():.4f}",
            #     CQL=f"{cql_loss.item():.4f}",
            # )

        # Epoch logging
        total_losses.append(float(np.mean(batch_tot)))
        td_losses.append(float(np.mean(batch_td)))
        cql_losses.append(float(np.mean(batch_cql)))

        with torch.no_grad():
            Qa = q_net(S[: min(512, N)])
            Q_means.append(Qa.mean().item())
            Q_stds.append(Qa.std().item())

        # Periodic eval on clean CartPole
        if (ep + 1) % eval_every == 0 or ep == 0:
            print(
                f"Epoch {ep+1}/{hyper.epochs} | "
                f"TD={td_losses[-1]:.4f} | CQL={cql_losses[-1]:.4f} | "
                f"Total={total_losses[-1]:.4f} | "
                f"Qmean={Q_means[-1]:.3f} | Qstd={Q_stds[-1]:.3f}"
            )

            eval_returns = evaluate_policy(q_net, S_mean, S_std, episodes=20)
            print(
                f"   ▶ Eval Return Mean = {eval_returns.mean():.2f}, "
                f"Std = {eval_returns.std():.2f}"
            )
            valist.append(
                (np.round(eval_returns.mean(), 3),
                 np.round(eval_returns.std(), 3))
            )

    return q_net, total_losses, td_losses, cql_losses, Q_means, Q_stds, valist
