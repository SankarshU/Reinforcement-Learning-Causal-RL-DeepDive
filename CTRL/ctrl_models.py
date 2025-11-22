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

class GeneratorCartPoleMLP(torch.nn.Module):
    """
    CTRL-style MONOTONIC generator.

    Input x = [s(4), acont(1), u(udim)]
    Output: next state s' (4 dims)

    Structure:
        out = g_sf(s,acont) + g_u(u)
    where:
        g_sf is standard MLP,
        g_u is monotone in u (all weights >= 0).
    """

    def __init__(self, input_dim, output_dim=4):
        super().__init__()

        # x = [s(4), acont(1), u(udim)]
        self.sf_dim = 5                         # s + acont
        self.udim   = input_dim - self.sf_dim   # latent dimension
        self.outdim = output_dim

        # =====================================================
        # g_sf(s,acont): standard MLP
        # =====================================================
        self.sf_fc1 = torch.nn.Linear(self.sf_dim, 256)
        self.sf_fc2 = torch.nn.Linear(256, 256)
        self.sf_fc3 = torch.nn.Linear(256, output_dim)

        for m in [self.sf_fc1, self.sf_fc2, self.sf_fc3]:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

        # =====================================================
        # g_u(u): Monotonic network
        # We enforce positivity using softplus(theta) (better than exp)
        # =====================================================
        hu = 64  # hidden width for monotone u-branch

        # Raw parameters (unconstrained)
        self.theta_u1 = torch.nn.Parameter(
            torch.randn(hu, self.udim) * 0.01
        )
        self.theta_u2 = torch.nn.Parameter(
            torch.randn(output_dim, hu) * 0.01
        )
        self.bu1 = torch.nn.Parameter(torch.zeros(hu))
        self.bu2 = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # Split into [s,f,u]
        sf = x[:, :self.sf_dim]     # (B,5)
        u  = x[:, self.sf_dim:]     # (B,udim)

        # =====================================================
        # g_sf(s,acont): standard MLP
        # =====================================================
        h_sf = F.relu(self.sf_fc1(sf))
        h_sf = F.relu(self.sf_fc2(h_sf))
        out_sf = self.sf_fc3(h_sf)     # (B,4)

        # =====================================================
        # g_u(u): monotone MLP with positive weights
        # =====================================================
        # softplus ensures positivity but avoids numeric explosion
        W_u1 = F.softplus(self.theta_u1)      # (hu, udim)
        W_u2 = F.softplus(self.theta_u2)      # (4, hu)

        h_u = F.relu(F.linear(u, W_u1, self.bu1))     # (B,hu)
        out_u = F.linear(h_u, W_u2, self.bu2)         # (B,4)

        # CTRL structure
        out = out_sf + out_u
        return out




class DiscriminatorCTRL(nn.Module):
    """
    D(s, a_cont, s') → logit score
    (no sigmoid in forward)
    
    Input:  s  : (batch, 4)
            a_cont : (batch, 1)
            sp : (batch, 4)

    Architecture: 256 → 256 → 128 → 1
    """

    def __init__(self, obsdim=4):
        super().__init__()

        input_dim = obsdim + 1 + obsdim   # 4 + 1 + 4 = 9

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)      # LOGIT (not probability)

        # Xavier initialization = stable for GANs
        layers = [self.fc1, self.fc2, self.fc3, self.fc4]
        for m in layers:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, s, a_cont, sp):
        """
        no sigmoid() here (GAN/BCEWithLogitsLoss expects raw logit)
        """
        x = torch.cat([s, a_cont, sp], dim=-1)   # (batch, 9)

        h = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        h = F.leaky_relu(self.fc2(h), negative_slope=0.1)
        h = F.leaky_relu(self.fc3(h), negative_slope=0.1)

        logit = self.fc4(h)
        return logit                # raw logit (real number)


class EncoderCTRL(nn.Module):
    """
    E(s') → (s_hat, acont_hat, u_hat)

    s_hat      : previous state (4-dim)
    acont_hat  : continuous action in [0,1]
    u_hat      : latent noise [-1,1]^udim
    """
    def __init__(self, obsdim=4, udim=8):
        super().__init__()

        # === Shared trunk ===
        self.body = nn.Sequential(
            nn.Linear(obsdim, 600), nn.LeakyReLU(0.02),
            nn.Linear(600, 600),    nn.LeakyReLU(0.02),
            nn.Linear(600, 400),    nn.LeakyReLU(0.02),
            nn.Linear(400, 200),    nn.LeakyReLU(0.02),
        )

        # === Heads ===
        self.s_head = nn.Linear(200, obsdim)   # linear (unbounded)
        
        # Predict action in [0,1]
        self.acont_head = nn.Sequential(
            nn.Linear(200, 1),
            nn.Sigmoid()                        # ensures a_hat ∈ [0,1]
        )

        self.u_head = nn.Sequential(
            nn.Linear(200, udim),
            nn.Tanh()                           # u_hat ∈ [-1,1]
        )

        # === Xavier init ===
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s_next):
        h = self.body(s_next)

        s_hat = self.s_head(h)                 # previous state
        acont_hat = self.acont_head(h)         # predicted continuous action ∈ [0,1]
        u_hat = self.u_head(h)                 # latent ∈ [-1,1]

        return s_hat, acont_hat, u_hat



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
