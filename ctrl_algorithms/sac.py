from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader

from ctrl_algorithms.data import OfflineDataset, SACDataset, load_ctrl_dataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


class Actor(nn.Module):
    """Squashed Gaussian policy producing actions in [0,1]."""

    def __init__(self, state_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action mean and log std for given state (batch, state_dim)."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))  # bound before scaling
        log_std = torch.clamp(self.log_std(x), -5.0, 2.0)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized action sample and log-prob. Action ∈ [0,1]."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = 0.5 * (y_t + 1.0)  # map [-1,1] -> [0,1]

        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob -= math.log(2.0)  # scale adjustment for [0,1]
        log_prob = log_prob.sum(dim=1)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean) in [0,1] for evaluation."""
        mean, _ = self.forward(state)
        return 0.5 * (torch.tanh(mean) + 1.0)


class Critic(nn.Module):
    """Twin Q-network for SAC."""

    def __init__(self, state_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values for state-action pairs. Shapes: state (B,4), action (B,1)."""
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


@dataclass
class SACConfig:
    dataset_path: Path
    output_dir: Path = Path("results/cartpole/sac")
    seed: int = 0
    epochs: int = 400
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float = -0.5
    eval_every: int = 50
    eval_episodes: int = 50


def _prepare_loader(data: OfflineDataset, batch_size: int) -> DataLoader:
    ds = SACDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def evaluate_sac_policy(
    actor: Actor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    episodes: int = 50,
    seed: int = 0,
) -> List[float]:
    """Evaluate SAC policy on clean CartPole-v1 using deterministic actions."""
    import gymnasium as gym

    actor.eval()
    device = next(actor.parameters()).device
    returns: List[float] = []
    for ep in range(episodes):
        env = gym.make("CartPole-v1")
        s_raw, _ = env.reset(seed=seed + ep)
        s = torch.tensor(s_raw, dtype=torch.float32, device=device)
        s = (s - state_mean[0].to(device)) / state_std[0].to(device)
        done = False
        trunc = False
        total_r = 0.0
        while not (done or trunc):
            with torch.no_grad():
                a_cont = actor.deterministic(s.unsqueeze(0)).squeeze(0)
            force = (2.0 * a_cont - 1.0) * 10.0
            a_bin = 1 if force.item() > 0 else 0
            sp_raw, r, done, trunc, _ = env.step(a_bin)
            total_r += float(r)
            s = torch.tensor(sp_raw, dtype=torch.float32, device=device)
            s = (s - state_mean[0].to(device)) / state_std[0].to(device)
        returns.append(total_r)
        env.close()
    return returns


def train_sac_offline(cfg: SACConfig) -> Dict[str, Any]:
    """Train SAC on the offline CTRL dataset."""
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_ctrl_dataset(cfg.dataset_path, device=device)
    loader = _prepare_loader(data, cfg.batch_size)

    actor = Actor().to(device)
    critic = Critic().to(device)
    critic_target = Critic().to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)

    if cfg.auto_alpha:
        log_alpha = torch.tensor(math.log(cfg.alpha), device=device, requires_grad=True)
        alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.lr)
    else:
        log_alpha = torch.tensor(math.log(cfg.alpha), device=device)
        alpha_opt = None

    metrics: Dict[str, List[float]] = {
        "critic_loss": [],
        "actor_loss": [],
        "alpha": [],
    }

    for ep in range(cfg.epochs):
        epoch_c_losses: List[float] = []
        epoch_a_losses: List[float] = []
        for batch in loader:
            s, a_cont, r, sp, d = [t.to(device) for t in batch]
            a_cont = a_cont.unsqueeze(1)
            r = r.unsqueeze(1)
            d = d.unsqueeze(1)

            with torch.no_grad():
                sp_action, sp_logp = actor.sample(sp)
                q1_target, q2_target = critic_target(sp, sp_action)
                q_target = torch.min(q1_target, q2_target) - log_alpha.exp() * sp_logp.unsqueeze(1)
                y = r + cfg.gamma * (1.0 - d) * q_target
                y = torch.clamp(y, -10.0, 500.0)

            q1, q2 = critic(s, a_cont)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            critic_opt.step()

            new_action, logp = actor.sample(s)
            q1_pi, q2_pi = critic(s, new_action)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (log_alpha.exp() * logp - q_pi).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            actor_opt.step()

            if cfg.auto_alpha and alpha_opt is not None:
                alpha_loss = -(log_alpha * (logp + cfg.target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

            with torch.no_grad():
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)

            epoch_c_losses.append(critic_loss.item())
            epoch_a_losses.append(actor_loss.item())

        metrics["critic_loss"].append(float(sum(epoch_c_losses) / len(epoch_c_losses)))
        metrics["actor_loss"].append(float(sum(epoch_a_losses) / len(epoch_a_losses)))
        metrics["alpha"].append(float(log_alpha.exp().item()))

        if (ep + 1) % cfg.eval_every == 0 or ep == 0:
            returns = evaluate_sac_policy(
                actor, data.state_mean, data.state_std, episodes=cfg.eval_episodes, seed=cfg.seed
            )
            metrics.setdefault("eval_returns", []).append(
                {"epoch": ep + 1, "mean": float(sum(returns) / len(returns))}
            )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), cfg.output_dir / "actor.pt")
    torch.save(critic.state_dict(), cfg.output_dir / "critic.pt")
    torch.save(critic_target.state_dict(), cfg.output_dir / "critic_target.pt")
    _save_json(asdict(cfg), cfg.output_dir / "config.json")
    _save_json(
        {
            **metrics,
            "state_mean": data.state_mean.cpu().tolist(),
            "state_std": data.state_std.cpu().tolist(),
        },
        cfg.output_dir / "metrics.json",
    )

    return metrics
