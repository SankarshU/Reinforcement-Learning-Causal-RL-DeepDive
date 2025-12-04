from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ctrl_algorithms.data import OfflineDataset, RainbowDataset, load_ctrl_dataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


class RainbowNet(nn.Module):
    """Dueling C51 network for discrete actions."""

    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 11,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, n_atoms)

        hidden = 256
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.adv = nn.Linear(hidden, n_actions * n_atoms)
        self.val = nn.Linear(hidden, n_atoms)

    def _dist(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        adv_logits = self.adv(h).view(-1, self.n_actions, self.n_atoms)
        val_logits = self.val(h).view(-1, 1, self.n_atoms)
        q_logits = val_logits + adv_logits - adv_logits.mean(dim=1, keepdim=True)
        return q_logits

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Return categorical logits shape (B, A, n_atoms)."""
        return self._dist(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return expected Q-values (B, A)."""
        logits = self._dist(x)
        probs = F.softmax(logits, dim=-1)
        support = self.support.to(x.device)
        return torch.sum(probs * support, dim=-1)


@dataclass
class RainbowConfig:
    dataset_path: Path
    output_dir: Path = Path("results/cartpole/rainbow")
    seed: int = 0
    epochs: int = 600
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 2.5e-4
    tau: float = 0.005
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    eval_every: int = 50


def _prepare_loader(data: OfflineDataset, batch_size: int) -> DataLoader:
    ds = RainbowDataset(data)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def _projection(
    next_dist: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    support: torch.Tensor,
) -> torch.Tensor:
    """Project target distribution onto fixed support."""
    batch_size = rewards.shape[0]
    n_atoms = support.shape[0]
    delta_z = float(support[1] - support[0])

    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    tz = rewards + gamma * (1.0 - dones) * support.unsqueeze(0)
    tz = tz.clamp(support[0], support[-1])
    b = (tz - support[0]) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    l = l.clamp(min=0, max=n_atoms - 1)
    u = u.clamp(min=0, max=n_atoms - 1)

    # Flattened index_add avoids advanced indexing shape issues
    offset = torch.arange(batch_size, device=next_dist.device).unsqueeze(1) * n_atoms
    proj = torch.zeros(batch_size * n_atoms, device=next_dist.device)
    proj.index_add_(0, (l + offset).view(-1), (next_dist * (u - b)).view(-1))
    proj.index_add_(0, (u + offset).view(-1), (next_dist * (b - l)).view(-1))
    return proj.view(batch_size, n_atoms)


def evaluate_rainbow_policy(
    q_net: RainbowNet,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
    episodes: int = 20,
    seed: int = 0,
) -> List[float]:
    """Evaluate Rainbow Q-network on clean CartPole-v1."""
    import gymnasium as gym

    q_net.eval()
    device = next(q_net.parameters()).device
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
                q_vals = q_net(s.unsqueeze(0))
                a_idx = q_vals.argmax(dim=1).item()
            a_cont = a_idx / 10.0
            force = (2.0 * a_cont - 1.0) * 10.0
            a_bin = 1 if force > 0 else 0
            sp_raw, r, done, trunc, _ = env.step(a_bin)
            total_r += float(r)
            s = torch.tensor(sp_raw, dtype=torch.float32, device=device)
            s = (s - state_mean[0].to(device)) / state_std[0].to(device)
        returns.append(total_r)
        env.close()
    return returns


def train_rainbow_offline(cfg: RainbowConfig) -> Dict[str, Any]:
    """Train Rainbow DQN (C51) on the offline CTRL dataset."""
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_ctrl_dataset(cfg.dataset_path, device=device)
    loader = _prepare_loader(data, cfg.batch_size)

    q_net = RainbowNet(
        state_dim=4,
        n_actions=11,
        n_atoms=cfg.n_atoms,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
    ).to(device)
    target_net = RainbowNet(
        state_dim=4,
        n_actions=11,
        n_atoms=cfg.n_atoms,
        v_min=cfg.v_min,
        v_max=cfg.v_max,
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())

    opt = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)

    support = q_net.support.to(device)
    metrics: Dict[str, List[float]] = {
        "loss": [],
    }

    for ep in range(cfg.epochs):
        batch_losses: List[float] = []
        for batch in loader:
            s, a_idx, r, sp, d = [t.to(device) for t in batch]
            logits = q_net.dist(s)
            log_prob = F.log_softmax(logits, dim=-1)
            chosen_log_prob = log_prob[torch.arange(s.size(0)), a_idx]

            with torch.no_grad():
                next_logits = target_net.dist(sp)
                next_prob = F.softmax(next_logits, dim=-1)
                next_q = torch.sum(next_prob * support, dim=-1)
                next_a = next_q.argmax(dim=1)
                next_dist = next_prob[torch.arange(s.size(0)), next_a]
                target_dist = _projection(next_dist, r, d, cfg.gamma, support)

            loss = -(target_dist * chosen_log_prob).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                for p, tp in zip(q_net.parameters(), target_net.parameters()):
                    tp.data.mul_(1.0 - cfg.tau).add_(cfg.tau * p.data)

            batch_losses.append(loss.item())

        metrics["loss"].append(float(sum(batch_losses) / len(batch_losses)))

        if (ep + 1) % cfg.eval_every == 0 or ep == 0:
            returns = evaluate_rainbow_policy(
                q_net, data.state_mean, data.state_std, episodes=10, seed=cfg.seed
            )
            metrics.setdefault("eval_returns", []).append(
                {"epoch": ep + 1, "mean": float(sum(returns) / len(returns))}
            )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(q_net.state_dict(), cfg.output_dir / "q_net.pt")
    torch.save(target_net.state_dict(), cfg.output_dir / "target_q_net.pt")
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
