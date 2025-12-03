from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class OfflineDataset:
    """Container for normalized offline CTRL transitions."""

    states: torch.Tensor
    actions: torch.Tensor
    actions_cont: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor


class SACDataset(Dataset):
    """Offline dataset exposing continuous actions for SAC."""

    def __init__(self, data: OfflineDataset):
        self.data = data

    def __len__(self) -> int:
        return self.data.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.data.states[idx],
            self.data.actions_cont[idx],
            self.data.rewards[idx],
            self.data.next_states[idx],
            self.data.dones[idx],
        )


class RainbowDataset(Dataset):
    """Offline dataset exposing discrete actions for Rainbow DQN."""

    def __init__(self, data: OfflineDataset):
        self.data = data

    def __len__(self) -> int:
        return self.data.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.data.states[idx],
            self.data.actions[idx],
            self.data.rewards[idx],
            self.data.next_states[idx],
            self.data.dones[idx],
        )


def _compute_dones(next_states: torch.Tensor) -> torch.Tensor:
    x_pos = next_states[:, 0]
    th = next_states[:, 2]
    done = (
        (x_pos < -2.4)
        | (x_pos > 2.4)
        | (th < -0.2095)
        | (th > 0.2095)
    ).float()
    return done


def load_ctrl_dataset(path: Path, device: Optional[torch.device] = None) -> OfflineDataset:
    """Load and normalize the CTRL CartPole dataset.

    Args:
        path: Path to a torch-saved dataset with keys s, a, acont, r, sp.
        device: Optional device to move tensors onto.

    Returns:
        OfflineDataset with normalized states and next states plus statistics.
    """
    raw = torch.load(path, map_location=device)

    states_raw = raw["s"].float().reshape(-1, 4)
    actions = raw["a"].long().reshape(-1)
    actions_cont = raw.get("acont")
    if actions_cont is None:
        raise ValueError("Dataset missing 'acont' needed for SAC.")
    actions_cont = actions_cont.float().reshape(-1)
    rewards = raw["r"].float().reshape(-1)
    next_states_raw = raw["sp"].float().reshape(-1, 4)

    state_mean = states_raw.mean(dim=0, keepdim=True)
    state_std = states_raw.std(dim=0, keepdim=True) + 1e-6

    states = (states_raw - state_mean) / state_std
    next_states = (next_states_raw - state_mean) / state_std
    dones = _compute_dones(next_states_raw)

    return OfflineDataset(
        states=states,
        actions=actions,
        actions_cont=actions_cont,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        state_mean=state_mean,
        state_std=state_std,
    )
