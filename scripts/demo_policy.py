#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(REPO_ROOT / "CTRL") not in sys.path:
    sys.path.append(str(REPO_ROOT / "CTRL"))

from ctrl_algorithms.data import load_ctrl_dataset
from ctrl_algorithms.rainbow import RainbowNet
from ctrl_algorithms.sac import Actor
from ctrl_trainer import QNetCTRL


def choose_env(record_dir: Optional[Path]) -> tuple:
    import gymnasium as gym

    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(record_dir),
            name_prefix="demo",
            episode_trigger=lambda ep: True,
        )
    else:
        env = gym.make("CartPole-v1", render_mode="human")
    return env


def map_action_from_idx(idx: int) -> int:
    a_cont = idx / 10.0
    force = (2.0 * a_cont - 1.0) * 10.0
    return 1 if force > 0 else 0


def run_demo(
    algo: str,
    model_path: Path,
    dataset_path: Path,
    episodes: int,
    seed: int,
    record_dir: Optional[Path],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_ctrl_dataset(dataset_path, device=device)
    state_mean = data.state_mean.to(device)
    state_std = data.state_std.to(device)

    if algo == "sac":
        policy = Actor().to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()
    elif algo == "rainbow":
        policy = RainbowNet().to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()
    elif algo == "d3qn":
        policy = QNetCTRL().to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval()
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    for ep in range(episodes):
        env = choose_env(record_dir)
        s_raw, _ = env.reset(seed=seed + ep)
        s = torch.tensor(s_raw, dtype=torch.float32, device=device)
        s = (s - state_mean[0]) / state_std[0]

        done = False
        trunc = False
        total_r = 0.0
        while not (done or trunc):
            with torch.no_grad():
                if algo == "sac":
                    a_cont = policy.deterministic(s.unsqueeze(0)).squeeze(0).item()
                    force = (2.0 * a_cont - 1.0) * 10.0
                    a_bin = 1 if force > 0 else 0
                else:
                    q_vals = policy(s.unsqueeze(0))
                    a_idx = q_vals.argmax(dim=1).item()
                    a_bin = map_action_from_idx(a_idx)

            s_raw, r, done, trunc, _ = env.step(a_bin)
            total_r += float(r)
            s = torch.tensor(s_raw, dtype=torch.float32, device=device)
            s = (s - state_mean[0]) / state_std[0]

        env.close()
        print(f"[{algo}] Episode {ep+1}/{episodes} return: {total_r:.1f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render or record a policy demo for CTRL CartPole.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo", choices=["sac", "rainbow", "d3qn"], required=True, help="Policy type."
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Checkpoint file.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Dataset to compute normalization stats.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--record-dir",
        type=Path,
        help="If set, record videos to this directory; otherwise uses on-screen render.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_demo(
        algo=args.algo,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        episodes=args.episodes,
        seed=args.seed,
        record_dir=args.record_dir,
    )


if __name__ == "__main__":
    main()
