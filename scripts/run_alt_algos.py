#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ctrl_algorithms.rainbow import RainbowConfig, train_rainbow_offline
from ctrl_algorithms.sac import SACConfig, train_sac_offline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Alternative CTRL algorithms (SAC, Rainbow DQN) on offline CartPole.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sac = sub.add_parser("sac", help="Train SAC offline.")
    sac.add_argument("--dataset-path", type=Path, required=True)
    sac.add_argument("--output-dir", type=Path, default=SACConfig.output_dir)
    sac.add_argument("--seed", type=int, default=SACConfig.seed)
    sac.add_argument("--epochs", type=int, default=SACConfig.epochs)
    sac.add_argument("--batch-size", type=int, default=SACConfig.batch_size)
    sac.add_argument("--gamma", type=float, default=SACConfig.gamma)
    sac.add_argument("--tau", type=float, default=SACConfig.tau)
    sac.add_argument("--lr", type=float, default=SACConfig.lr)
    sac.add_argument("--alpha", type=float, default=SACConfig.alpha)
    sac.add_argument("--auto-alpha", action="store_true", default=True)
    sac.add_argument(
        "--no-auto-alpha",
        dest="auto_alpha",
        action="store_false",
        help="Disable entropy auto-tuning.",
    )
    sac.add_argument(
        "--target-entropy", type=float, default=SACConfig.target_entropy
    )
    sac.add_argument("--eval-every", type=int, default=SACConfig.eval_every)

    rb = sub.add_parser("rainbow", help="Train Rainbow DQN offline.")
    rb.add_argument("--dataset-path", type=Path, required=True)
    rb.add_argument("--output-dir", type=Path, default=RainbowConfig.output_dir)
    rb.add_argument("--seed", type=int, default=RainbowConfig.seed)
    rb.add_argument("--epochs", type=int, default=RainbowConfig.epochs)
    rb.add_argument("--batch-size", type=int, default=RainbowConfig.batch_size)
    rb.add_argument("--gamma", type=float, default=RainbowConfig.gamma)
    rb.add_argument("--lr", type=float, default=RainbowConfig.lr)
    rb.add_argument("--tau", type=float, default=RainbowConfig.tau)
    rb.add_argument("--n-atoms", type=int, default=RainbowConfig.n_atoms)
    rb.add_argument("--v-min", type=float, default=RainbowConfig.v_min)
    rb.add_argument("--v-max", type=float, default=RainbowConfig.v_max)
    rb.add_argument("--eval-every", type=int, default=RainbowConfig.eval_every)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sac":
        cfg = SACConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            lr=args.lr,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha,
            target_entropy=args.target_entropy,
            eval_every=args.eval_every,
        )
        train_sac_offline(cfg)
        print(f"SAC artifacts saved to {cfg.output_dir}")
        return

    if args.command == "rainbow":
        cfg = RainbowConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            tau=args.tau,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
            eval_every=args.eval_every,
        )
        train_rainbow_offline(cfg)
        print(f"Rainbow artifacts saved to {cfg.output_dir}")
        return


if __name__ == "__main__":
    main()
