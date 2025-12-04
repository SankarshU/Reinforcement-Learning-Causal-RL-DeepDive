#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def run(cmd: List[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end runner: dataset -> BiCoGAN -> D3QN real/CF -> SAC/Rainbow -> plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--cf-k", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/SD_dataset_clean.pt"),
        help="Where to save/load the dataset.",
    )
    parser.add_argument(
        "--bicogan-dir",
        type=Path,
        default=Path("results/cartpole/bicogan"),
        help="BiCoGAN checkpoint directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/plots"),
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    # 1) Dataset
    run(
        [
            "python",
            "scripts/run_ctrl.py",
            "dataset",
            "--episodes",
            str(args.episodes),
            "--horizon",
            str(args.horizon),
            "--output",
            str(args.dataset),
        ]
    )

    # 2) BiCoGAN
    run(
        [
            "python",
            "scripts/run_ctrl.py",
            "train-bicogan",
            "--dataset-path",
            str(args.dataset),
            "--output-dir",
            str(args.bicogan_dir),
        ]
    )

    # 3) D3QN real-only
    run(
        [
            "python",
            "scripts/run_ctrl.py",
            "train-d3qn",
            "--dataset-path",
            str(args.dataset),
            "--output-dir",
            "results/cartpole/d3qn_real",
        ]
    )

    # 4) D3QN CF
    run(
        [
            "python",
            "scripts/run_ctrl.py",
            "train-d3qn",
            "--dataset-path",
            str(args.dataset),
            "--use-cf",
            "--cf-k",
            str(args.cf_k),
            "--bicogan-dir",
            str(args.bicogan_dir),
            "--output-dir",
            "results/cartpole/d3qn_cf",
        ]
    )

    # 5) SAC
    run(
        [
            "python",
            "scripts/run_alt_algos.py",
            "sac",
            "--dataset-path",
            str(args.dataset),
            "--output-dir",
            "results/cartpole/sac",
        ]
    )

    # 6) Rainbow
    run(
        [
            "python",
            "scripts/run_alt_algos.py",
            "rainbow",
            "--dataset-path",
            str(args.dataset),
            "--output-dir",
            "results/cartpole/rainbow",
        ]
    )

    # 7) Plots
    run(
        [
            "python",
            "scripts/plot_results.py",
            "--sac-metrics",
            "results/cartpole/sac/metrics.json",
            "--rainbow-metrics",
            "results/cartpole/rainbow/metrics.json",
            "--d3qn-real",
            "results/cartpole/d3qn_real/metrics.json",
            "--d3qn-cf",
            "results/cartpole/d3qn_cf/metrics.json",
            "--output-dir",
            str(args.plots_dir),
        ]
    )


if __name__ == "__main__":
    main()
