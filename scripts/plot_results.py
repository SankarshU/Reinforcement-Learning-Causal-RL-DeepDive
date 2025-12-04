#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"[skip] missing {path}")
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[skip] failed to parse {path}: {exc}")
        return None


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_sac(path: Path, out_dir: Path) -> None:
    data = load_json(path)
    if not data:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(data.get("critic_loss", []), label="critic")
    ax[0].plot(data.get("actor_loss", []), label="actor")
    ax[0].set_title("SAC losses")
    ax[0].set_xlabel("epoch")
    ax[0].legend()

    evals = data.get("eval_returns") or []
    if evals:
        ax[1].plot(
            [e["epoch"] for e in evals],
            [e["mean"] for e in evals],
            marker="o",
        )
        ax[1].set_title("SAC eval mean return")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("mean return")
    save_fig(fig, out_dir / "sac_overview.png")


def plot_rainbow(path: Path, out_dir: Path) -> None:
    data = load_json(path)
    if not data:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(data.get("loss", []))
    ax[0].set_title("Rainbow loss")
    ax[0].set_xlabel("epoch")

    evals = data.get("eval_returns") or []
    if evals:
        ax[1].plot(
            [e["epoch"] for e in evals],
            [e["mean"] for e in evals],
            marker="o",
        )
        ax[1].set_title("Rainbow eval mean return")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("mean return")
    save_fig(fig, out_dir / "rainbow_overview.png")


def plot_d3qn(real_path: Path, cf_path: Path, out_dir: Path) -> None:
    real = load_json(real_path) or {}
    cf = load_json(cf_path) or {}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, data, color in [
        ("real", real, "tab:blue"),
        ("cf", cf, "tab:orange"),
    ]:
        if not data:
            continue
        ax[0].plot(data.get("total_losses", []), label=f"{name} total", color=color)
        ax[0].plot(
            data.get("td_losses", []),
            label=f"{name} td",
            color=color,
            alpha=0.5,
            linestyle="--",
        )
        evals = data.get("eval_returns") or []
        if evals:
            ax[1].plot(
                [i + 1 for i, _ in enumerate(evals)],
                [v[0] if isinstance(v, (list, tuple)) else v["mean"] for v in evals],
                label=name,
                marker="o",
                color=color,
            )
    ax[0].set_title("D3QN losses")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].set_title("D3QN eval mean return")
    ax[1].set_xlabel("eval idx")
    ax[1].legend()
    save_fig(fig, out_dir / "d3qn_overview.png")


def plot_bar_comparison(results: Dict[str, Path], out_dir: Path) -> None:
    labels: List[str] = []
    vals: List[float] = []
    for name, path in results.items():
        data = load_json(path)
        if not data:
            continue
        evals = data.get("eval_returns") or []
        if not evals:
            continue
        last = evals[-1]
        if isinstance(last, dict):
            vals.append(float(last.get("mean", 0.0)))
        else:
            try:
                vals.append(float(last[0]))
            except (TypeError, ValueError, IndexError):
                continue
        labels.append(name)
    if not labels:
        print("[skip] no eval data for bar chart")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color="tab:green")
    ax.set_ylabel("Mean return")
    ax.set_title("Final eval comparison")
    save_fig(fig, out_dir / "comparison_bar.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate plots for CTRL CartPole experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sac-metrics",
        type=Path,
        default=Path("results/cartpole/sac/metrics.json"),
        help="Path to SAC metrics.json.",
    )
    parser.add_argument(
        "--rainbow-metrics",
        type=Path,
        default=Path("results/cartpole/rainbow/metrics.json"),
        help="Path to Rainbow metrics.json.",
    )
    parser.add_argument(
        "--d3qn-real",
        type=Path,
        default=Path("results/cartpole/d3qn_real/metrics.json"),
        help="Path to real-only D3QN metrics.json.",
    )
    parser.add_argument(
        "--d3qn-cf",
        type=Path,
        default=Path("results/cartpole/d3qn_cf/metrics.json"),
        help="Path to counterfactual D3QN metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="Directory to write png plots.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir

    plot_sac(args.sac_metrics, out_dir)
    plot_rainbow(args.rainbow_metrics, out_dir)
    plot_d3qn(args.d3qn_real, args.d3qn_cf, out_dir)
    plot_bar_comparison(
        {
            "SAC": args.sac_metrics,
            "Rainbow": args.rainbow_metrics,
            "D3QN real": args.d3qn_real,
            "D3QN CF": args.d3qn_cf,
        },
        out_dir,
    )


if __name__ == "__main__":
    main()
