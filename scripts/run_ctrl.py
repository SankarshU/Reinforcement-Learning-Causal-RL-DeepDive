#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "CTRL"))

from ctrl_data import CTRLTransitionDataset, build_training_buffer, make_SD_dataset
from ctrl_models import DiscriminatorCTRL, EncoderCTRL, GeneratorCartPoleMLP
from ctrl_trainer import BiCoGAN, D3QNHyperParams, train_offline_d3qn
from ctrl_utilities import evaluate_policy


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


@dataclass
class DatasetConfig:
    episodes: int = 250
    horizon: int = 200
    seed: int = 0
    output: Path = Path("data/SD_dataset_clean.pt")


@dataclass
class BiCoGANConfig:
    dataset_path: Path
    output_dir: Path = Path("results/cartpole/bicogan")
    batch_size: int = 54
    seed: int = 0
    udim: int = 4
    pre_train_lr: float = 1e-3
    pre_train_epochs: int = 15
    lr: float = 1e-4
    disc_lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.9
    num_epochs: int = 50
    alpha: float = 0.1
    rho: float = 5e-5
    phi: float = 10.0


@dataclass
class D3QNCliConfig:
    dataset_path: Path
    output_dir: Path = Path("results/cartpole/d3qn")
    seed: int = 0
    use_cf: bool = False
    cf_k: int = 1
    bicogan_dir: Optional[Path] = None
    udim: int = 4
    epochs: int = 1000
    batch_size: int = 512
    gamma: float = 0.99
    lr: float = 1.5e-4
    tau: float = 0.005
    alpha_cql: float = 0.02
    reward_clip: Optional[float] = None
    target_clip: Optional[float] = 20.0
    max_grad_norm: Optional[float] = 1.0


def generate_dataset(cfg: DatasetConfig) -> Path:
    set_seed(cfg.seed)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    dataset = make_SD_dataset(
        num_eps=cfg.episodes,
        horizon=cfg.horizon,
        seed=cfg.seed,
        save_path=str(cfg.output),
    )
    meta_path = cfg.output.with_suffix(cfg.output.suffix + ".meta.json")
    save_json(
        {
            "episodes": cfg.episodes,
            "horizon": cfg.horizon,
            "seed": cfg.seed,
            "output": str(cfg.output),
            "shapes": {k: list(v.shape) for k, v in dataset.items()},
        },
        meta_path,
    )
    return cfg.output


def load_bicogan(
    ckpt_dir: Path, udim: int, device: torch.device
) -> Tuple[GeneratorCartPoleMLP, EncoderCTRL]:
    obsdim = 4
    G = GeneratorCartPoleMLP(input_dim=obsdim + 1 + udim).to(device)
    E = EncoderCTRL(obsdim=obsdim, udim=udim).to(device)

    g_path = ckpt_dir / "G.pt"
    e_path = ckpt_dir / "E.pt"
    d_path = ckpt_dir / "D.pt"
    if not (g_path.exists() and e_path.exists()):
        raise FileNotFoundError(
            f"Missing BiCoGAN weights under {ckpt_dir}. "
            "Expect G.pt and E.pt (optional D.pt)."
        )

    G.load_state_dict(torch.load(g_path, map_location=device))
    E.load_state_dict(torch.load(e_path, map_location=device))
    if d_path.exists():
        torch.load(d_path, map_location="cpu")
    return G, E


def train_bicogan(cfg: BiCoGANConfig) -> Path:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(cfg.dataset_path)
    loader = DataLoader(
        CTRLTransitionDataset(data),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    obsdim = 4
    G = GeneratorCartPoleMLP(input_dim=obsdim + 1 + cfg.udim).to(device)
    E = EncoderCTRL(obsdim=obsdim, udim=cfg.udim).to(device)
    D = DiscriminatorCTRL(obsdim=obsdim).to(device)

    trainer = BiCoGAN(G, E, D, cfg)
    trainer.train(loader)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), cfg.output_dir / "G.pt")
    torch.save(E.state_dict(), cfg.output_dir / "E.pt")
    torch.save(D.state_dict(), cfg.output_dir / "D.pt")
    save_json(asdict(cfg), cfg.output_dir / "config.json")
    save_json(trainer.history, cfg.output_dir / "history.json")
    return cfg.output_dir


def train_d3qn(cfg: D3QNCliConfig) -> Path:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    real_data = torch.load(cfg.dataset_path)
    if cfg.use_cf:
        if cfg.bicogan_dir is None:
            raise ValueError("use_cf=True requires --bicogan-dir.")
        G, E = load_bicogan(cfg.bicogan_dir, cfg.udim, device)
        S_raw, A, R, SP_raw = build_training_buffer(
            real_data,
            use_cf=True,
            G=G,
            E=E,
            cf_k=cfg.cf_k,
        )
        label = f"Real+CF (k={cfg.cf_k}) D3QN+CQL"
    else:
        S_raw = real_data["s"].float().reshape(-1, 4)
        A = real_data["a"].long().reshape(-1)
        R = real_data["r"].float().reshape(-1)
        SP_raw = real_data["sp"].float().reshape(-1, 4)
        label = "Real-only D3QN+CQL"

    D_flags = (
        (SP_raw[:, 0] < -2.4)
        | (SP_raw[:, 0] > 2.4)
        | (SP_raw[:, 2] < -0.2095)
        | (SP_raw[:, 2] > 0.2095)
    ).float()

    S_mean = S_raw.mean(dim=0, keepdim=True)
    S_std = S_raw.std(dim=0, keepdim=True) + 1e-6
    S = (S_raw - S_mean) / S_std
    SP = (SP_raw - S_mean) / S_std

    hyper = D3QNHyperParams(
        epochs=cfg.epochs,
        gamma=cfg.gamma,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        tau=cfg.tau,
        alpha_cql=cfg.alpha_cql,
        reward_clip=cfg.reward_clip,
        target_clip=cfg.target_clip,
        max_grad_norm=cfg.max_grad_norm,
    )

    q_net, tot, tdl, cql, qmean, qstd, evals = train_offline_d3qn(
        S,
        A,
        R,
        SP,
        D_flags,
        hyper,
        S_mean,
        S_std,
        eval_every=20,
        label=label,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(q_net.state_dict(), cfg.output_dir / "q_net.pt")
    save_json(asdict(cfg), cfg.output_dir / "config.json")
    save_json(
        {
            "total_losses": tot,
            "td_losses": tdl,
            "cql_losses": cql,
            "q_means": qmean,
            "q_stds": qstd,
            "eval_returns": evals,
            "state_mean": S_mean.cpu().tolist(),
            "state_std": S_std.cpu().tolist(),
        },
        cfg.output_dir / "metrics.json",
    )

    final_returns = evaluate_policy(q_net, S_mean, S_std, episodes=20)
    save_json(
        {
            "mean_return": float(final_returns.mean()),
            "std_return": float(final_returns.std()),
        },
        cfg.output_dir / "final_eval.json",
    )
    return cfg.output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CTRL CartPole experiment orchestration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("dataset", help="Generate SD dataset.")
    ds.add_argument("--episodes", type=int, default=DatasetConfig.episodes)
    ds.add_argument("--horizon", type=int, default=DatasetConfig.horizon)
    ds.add_argument("--seed", type=int, default=DatasetConfig.seed)
    ds.add_argument("--output", type=Path, default=DatasetConfig.output)

    bicogan = sub.add_parser("train-bicogan", help="Train BiCoGAN.")
    bicogan.add_argument("--dataset-path", type=Path, required=True)
    bicogan.add_argument(
        "--output-dir", type=Path, default=BiCoGANConfig.output_dir
    )
    bicogan.add_argument(
        "--batch-size", type=int, default=BiCoGANConfig.batch_size
    )
    bicogan.add_argument("--seed", type=int, default=BiCoGANConfig.seed)
    bicogan.add_argument("--udim", type=int, default=BiCoGANConfig.udim)
    bicogan.add_argument(
        "--pre-train-lr", type=float, default=BiCoGANConfig.pre_train_lr
    )
    bicogan.add_argument(
        "--pre-train-epochs", type=int, default=BiCoGANConfig.pre_train_epochs
    )
    bicogan.add_argument("--lr", type=float, default=BiCoGANConfig.lr)
    bicogan.add_argument("--disc-lr", type=float, default=BiCoGANConfig.disc_lr)
    bicogan.add_argument("--beta1", type=float, default=BiCoGANConfig.beta1)
    bicogan.add_argument("--beta2", type=float, default=BiCoGANConfig.beta2)
    bicogan.add_argument(
        "--num-epochs", type=int, default=BiCoGANConfig.num_epochs
    )
    bicogan.add_argument("--alpha", type=float, default=BiCoGANConfig.alpha)
    bicogan.add_argument("--rho", type=float, default=BiCoGANConfig.rho)
    bicogan.add_argument("--phi", type=float, default=BiCoGANConfig.phi)

    d3qn = sub.add_parser("train-d3qn", help="Train offline D3QN+CQL.")
    d3qn.add_argument("--dataset-path", type=Path, required=True)
    d3qn.add_argument(
        "--output-dir", type=Path, default=D3QNCliConfig.output_dir
    )
    d3qn.add_argument("--seed", type=int, default=D3QNCliConfig.seed)
    d3qn.add_argument("--use-cf", action="store_true")
    d3qn.add_argument("--cf-k", type=int, default=D3QNCliConfig.cf_k)
    d3qn.add_argument(
        "--bicogan-dir", type=Path, help="Directory with G.pt/E.pt."
    )
    d3qn.add_argument("--udim", type=int, default=D3QNCliConfig.udim)
    d3qn.add_argument("--epochs", type=int, default=D3QNCliConfig.epochs)
    d3qn.add_argument(
        "--batch-size", type=int, default=D3QNCliConfig.batch_size
    )
    d3qn.add_argument("--gamma", type=float, default=D3QNCliConfig.gamma)
    d3qn.add_argument("--lr", type=float, default=D3QNCliConfig.lr)
    d3qn.add_argument("--tau", type=float, default=D3QNCliConfig.tau)
    d3qn.add_argument(
        "--alpha-cql", type=float, default=D3QNCliConfig.alpha_cql
    )
    d3qn.add_argument(
        "--reward-clip", type=float, default=D3QNCliConfig.reward_clip
    )
    d3qn.add_argument(
        "--target-clip", type=float, default=D3QNCliConfig.target_clip
    )
    d3qn.add_argument(
        "--max-grad-norm", type=float, default=D3QNCliConfig.max_grad_norm
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "dataset":
        cfg = DatasetConfig(
            episodes=args.episodes,
            horizon=args.horizon,
            seed=args.seed,
            output=args.output,
        )
        output = generate_dataset(cfg)
        print(f"Saved dataset to {output}")
        return

    if args.command == "train-bicogan":
        cfg = BiCoGANConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            udim=args.udim,
            pre_train_lr=args.pre_train_lr,
            pre_train_epochs=args.pre_train_epochs,
            lr=args.lr,
            disc_lr=args.disc_lr,
            beta1=args.beta1,
            beta2=args.beta2,
            num_epochs=args.num_epochs,
            alpha=args.alpha,
            rho=args.rho,
            phi=args.phi,
        )
        output = train_bicogan(cfg)
        print(f"Saved BiCoGAN artifacts to {output}")
        return

    if args.command == "train-d3qn":
        cfg = D3QNCliConfig(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            seed=args.seed,
            use_cf=args.use_cf,
            cf_k=args.cf_k,
            bicogan_dir=args.bicogan_dir,
            udim=args.udim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            tau=args.tau,
            alpha_cql=args.alpha_cql,
            reward_clip=args.reward_clip,
            target_clip=args.target_clip,
            max_grad_norm=args.max_grad_norm,
        )
        output = train_d3qn(cfg)
        print(f"Saved D3QN artifacts to {output}")
        return


if __name__ == "__main__":
    main()
