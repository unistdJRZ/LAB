from __future__ import annotations

import argparse
import os
import random
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from cf_framework import CFModel, LossLogger, train_one_epoch, AutoPivoitDataset
from cf_framework.config import Config


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loader(cfg: Config, mu: float, pivot_file: Optional[str], seed: int) -> DataLoader:
    from torchvision import datasets, transforms  # lazy import

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(cfg.data.root, transform=tfm)
    ds = AutoPivoitDataset(ds, mu=mu, seed=seed, pivot_file=pivot_file)
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return loader


def infer_d_in_channels(model: CFModel, batch: torch.Tensor) -> int:
    model.train()
    with torch.no_grad():
        out = model(batch)
        inter = out[1] if (isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], dict)) else {}
        d_source = getattr(getattr(model, "model_cfg", None).intermediate, "d_source", "logits").lower()
        if d_source.startswith("target:") and isinstance(inter, dict):
            key = d_source.split(":", 1)[1].strip()
            feat = inter.get(key)
        else:
            feat = out[0] if isinstance(out, tuple) else out
        if not isinstance(feat, torch.Tensor):
            raise RuntimeError("Could not infer discriminator input feature from model output.")
        if feat.dim() == 2:
            feat = feat.unsqueeze(-1).unsqueeze(-1)
        elif feat.dim() == 3:
            feat = feat.unsqueeze(-2)
        elif feat.dim() != 4:
            feat = feat.view(feat.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        return int(feat.shape[1])


class WandbLossLogger(LossLogger):
    def __init__(self, log_dir: str, wandb_run, filename: str = "loss.csv") -> None:
        super().__init__(log_dir=log_dir, filename=filename)
        self._wandb = wandb_run

    def log(self, step: int, epoch: int, loss: float, lr: float, extra: Optional[Dict] = None):
        super().log(step, epoch, loss, lr, extra)
        row = {"step": step, "epoch": epoch, "loss": loss, "lr": lr}
        if extra:
            row.update(extra)
        try:
            self._wandb.log(row)
        except Exception:
            pass


def plot_curves(csv_path: str, out_path: str, wandb_run=None) -> None:
    try:
        import csv
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting skipped (matplotlib not available): {e}")
        return

    steps = []
    metrics: Dict[str, list] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # track common metrics if present
        track = [n for n in fieldnames if n in ("loss", "cls", "g_adv", "d_hinge")]
        for row in reader:
            steps.append(int(row.get("step", len(steps) + 1)))
            for k in track:
                try:
                    v = float(row.get(k, "nan"))
                except Exception:
                    continue
                metrics.setdefault(k, []).append(v)

    if not steps or not metrics:
        print("No data to plot from CSV.")
        return

    plt.figure(figsize=(8, 5))
    for k, v in metrics.items():
        plt.plot(steps[: len(v)], v, label=k)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    try:
        if wandb_run is not None:
            import wandb  # type: ignore
            wandb_run.log({"training_curves": wandb.Image(out_path)})
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Train CFModel with optional adversarial D and pivoted dataset")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--mu", type=float, default=0.5, help="Pivot ratio mu per class")
    ap.add_argument("--pivot-file", type=str, default=None, help="Path to pivot YAML (load/save)")
    ap.add_argument("--device", type=str, default=None, help="Override device, e.g., cuda or cpu")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap.add_argument("--log-dir", type=str, default=None, help="Override log dir")
    ap.add_argument("--ckpt-dir", type=str, default=None, help="Directory to save checkpoints (defaults to log dir)")
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    ap.add_argument("--g-adv-weight", type=float, default=0.1, help="Weight for G adversarial term")
    ap.add_argument("--history-len", type=int, default=64, help="History queue length for D training")
    ap.add_argument("--d-steps", type=int, default=1, help="Number of D steps per G step")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    cfg = Config.from_file(args.config)
    if args.device:
        cfg.training.device = args.device
    if args.epochs is not None:
        cfg.training.epochs = int(args.epochs)
    if args.log_dir:
        cfg.training.log_dir = args.log_dir

    set_seed(args.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device.startswith("cuda") else "cpu")

    # Build model and data
    model = CFModel(cfg).to(device)
    loader = build_loader(cfg, mu=args.mu, pivot_file=args.pivot_file, seed=args.seed)

    # Optimizers
    g_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    # Infer D channels and create D optimizer
    first_images, _, _ = next(iter(loader))
    first_images = first_images.to(device)
    in_ch = infer_d_in_channels(model, first_images)
    disc = model._get_or_init_discriminator(in_ch)
    d_optimizer = torch.optim.Adam(disc.parameters(), lr=cfg.training.lr)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Logging (CSV + optional W&B)
    wandb_run = None
    logger: LossLogger
    if args.wandb:
        try:
            import wandb  # type: ignore

            wandb_run = wandb.init(project=(args.wandb_project or os.getenv("WANDB_PROJECT") or "cf-framework"), config={
                "config": args.config,
                "mu": args.mu,
                "seed": args.seed,
                "epochs": cfg.training.epochs,
                "lr": cfg.training.lr,
            })
            logger = WandbLossLogger(cfg.training.log_dir, wandb_run)
        except Exception as e:
            print(f"W&B disabled (failed to init): {e}")
            logger = LossLogger(cfg.training.log_dir)
    else:
        logger = LossLogger(cfg.training.log_dir)

    os.makedirs(cfg.training.log_dir, exist_ok=True)
    ckpt_dir = args.ckpt_dir or cfg.training.log_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    best = float("inf")
    for epoch in range(1, cfg.training.epochs + 1):
        avg = train_one_epoch(
            model,
            loader,
            g_optimizer,
            criterion,
            device,
            epoch,
            logger,
            d_optimizer=d_optimizer,
            g_adv_weight=float(args.g_adv_weight),
            history_len=int(args.history_len),
            d_steps=int(args.d_steps),
        )
        print(f"Epoch {epoch} avg loss: {avg:.4f}")

        # Save checkpoints
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "avg_loss": avg,
            "config": cfg,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))
        if avg < best:
            best = avg
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))

    # Plot curves locally and to W&B if enabled
    csv_path = os.path.join(cfg.training.log_dir, "loss.csv")
    png_path = os.path.join(cfg.training.log_dir, "training_curve.png")
    plot_curves(csv_path, png_path, wandb_run=wandb_run)

    logger.close()
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()

