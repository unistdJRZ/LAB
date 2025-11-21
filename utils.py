from __future__ import annotations

import os
import random
from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from cf_framework import CFModel, LossLogger, AutoPivoitDataset
from cf_framework.config import Config
from cf_framework.data import collate_pivot


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


def build_loader(cfg: Config) -> DataLoader:
    from torchvision import datasets as tv_datasets, transforms  # lazy import

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
            transforms.ToTensor(),
        ]
    )

    if cfg.data.source == "imagefolder":
        base = tv_datasets.ImageFolder(cfg.data.root, transform=None)
        wrapped = AutoPivoitDataset(
            base,
            mu=cfg.data.mu,
            seed=cfg.training.epochs + 100 if hasattr(cfg, "training") else 42,
            pivot_file=cfg.data.pivot_file,
        )
        base.transform = tfm  # type: ignore[attr-defined]
    else:
        try:
            import datasets as hf  # type: ignore
        except Exception as e:
            raise RuntimeError("Please install 'datasets' to use Hugging Face datasets") from e
        hf_ds = hf.load_dataset(cfg.data.hf_name, split=cfg.data.hf_split)
        wrapped = AutoPivoitDataset(
            hf_ds,
            mu=cfg.data.mu,
            seed=42,
            pivot_file=cfg.data.pivot_file,
            image_key=cfg.data.image_key,
            label_key=cfg.data.label_key,
            transform=tfm,
        )

    loader = DataLoader(
        wrapped,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_pivot,
        persistent_workers=True
    )
    return loader


def build_val_loader(cfg: Config) -> DataLoader:
    from torchvision import datasets as tv_datasets, transforms  # lazy import

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
            transforms.ToTensor(),
        ]
    )

    if cfg.data.source == "imagefolder":
        val_root = os.path.join(cfg.data.root, "val")
        if os.path.isdir(val_root):
            base = tv_datasets.ImageFolder(val_root, transform=tfm)
            wrapped = base
        else:
            base = tv_datasets.ImageFolder(cfg.data.root, transform=tfm)
            wrapped = base
    else:
        try:
            import datasets as hf  # type: ignore
        except Exception as e:
            raise RuntimeError("Please install 'datasets' to use Hugging Face datasets") from e
        val_ds = None
        for split in ("validation", "val", "test", cfg.data.hf_split):
            try:
                val_ds = hf.load_dataset(cfg.data.hf_name, split=split)
                break
            except Exception:
                val_ds = None
                continue
        if val_ds is None:  # type: ignore
            raise RuntimeError("Could not load any validation split for the HF dataset.")
        wrapped = AutoPivoitDataset(
            val_ds,  # type: ignore
            mu=0.5,
            seed=123,
            pivot_file=None,
            image_key=cfg.data.image_key,
            label_key=cfg.data.label_key,
            transform=tfm,
            save_pivot=False,
        )

    loader = DataLoader(
        wrapped,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=collate_pivot,
        persistent_workers=True
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
        row = {"step": step, "epoch": epoch, "loss_G": loss, "lr": lr}
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
        track = [n for n in fieldnames if n in ("loss_G", "loss_D", "cls", "g_adv", "val_loss", "val_acc1")]
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

