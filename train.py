from __future__ import annotations

import argparse
import os
import torch

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# No custom transform functions here to keep pickling simple on Windows

from cf_framework import CFModel
from cf_framework.config import Config

from utils import set_seed, build_loader, build_val_loader

try:
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger, CSVLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    _LIGHTNING_OK = True
except Exception:
    _LIGHTNING_OK = False


def main():
    ap = argparse.ArgumentParser(description="Train CFModel with optional adversarial D and pivoted dataset")
    ap.add_argument("--config", required=False, default=None, help="Path to YAML config (defaults to configs/default.yaml)")
    ap.add_argument("--ckpt-dir", required=False, default=None, help="Directory to save checkpoints (defaults to training.log_dir/checkpoints)")
    ap.add_argument("--eval-every", type=int, default=1, help="Run validation every K epochs (default: 1)")
    args = ap.parse_args()

    if args.config:
        cfg = Config.from_file(args.config)
    else:
        cfg_path = os.path.join("configs", "default.yaml")
        cfg = Config.from_file(cfg_path)
    # No extra CLI overrides; use config values

    set_seed(42)

    # Resolve run directory (optionally using task name)
    run_root = cfg.training.log_dir
    task_name = getattr(cfg.training, "task_name", None)
    if task_name:
        run_dir = os.path.join(run_root, task_name)
    else:
        run_dir = run_root
    os.makedirs(run_dir, exist_ok=True)

    # Data
    loader = build_loader(cfg)
    val_loader = build_val_loader(cfg)

    # Force Lightning path only
    if not _LIGHTNING_OK:
        raise RuntimeError("Please install 'lightning' to run training (pip install lightning)")
    logger = None
    if getattr(cfg.training, "wandb", False):
        try:
            logger = WandbLogger(
                project=(cfg.training.wandb_project or os.getenv("WANDB_PROJECT") or "cf-framework"),
                name=(task_name or None),
            )
        except Exception:
            logger = CSVLogger(save_dir=run_root, name=(task_name or "lightning"))
    else:
        logger = CSVLogger(save_dir=run_root, name=(task_name or "lightning"))

    ckpt_dir = args.ckpt_dir or os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
    )
    module = CFModel(
        cfg,
        g_adv_weight=float(getattr(cfg.training, "g_adv_weight", 0.1)),
        d_steps=int(getattr(cfg.training, "d_steps", 1)),
    )
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        callbacks=[ckpt],
        default_root_dir=run_dir,
        check_val_every_n_epoch=int(getattr(args, "eval_every", 1)),
    )
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=val_loader)
    return


if __name__ == "__main__":
    main()

