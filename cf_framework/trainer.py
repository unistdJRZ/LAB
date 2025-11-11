from __future__ import annotations

from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader

from .logging import LossLogger


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    g_optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: LossLogger,
    d_optimizer: Optional[torch.optim.Optimizer] = None,
    g_adv_weight: float = 0.1,
    history_len: int = 64,
    d_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    step = 0
    history_feats: List[torch.Tensor] = []
    history_labels: List[torch.Tensor] = []

    for batch in dataloader:
        step += 1
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            images, targets, d_bool = batch[0], batch[1], batch[2]
        else:
            images, targets = batch[0], batch[1]
            d_bool = torch.ones_like(targets, dtype=torch.bool)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        d_sign = torch.where(d_bool.to(device), torch.tensor(1.0, device=device), torch.tensor(-1.0, device=device))#辅助任务标签

        g_optimizer.zero_grad(set_to_none=True)
        out = model(images)

        # Parse outputs
        d_score_from_model = None
        if isinstance(out, tuple):#取中间特征和分类值
            if len(out) >= 2 and isinstance(out[1], dict):
                logits = out[0]
                if len(out) >= 3:
                    d_score_from_model = out[2]
                inter = out[1]
            else:
                logits = out[0]
                inter = {}
        else:
            logits = out  # type: ignore
            inter = {}

        cls_loss = criterion(logits, targets)#分类损失

        # Determine feature for D
        try:
            d_source = getattr(getattr(model, "model_cfg", None).intermediate, "d_source", "logits").lower()
        except Exception:
            d_source = "logits"

        feat_for_d: Optional[torch.Tensor] = None#获取判别器输入特征
        if isinstance(d_source, str) and d_source.startswith("target:") and isinstance(inter, dict):
            key = d_source.split(":", 1)[1].strip()
            feat_for_d = inter.get(key)
        else:
            feat_for_d = logits

        if isinstance(feat_for_d, torch.Tensor):#将非[B, C, H, W]的特征转换为4维
            if feat_for_d.dim() == 2:#[B, D]
                feat_for_d = feat_for_d.unsqueeze(-1).unsqueeze(-1)
            elif feat_for_d.dim() == 3:#[B, N, D] -> [B, N, 1, D]
                feat_for_d = feat_for_d.unsqueeze(-2)
            elif feat_for_d.dim() != 4:
                feat_for_d = feat_for_d.view(feat_for_d.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        else:
            feat_for_d = None

        g_adv_loss = torch.tensor(0.0, device=device)
        if feat_for_d is not None and hasattr(model, "_get_or_init_discriminator"):
            disc = model._get_or_init_discriminator(feat_for_d.shape[1])  # type: ignore[attr-defined]
            prev_reqs = [p.requires_grad for p in disc.parameters()]
            for p in disc.parameters():
                p.requires_grad = False
            score_map = disc(feat_for_d)#判别器输出
            score_vec = score_map.mean(dim=(1, 2, 3))# [B, C, H, W] -> [B]
            # Minimize y * D(x): drives D down for y=+1 and up for y=-1
            g_adv_loss = (d_sign * score_vec).mean()
            for p, old in zip(disc.parameters(), prev_reqs):
                p.requires_grad = old

        g_loss = cls_loss + float(g_adv_weight) * g_adv_loss
        g_loss.backward()
        g_optimizer.step()

        # Update history (by batch)
        if feat_for_d is not None:
            history_feats.append(feat_for_d.detach())
            history_labels.append(d_sign.detach())
            if len(history_feats) > int(history_len):
                history_feats.pop(0)
                history_labels.pop(0)

        # Train D using hinge loss on sampled history
        d_loss_val = None
        if d_optimizer is not None and len(history_feats) > 0 and hasattr(model, "_get_or_init_discriminator"):
            disc = model._get_or_init_discriminator(history_feats[-1].shape[1])  # type: ignore[attr-defined]
            for _ in range(int(d_steps)):
                d_optimizer.zero_grad(set_to_none=True)
                idx = random.randrange(len(history_feats))
                feat_sample = history_feats[idx].to(device)
                y_sample = history_labels[idx].to(device)
                for p in disc.parameters():
                    p.requires_grad = True
                d_out_map = disc(feat_sample)  # N x 1 x H' x W'

                # Split into positive (True, label +1) and negative (False, label -1)
                pos_mask = y_sample > 0
                neg_mask = ~pos_mask
                loss_real = torch.tensor(0.0, device=device)
                loss_fake = torch.tensor(0.0, device=device)
                if pos_mask.any():
                    loss_real = F.relu(1.0 - d_out_map[pos_mask]).mean()
                if neg_mask.any():
                    loss_fake = F.relu(1.0 + d_out_map[neg_mask]).mean()
                if pos_mask.any() and neg_mask.any():
                    d_loss = 0.5 * (loss_real + loss_fake)
                else:
                    d_loss = loss_real + loss_fake

                d_loss.backward()
                d_optimizer.step()
                d_loss_val = float(d_loss.detach().item())

        total_loss += float(g_loss.item())
        current_lr = 0.0
        for g in g_optimizer.param_groups:
            current_lr = g.get("lr", 0.0)
            break
        extra = {"cls": float(cls_loss.detach().item()), "g_adv": float(g_adv_loss.detach().item())}
        if d_loss_val is not None:
            extra["d_hinge"] = d_loss_val
        logger.log(step=step, epoch=epoch, loss=float(g_loss.item()), lr=float(current_lr), extra=extra or None)

    return total_loss / max(1, step)
