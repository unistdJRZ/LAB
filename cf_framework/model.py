from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import lightning as L

from .config import Config, ModelConfig
from .norms import ActNorm
from .hooks import capture_intermediates
from .utils import find_and_replace_classifier

def _lazy_import_transformers():
    try:
        import transformers  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Transformers not installed. Please install 'transformers' to use provider=transformers"
        ) from e
    return transformers


def _lazy_import_torchvision():
    try:
        import torchvision  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "torchvision not installed. Please install 'torchvision' to use provider=torchvision"
        ) from e
    return torchvision


class CFModel(L.LightningModule):
    """Single LightningModule for CF training with backbone + discriminator.

    - Builds `torchvision` or `transformers` backbone from YAML
    - Replaces/attaches a classification head to match `num_classes`
    - Captures differentiable intermediate outputs via decorator
    - Implements specialized training with optional PatchGAN discriminator (manual optimization)
    """

    def __init__(
        self,
        config: Config | ModelConfig,
        g_adv_weight: float = 0.1,
        d_steps: int = 1,
    ) -> None:
        super().__init__()
        # Accept either full Config or just ModelConfig for flexibility
        if isinstance(config, Config):
            self.cfg = config
            self.model_cfg = config.model
        else:
            self.cfg = Config(model=config)  # type: ignore[arg-type]
            self.model_cfg = config

        provider = self.model_cfg.provider.lower()
        if provider not in {"torchvision", "transformers"}:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider
        self.num_classes = int(self.model_cfg.num_classes)
        self.backbone: nn.Module
        self.classifier: nn.Module | None = None
        self._disc: Optional[nn.Module] = None  # lazy PatchGAN discriminator
        self.criterion = nn.CrossEntropyLoss()
        self.g_adv_weight = float(g_adv_weight)
        self.d_steps = int(d_steps)
        # D-queue hyperparameters
        self.d_queue_k = int(getattr(self.cfg.training, "d_queue_k", 1))
        raw_bd = int(getattr(self.cfg.training, "d_batch_size", 0))
        self.d_batch_size = int(raw_bd if raw_bd and raw_bd > 0 else self.cfg.training.batch_size)

        # Queues for discriminator features and pivot sign
        self._d_feat_queue: list[torch.Tensor] = []
        self._d_sign_queue: list[torch.Tensor] = []

        # Manual optimization to control G then D sequence
        self.automatic_optimization = False

        if provider == "torchvision":
            self._build_torchvision()
        else:
            self._build_transformers()

    def _build_torchvision(self) -> None:
        tv = _lazy_import_torchvision()
        name = self.model_cfg.name
        pretrained = self.model_cfg.pretrained

        # Newer torchvision versions prefer weights=...; fall back to pretrained=True for older.
        factory = getattr(tv.models, name)
        model = None
        try:
            # Try Weights API
            weights = None
            if pretrained:
                try:
                    default_weights = getattr(tv.models, f"{name}_Weights").DEFAULT
                    weights = default_weights
                except Exception:
                    pass
            model = factory(weights=weights)
        except TypeError:
            model = factory(pretrained=pretrained)

        # Replace classifier to match num_classes
        replaced = find_and_replace_classifier(model, self.num_classes)
        if not replaced:
            # As a fallback, append a fresh head
            in_features = 512
            try:
                # heuristic: get feature size from last linear if exists
                last_linear = None
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        last_linear = m
                if last_linear is not None:
                    in_features = last_linear.in_features
            except Exception:
                pass
            self.classifier = nn.Linear(in_features, self.num_classes)
            model = nn.Sequential(model, self.classifier)

        self.backbone = model

    def _build_transformers(self) -> None:
        tr = _lazy_import_transformers()
        name = self.model_cfg.name
        pretrained = self.model_cfg.pretrained

        # Use AutoModel (feature extractor) and add our classifier head
        if pretrained:
            base = tr.AutoModel.from_pretrained(name)
        else:
            cfg = tr.AutoConfig.from_pretrained(name)
            base = tr.AutoModel.from_config(cfg)

        hidden_size = getattr(base.config, "hidden_size", None)
        if hidden_size is None:
            # Some vision models use 'hidden_sizes' or other fields; attempt fallback
            hidden_size = (
                getattr(base.config, "hidden_sizes", [None])[-1]
                or getattr(base.config, "embed_dim", None)
                or getattr(base.config, "vision_config", None).hidden_size  # type: ignore
            )
        if hidden_size is None:
            raise RuntimeError("Unable to infer hidden size for classifier head")

        self.backbone = base
        self.classifier = nn.Linear(int(hidden_size), self.num_classes)

    def _targets(self):  # used by decorator
        return list(self.model_cfg.intermediate.targets or [])

    @capture_intermediates(lambda self: self._targets())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing logits.

        Returns either logits (for torchvision) or logits built from transformer outputs.
        The decorator wraps this and returns (logits, intermediates).
        """
        if self.provider == "torchvision":
            logits = self.backbone(x)
            return logits
        else:
            # transformers vision models expect pixel_values input
            outputs = self.backbone(pixel_values=x)
            key = (self.model_cfg.intermediate.feature_key or "pooler").lower()
            if key in {"pooler", "pooler_output"} and hasattr(outputs, "pooler_output"):
                feats = outputs.pooler_output
            elif key in {"cls", "cls_token"} and hasattr(outputs, "last_hidden_state"):
                feats = outputs.last_hidden_state[:, 0]
            elif key in {"mean", "avg", "mean_pool"} and hasattr(outputs, "last_hidden_state"):
                feats = outputs.last_hidden_state.mean(dim=1)
            elif key in {"last_hidden", "last_hidden_state"} and hasattr(outputs, "last_hidden_state"):
                # Flatten CLS-like use; here we mean-pool for a fixed size
                feats = outputs.last_hidden_state.mean(dim=1)
            else:
                # best-effort fallback
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feats = outputs.pooler_output
                else:
                    feats = outputs.last_hidden_state[:, 0]

            logits = self.classifier(feats) if self.classifier is not None else feats
            return logits

    # --- Lightning API ---
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        # Initialize D using logits channels (num_classes) by default
        in_ch = int(self.model_cfg.num_classes)
        disc = self._get_or_init_discriminator(in_ch)
        d_opt = torch.optim.Adam(disc.parameters(), lr=self.cfg.training.lr)
        return [g_opt, d_opt]

    def training_step(self, batch, batch_idx: int):
        opt_g, opt_d = self.optimizers()  # type: ignore[misc]

        images, targets, d_bool = self._unpack_batch(batch)
        d_sign = torch.where(d_bool, torch.tensor(1.0, device=images.device), torch.tensor(-1.0, device=images.device))

        # Forward G
        out = self.forward(images)
        logits, inter = self._parse_out(out)

        cls_loss = self.criterion(logits, targets)

        # Adversarial term for G (freeze D)
        feat_for_d = self._feature_for_d(logits, inter)
        g_adv_loss = torch.tensor(0.0, device=images.device)
        if feat_for_d is not None:
            # Enqueue current batch for D
            self._enqueue_d_batch(feat_for_d.detach(), d_sign.detach())

            disc = self._get_or_init_discriminator(feat_for_d.shape[1])
            reqs = [p.requires_grad for p in disc.parameters()]
            for p in disc.parameters():
                p.requires_grad = False
            score_map = disc(feat_for_d)
            score_vec = score_map.mean(dim=(1, 2, 3))
            g_adv_loss = (d_sign * score_vec).mean()#正样本*+1 - 负样本*-1，保证损失为正
            for p, old in zip(disc.parameters(), reqs):
                p.requires_grad = old

        # Optionally delay adversarial term until a configured starting step
        try:
            start_step = int(getattr(self.cfg.training, "starting_step", 0))
        except Exception:
            start_step = 0
        eff_g_adv_weight = float(self.g_adv_weight) if int(self.global_step) >= start_step else 0.0
        g_loss = cls_loss + eff_g_adv_weight * g_adv_loss

        # Step G
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        # Train D using hinge loss
        d_loss_val = None
        # Sample a D batch from the queue (K * B window)
        for _ in range(int(self.d_steps)):
            sample = self._sample_d_batch()
            if sample is not None:
                feat_d, sign_d = sample
                disc = self._get_or_init_discriminator(feat_d.shape[1])
                opt_d.zero_grad(set_to_none=True)
                for p in disc.parameters():
                    p.requires_grad = True
                d_out_map = disc(feat_d.detach())
                pos_mask = sign_d > 0
                neg_mask = ~pos_mask
                loss_real = torch.tensor(0.0, device=images.device)
                loss_fake = torch.tensor(0.0, device=images.device)
                if pos_mask.any():
                    loss_real = torch.nn.functional.relu(1.0 - d_out_map[pos_mask]).mean()
                if neg_mask.any():
                    loss_fake = torch.nn.functional.relu(1.0 + d_out_map[neg_mask]).mean()
                d_loss = 0.5 * (loss_real + loss_fake) if (pos_mask.any() and neg_mask.any()) else (loss_real + loss_fake)
                self.manual_backward(d_loss)
                opt_d.step()
                d_loss_val = float(d_loss.detach().item())

        # Logging
        self.log("loss_G", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("cls", cls_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("g_adv", g_adv_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if d_loss_val is not None:
            self.log("loss_D", d_loss_val, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return {"loss": g_loss}

    # --- D-queue helpers ---
    def _enqueue_d_batch(self, feat: torch.Tensor, sign: torch.Tensor) -> None:
        try:
            if self.d_queue_k <= 0:
                return
            self._d_feat_queue.append(feat)
            self._d_sign_queue.append(sign)
            # Keep only last K batches
            while len(self._d_feat_queue) > self.d_queue_k:
                self._d_feat_queue.pop(0)
                self._d_sign_queue.pop(0)
        except Exception:
            # Best-effort: don't break training on queue issues
            self._d_feat_queue = []
            self._d_sign_queue = []

    def _sample_d_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            if not self._d_feat_queue:
                return None
            feats = torch.cat(self._d_feat_queue, dim=0)
            signs = torch.cat(self._d_sign_queue, dim=0)
            total = feats.size(0)
            if total == 0:
                return None
            b_d = int(self.d_batch_size)
            if b_d <= 0:
                b_d = min(self.cfg.training.batch_size, total)
            # Sample with replacement if not enough cached
            if total >= b_d:
                idx = torch.randint(0, total, (b_d,), device=feats.device)
            else:
                idx = torch.randint(0, total, (b_d,), device=feats.device)
            return feats.index_select(0, idx), signs.index_select(0, idx)
        except Exception:
            return None

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        images, targets, _ = self._unpack_batch(batch)
        out = self.forward(images)
        logits, _ = self._parse_out(out)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        acc1 = (preds == targets).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc1", acc1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    # --- helpers for Lightning ---
    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            images, targets, d_bool = batch[0], batch[1], batch[2]
        else:
            images, targets = batch[0], batch[1]
            d_bool = torch.ones_like(targets, dtype=torch.bool)
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        d_bool = d_bool.to(self.device, non_blocking=True)
        return images, targets, d_bool

    def _parse_out(self, out) -> Tuple[torch.Tensor, Dict]:
        if isinstance(out, tuple):
            if len(out) >= 2 and isinstance(out[1], dict):
                logits = out[0]
                inter = out[1]
            else:
                logits = out[0]
                inter = {}
        else:
            logits = out  # type: ignore
            inter = {}
        return logits, inter

    def _feature_for_d(self, logits: torch.Tensor, inter: Dict) -> Optional[torch.Tensor]:
        try:
            d_source = (self.model_cfg.intermediate.d_source or "logits").lower()
        except Exception:
            d_source = "logits"

        feat_for_d: Optional[torch.Tensor]
        if isinstance(d_source, str) and d_source.startswith("target:") and isinstance(inter, dict):
            key = d_source.split(":", 1)[1].strip()
            feat_for_d = inter.get(key)
        else:
            feat_for_d = logits

        if isinstance(feat_for_d, torch.Tensor):
            if feat_for_d.dim() == 2:
                feat_for_d = feat_for_d.unsqueeze(-1).unsqueeze(-1)
            elif feat_for_d.dim() == 3:
                feat_for_d = feat_for_d.unsqueeze(-2)
            elif feat_for_d.dim() != 4:
                feat_for_d = feat_for_d.view(feat_for_d.size(0), -1).unsqueeze(-1).unsqueeze(-1)
        else:
            feat_for_d = None
        return feat_for_d

    # --- Discriminator support (training only) ---
    def _maybe_discriminate(self, out, intermediates: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute discriminator score based on config-selected feature.

        Returns a tensor of shape (N,) with a single score per sample, or None on failure.
        """
        try:
            src = (self.model_cfg.intermediate.d_source or "logits").strip()
            feat: Optional[torch.Tensor] = None
            # Unpack logits from possible tuples
            logits = out
            if isinstance(out, tuple):
                logits = out[0]

            if src.lower() == "logits":
                if isinstance(logits, torch.Tensor):
                    feat = logits
            elif src.lower().startswith("target:"):
                key = src.split(":", 1)[1].strip()
                feat = intermediates.get(key)

            if feat is None or not isinstance(feat, torch.Tensor):
                return None

            # Ensure N x C x H x W shape
            if feat.dim() == 2:
                feat = feat.unsqueeze(-1).unsqueeze(-1)  # N x C x 1 x 1
            elif feat.dim() == 4:
                pass
            elif feat.dim() == 3:
                # N x C x L -> treat L as spatial W, make H=1
                feat = feat.unsqueeze(-2)
            else:
                # unsupported
                return None

            n, c, h, w = feat.shape
            disc = self._get_or_init_discriminator(c)
            score_map = disc(feat)  # N x 1 x h' x w'
            # Global average to a single score per sample
            score = score_map.mean(dim=(1, 2, 3))  # N
            return score
        except Exception:
            return None

    def _get_or_init_discriminator(self, in_ch: int) -> nn.Module:
        device = next(self.parameters()).device
        # Read discriminator config if available
        try:
            dcfg = getattr(self.model_cfg, "discriminator", None)
            ndf = int(getattr(dcfg, "ndf", 64))
            n_layers = int(getattr(dcfg, "n_layers", 3))
            use_actnorm = bool(getattr(dcfg, "use_actnorm", False))
        except Exception:
            ndf, n_layers, use_actnorm = 64, 3, False

        def _make():
            return PatchDiscriminator(in_ch, ndf=ndf, n_layers=n_layers, use_actnorm=use_actnorm).to(device)

        if self._disc is None:
            self._disc = _make()
        else:
            ch_mismatch = getattr(self._disc, "in_ch", in_ch) != in_ch
            if ch_mismatch:
                self._disc = _make()
            else:
                self._disc = self._disc.to(device)
        return self._disc


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (N-layer) similar to pix2pix/CycleGAN.

    Follows the structure in your example: kernel=4, stride=2 downsampling layers with
    LeakyReLU(0.2), BatchNorm after the first layer, then a stride=1 conv block and a final
    conv to 1 channel prediction map.
    """

    def __init__(self, in_ch: int, ndf: int = 64, n_layers: int = 3, use_actnorm: bool = False):
        super().__init__()
        self.in_ch = in_ch
        kw = 4
        padw = 1

        sequence: list[nn.Module] = []
        # First layer: no norm
        sequence += [nn.Conv2d(in_ch, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=True)
            sequence += [conv]
            if use_actnorm:
                sequence += [ActNorm(ndf * nf_mult)]
            else:
                sequence += [nn.BatchNorm2d(ndf * nf_mult)]
            sequence += [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=True)
        sequence += [conv]
        if use_actnorm:
            sequence += [ActNorm(ndf * nf_mult)]
        else:
            sequence += [nn.BatchNorm2d(ndf * nf_mult)]
        sequence += [nn.LeakyReLU(0.2, True)]

        # Final 1-channel logits map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.main = nn.Sequential(*sequence)
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m: nn.Module):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1 and hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    # --- Lifecycle hooks ---
    def on_fit_start(self) -> None:
        """Log full config to Weights & Biases at run start if enabled."""
        try:
            from lightning.pytorch.loggers import WandbLogger  # type: ignore
        except Exception:
            WandbLogger = None  # type: ignore
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        try:
            if WandbLogger is not None and isinstance(logger, WandbLogger):
                run = logger.experiment
                try:
                    from dataclasses import asdict
                    cfg_dict = asdict(self.cfg)
                except Exception:
                    cfg_dict = {}
                try:
                    run.config.update(cfg_dict, allow_val_change=True)
                except Exception:
                    pass
        except Exception:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
