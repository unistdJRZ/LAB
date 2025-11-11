from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

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


class CFModel(nn.Module):
    """Configurable image classification model supporting torchvision and transformers backbones.

    - Loads a backbone per YAML config
    - Replaces/attaches a classification head to match `num_classes`
    - Uses a decorator to capture differentiable intermediate outputs specified via config
    """

    def __init__(self, config: Config | ModelConfig):
        super().__init__()
        if isinstance(config, Config):
            self.model_cfg = config.model
        else:
            self.model_cfg = config

        provider = self.model_cfg.provider.lower()
        if provider not in {"torchvision", "transformers"}:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider
        self.num_classes = int(self.model_cfg.num_classes)
        self.backbone: nn.Module
        self.classifier: nn.Module | None = None
        self._disc: Optional[nn.Module] = None  # lazy PatchGAN discriminator

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
