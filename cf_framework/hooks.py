from __future__ import annotations

from typing import Callable, Dict, List
import torch
import torch.nn as nn

from .utils import resolve_module


def capture_intermediates(targets_getter: Callable[[object], List[str]]):
    """Decorator factory to capture differentiable intermediate outputs during forward.

    - `targets_getter(self)` returns list of dotted module names inside `self.backbone`.
    - The wrapper registers forward hooks on those modules for this call only.
    - Returns a tuple: (original_forward_output, intermediates_dict)
      If the wrapped forward already returns a tuple, it is preserved as the first element.

    The captured tensors remain part of the computation graph and are differentiable.
    """

    def decorator(forward_fn):
        def wrapper(self, *args, **kwargs):
            targets = targets_getter(self) or []
            intermediates: Dict[str, torch.Tensor] = {}
            hooks: List[torch.utils.hooks.RemovableHandle] = []

            if hasattr(self, "backbone") and isinstance(self.backbone, nn.Module):
                for name in targets:
                    module = resolve_module(self.backbone, name)
                    if module is None:
                        continue

                    def make_hook(nm):
                        def _hook(mod, inp, out):
                            intermediates[nm] = out
                        return _hook

                    hooks.append(module.register_forward_hook(make_hook(name)))

            try:
                out = forward_fn(self, *args, **kwargs)
            finally:
                for h in hooks:
                    h.remove()

            # Optionally compute discriminator output only during training
            d_out = None
            if getattr(self, "training", False) and hasattr(self, "_maybe_discriminate"):
                try:
                    d_out = self._maybe_discriminate(out, intermediates)
                except Exception:
                    d_out = None

            if d_out is not None:
                return out, intermediates, d_out
            return out, intermediates

        return wrapper

    return decorator
