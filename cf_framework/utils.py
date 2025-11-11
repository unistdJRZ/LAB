from __future__ import annotations

from typing import Optional
import torch.nn as nn


def resolve_module(root: nn.Module, dotted_name: str) -> Optional[nn.Module]:
    """Resolve a child module by dotted path from a root module.

    Example: "layer4.2.relu" on a ResNet.
    Returns None if not found.
    """
    current: nn.Module = root
    if not dotted_name:
        return None
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            # try numeric index for Sequential-like modules
            try:
                idx = int(part)
                if isinstance(current, (nn.Sequential, nn.ModuleList)) and idx < len(current):
                    current = current[idx]
                    continue
                return None
            except ValueError:
                return None
        current = getattr(current, part)
        if not isinstance(current, nn.Module):
            return None
    return current


def find_and_replace_classifier(model: nn.Module, num_classes: int) -> bool:
    """Attempt to find and replace the final classifier layer with a new Linear.

    Handles common torchvision patterns: ResNet(fc), DenseNet(classifier),
    EfficientNet(classifier Sequential), MobileNet(classifier Sequential),
    ViT(heads.head) etc.
    Returns True if replacement succeeded.
    """
    # 1) Direct attributes by common names
    for attr in ["fc", "classifier", "head", "heads"]:
        if hasattr(model, attr):
            module = getattr(model, attr)
            # heads may contain another 'head'
            if hasattr(module, "head") and isinstance(module.head, nn.Linear):
                in_features = module.head.in_features
                module.head = nn.Linear(in_features, num_classes)
                return True
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                setattr(model, attr, nn.Linear(in_features, num_classes))
                return True
            # If it's a Sequential, replace last Linear
            if isinstance(module, nn.Sequential) and len(module) > 0:
                # find last Linear
                for i in range(len(module) - 1, -1, -1):
                    if isinstance(module[i], nn.Linear):
                        in_features = module[i].in_features
                        module[i] = nn.Linear(in_features, num_classes)
                        return True
    # 2) Try to walk modules and replace last Linear with out_features==num_classes? Not reliable.
    # Instead replace the last Linear encountered.
    last_linear_name = None
    last_linear = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear_name, last_linear = name, m
    if last_linear is not None and last_linear_name is not None:
        # Resolve parent and index
        parts = last_linear_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        in_features = last_linear.in_features
        new_linear = nn.Linear(in_features, num_classes)
        if isinstance(parent, nn.Sequential):
            idx = int(leaf)
            parent[idx] = new_linear
        else:
            setattr(parent, leaf, new_linear)
        return True
    return False

