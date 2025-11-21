from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Inspect backbone and print I/O shapes for a dummy input.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to YAML config (defaults to configs/default.yaml)'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Use pretrained weights if available (may trigger downloads).'
    )
    return parser.parse_args()


def iter_children(module) -> Iterable[Tuple[str, object]]:
    return module._modules.items()  # type: ignore[attr-defined]


def print_tree(module, name: str = 'root', indent: int = 0) -> None:
    prefix = '  ' * indent
    cls_name = module.__class__.__name__
    print(f'{prefix}{name}: {cls_name}')
    for child_name, child in iter_children(module):
        if child is None:
            continue
        print_tree(child, name=child_name, indent=indent + 1)


def _build_torchvision_backbone(name: str, pretrained: bool):
    try:
        import torchvision as tv  # type: ignore
    except Exception:
        return None
    factory = getattr(tv.models, name)
    try:
        weights = None
        if pretrained:
            try:
                default_weights = getattr(tv.models, f'{name}_Weights').DEFAULT
                weights = default_weights
            except Exception:
                pass
        model = factory(weights=weights)
    except TypeError:
        model = factory(pretrained=pretrained)
    return model


def _build_transformers_backbone(name: str, pretrained: bool):
    try:
        import transformers as tr  # type: ignore
    except Exception:
        return None
    if pretrained:
        return tr.AutoModel.from_pretrained(name)
    cfg = tr.AutoConfig.from_pretrained(name)
    return tr.AutoModel.from_config(cfg)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_backbone(raw_cfg: Dict[str, Any], use_pretrained: bool):
    model_raw = raw_cfg.get('model', {})
    provider = str(model_raw.get('provider', 'torchvision')).lower()
    name = str(model_raw.get('name'))
    input_size = int(model_raw.get('input_size', 224))
    pretrained = bool(model_raw.get('pretrained', True)) if use_pretrained else False
    if provider == 'torchvision':
        return _build_torchvision_backbone(name, pretrained), provider, name, input_size
    elif provider == 'transformers':
        return _build_transformers_backbone(name, pretrained), provider, name, input_size
    else:
        return None, provider, name, input_size


def forward_and_print_shapes(model, provider: str, input_size: int) -> None:
    try:
        import torch  # type: ignore
    except Exception:
        print('PyTorch is not installed; cannot run forward pass.')
        return

    if model is None:
        print('Backbone library not installed; cannot enumerate modules.')
        return

    # 1) Print full module structure tree
    print('\nBackbone Structure:')
    print_tree(model)

    # 2) List depth-1 modules
    children = list(model.named_children())
    if not children:
        print('No top-level submodules found.')

    print('\nTop-Level Modules (name: class):')
    for name, m in children:
        print(f'- {name}: {m.__class__.__name__}')

    # Prepare input tensor [1, 3, 224, 224] (or configured size)
    x = torch.randn(1, 3, int(input_size), int(input_size))
    print(f'\nInput shape: {tuple(x.shape)}')

    # Register forward hooks on only depth-1 children
    captured: Dict[str, Any] = {}

    def _desc(out: Any) -> Any:
        try:
            if hasattr(torch, 'is_tensor') and torch.is_tensor(out):
                return tuple(out.shape)
            if isinstance(out, (list, tuple)):
                return [
                    tuple(t.shape) if hasattr(torch, 'is_tensor') and torch.is_tensor(t) else type(t).__name__
                    for t in out
                ]
            # HuggingFace outputs or similar
            if hasattr(out, 'last_hidden_state'):
                return {'last_hidden_state': tuple(out.last_hidden_state.shape)}
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                return {'pooler_output': tuple(out.pooler_output.shape)}
            if hasattr(out, 'shape'):
                try:
                    return tuple(out.shape)  # type: ignore[arg-type]
                except Exception:
                    return str(out)
            return type(out).__name__
        except Exception:
            return '<unavailable>'

    hooks = []
    for name, m in children:
        try:
            h = m.register_forward_hook(lambda mod, inp, out, n=name: captured.__setitem__(n, _desc(out)))
            hooks.append(h)
        except Exception:
            pass

    # Run a forward pass
    model.eval()
    with torch.no_grad():
        try:
            if provider == 'torchvision':
                _ = model(x)
            elif provider == 'transformers':
                _ = model(pixel_values=x)
            else:
                print('Unsupported provider; cannot run forward.')
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

    print('\nTop-Level Outputs (name: shape):')
    for name, _ in children:
        val = captured.get(name, '<no output captured>')
        print(f'- {name}: {val}')


def main() -> None:
    args = parse_args()
    raw = load_config(args.config)
    model, provider, name, input_size = build_backbone(raw, use_pretrained=args.pretrained)

    print(f'Backbone Provider: {provider}')
    print(f'Backbone Name: {name}')

    forward_and_print_shapes(model, provider, input_size)


if __name__ == '__main__':
    main()
