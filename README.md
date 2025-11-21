**Overview**
- Minimal deep learning experiment framework for image classification.
- Single LightningModule `CFModel` builds the backbone and handles training.
- Supports `torchvision` and `transformers` backbones.
- Captures differentiable intermediate activations via a decorator and YAML-configured targets.
- Training-only PatchGAN discriminator computes a per-sample score from a configured feature source.

**Key Pieces**
- `cf_framework/CFModel` (LightningModule) builds the backbone and head from YAML and implements training.
- `cf_framework/hooks.py` provides `capture_intermediates` decorator returning `(logits, intermediates)`.
  - In training mode, it returns `(logits, intermediates, d_score)` where `d_score` is the discriminator output.
- `cf_framework/logging.py` logs losses to CSV in `training.log_dir` (if you want custom CSV logging).

**Config**
- See `configs/example_torchvision.yaml` (ResNet18) and `configs/example_transformers.yaml` (ViT).
- `model.intermediate.targets` lists dotted module names to tap during forward.
- `model.intermediate.feature_key` (transformers only): `pooler | cls | mean | last_hidden`.
 - `model.intermediate.d_source`: `"logits"` or `"target:<name>"` to choose the discriminator input feature.

**Usage Sketch**
- Install dependencies: `torch`, `lightning`, plus `torchvision` and/or `transformers` as needed.
- Create your dataloaders (e.g., using `torchvision.datasets.ImageFolder`).
- Initialize Lightning training:

```
from cf_framework import CFModel
from cf_framework.config import Config
import lightning as L
from lightning.pytorch.loggers import CSVLogger

cfg = Config.from_file("configs/example_torchvision.yaml")

module = CFModel(cfg, g_adv_weight=0.1, d_steps=1)
trainer = L.Trainer(max_epochs=cfg.training.epochs, logger=CSVLogger(cfg.training.log_dir, name="lightning"))
trainer.fit(module, train_dataloaders=your_train_loader, val_dataloaders=your_val_loader)
```

**AutoPivoitDataset Example**
- Wrap a torchvision dataset to generate reproducible per-class balanced pivots; feed to Lightning trainer with `CFModel`.

```
from cf_framework import CFModel, AutoPivoitDataset
from cf_framework.config import Config
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightning as L
from lightning.pytorch.loggers import CSVLogger

cfg = Config.from_file("configs/example_torchvision.yaml")
tfm = transforms.Compose([
    transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
    transforms.ToTensor(),
])
base = datasets.ImageFolder(cfg.data.root, transform=tfm)
loader = DataLoader(AutoPivoitDataset(base, mu=0.5, seed=42, pivot_file="pivots/example.yaml"),
                    batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers)

module = CFModel(cfg, g_adv_weight=0.1, d_steps=1)
trainer = L.Trainer(max_epochs=cfg.training.epochs, logger=CSVLogger(cfg.training.log_dir, name="lightning"))
trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
```

**CLI Training**
- If `--config` is not provided, the script loads `configs/default.yaml`.
- Train from a YAML config and save checkpoints + curves:

```
python train.py --config configs/example_torchvision.yaml \
  --mu 0.5 \
  --pivot-file pivots/resnet18_mu0.5.yaml \
  --device cuda \
  --epochs 5 \
  --log-dir runs/resnet18_exp \
  --ckpt-dir runs/resnet18_exp/ckpts \
  --g-adv-weight 0.1 \
  --history-len 64 \
  --d-steps 1 \
  --wandb --wandb-project cf-framework
```

- Artifacts:
  - Lightning logs in `<log_dir>/lightning` (CSV if using CSVLogger)
  - Checkpoints via Lightning callback

**Intermediate Activations**
- The model's `forward` is wrapped by a decorator that registers temporary forward hooks on modules listed under `model.intermediate.targets`.
- The returned value is `(logits, intermediates_dict)`. Example:

```
out = model(images)
if isinstance(out, tuple) and len(out) == 3:
    logits, inter, d_score = out
else:
    logits, inter = out
feat = inter.get("layer4.1.relu")
```

**Notes**
- Transformers vision models are called with `pixel_values=x` under the hood.
- If `transformers` or `torchvision` are not installed, helpful errors are raised on use.
