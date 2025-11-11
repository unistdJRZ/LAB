**Overview**
- Minimal deep learning experiment framework for image classification.
- Supports `torchvision` and `transformers` backbones via a single `CFModel`.
- Captures differentiable intermediate activations via a decorator and YAML-configured targets.
- Logs loss to console and CSV via `LossLogger`.
 - Training-only PatchGAN discriminator computes a per-sample score from a configured feature source.

**Key Pieces**
- `cf_framework/CFModel` builds the backbone and head from YAML.
- `cf_framework/hooks.py` provides `capture_intermediates` decorator returning `(logits, intermediates)`.
  - In training mode, it returns `(logits, intermediates, d_score)` where `d_score` is the discriminator output.
- `cf_framework/logging.py` logs losses to CSV in `training.log_dir`.
- `cf_framework/trainer.py` includes a simple `train_one_epoch` loop.

**Config**
- See `configs/example_torchvision.yaml` (ResNet18) and `configs/example_transformers.yaml` (ViT).
- `model.intermediate.targets` lists dotted module names to tap during forward.
- `model.intermediate.feature_key` (transformers only): `pooler | cls | mean | last_hidden`.
 - `model.intermediate.d_source`: `"logits"` or `"target:<name>"` to choose the discriminator input feature.

**Usage Sketch**
- Install dependencies: `torch`, plus `torchvision` and/or `transformers` as needed.
- Create your dataloaders (e.g., using `torchvision.datasets.ImageFolder`).
- Initialize and train:

```
from cf_framework import CFModel, LossLogger, train_one_epoch
from cf_framework.config import Config
import torch
from torch import nn

cfg = Config.from_file("configs/example_torchvision.yaml")
device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

model = CFModel(cfg).to(device)

# Dummy dataset example
dataset = torch.utils.data.TensorDataset(
    torch.randn(64, 3, cfg.model.input_size, cfg.model.input_size),
    torch.randint(0, cfg.model.num_classes, (64,))
)
loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.training.batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
criterion = nn.CrossEntropyLoss()
logger = LossLogger(cfg.training.log_dir)

for epoch in range(1, cfg.training.epochs + 1):
    avg_loss = train_one_epoch(model, loader, optimizer, criterion, device, epoch, logger)
    print("epoch", epoch, "avg_loss", avg_loss)

logger.close()
```

**AutoPivoitDataset Example**
- Wrap a torchvision dataset to generate reproducible per-class balanced pivots and train G/F vs D adversarially.

```
from cf_framework import CFModel, LossLogger, train_one_epoch, AutoPivoitDataset
from cf_framework.config import Config
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

cfg = Config.from_file("configs/example_torchvision.yaml")
device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

# Base dataset (ImageFolder-style)
tfm = transforms.Compose([
    transforms.Resize((cfg.model.input_size, cfg.model.input_size)),
    transforms.ToTensor(),
])
base = datasets.ImageFolder(cfg.data.root, transform=tfm)

# Wrap with AutoPivoitDataset to add d_bool (True/False) per-sample
wrapped = AutoPivoitDataset(base, mu=0.5, seed=42, pivot_file="pivots/example.yaml")
loader = DataLoader(wrapped, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers)

model = CFModel(cfg).to(device)

# Build G optimizer over model params
g_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
criterion = nn.CrossEntropyLoss()
logger = LossLogger(cfg.training.log_dir)

# Initialize D optimizer by probing a batch to infer discriminator input channels
first_images, first_y, first_d = next(iter(loader))
first_images = first_images.to(device)
model.train()
with torch.no_grad():
    out = model(first_images)
    # Derive D feature the same way trainer does
    inter = out[1] if (isinstance(out, tuple) and len(out) >= 2 and isinstance(out[1], dict)) else {}
    d_source = getattr(getattr(model, "model_cfg", None).intermediate, "d_source", "logits").lower()
    feat = inter.get(d_source.split(":",1)[1]) if d_source.startswith("target:") else (out[0] if isinstance(out, tuple) else out)
    if feat.dim() == 2: feat = feat.unsqueeze(-1).unsqueeze(-1)
    if feat.dim() == 3: feat = feat.unsqueeze(-2)
    in_ch = feat.shape[1]
disc = model._get_or_init_discriminator(in_ch).to(device)
d_optimizer = torch.optim.Adam(disc.parameters(), lr=cfg.training.lr)

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
        g_adv_weight=0.1,
        history_len=64,
        d_steps=1,
    )
    print("epoch", epoch, "avg_loss", avg)

logger.close()
```

**CLI Training**
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
  - CSV logs: `<log_dir>/loss.csv`
  - Plots: `<log_dir>/training_curve.png`
  - Checkpoints: `<ckpt_dir>/epoch_*.pt`, `last.pt`, `best.pt`

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
