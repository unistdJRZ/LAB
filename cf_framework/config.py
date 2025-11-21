from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class IntermediateConfig:
    targets: List[str] = field(default_factory=list)
    # transformers-only: how to derive features from model outputs
    feature_key: str = "pooler"  # one of: pooler, cls, mean, last_hidden
    # discriminator feature source: "logits" or "target:<name>" where <name> is in targets
    d_source: str = "logits"


@dataclass
class DiscriminatorConfig:
    ndf: int = 64
    n_layers: int = 3
    use_actnorm: bool = False


@dataclass
class ModelConfig:
    provider: str  # "torchvision" | "transformers"
    name: str
    pretrained: bool = True
    num_classes: int = 1000
    intermediate: IntermediateConfig = field(default_factory=IntermediateConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    input_size: int = 224


@dataclass
class TrainConfig:
    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    epochs: int = 1
    batch_size: int = 32
    # GAN-related hyperparameters
    g_adv_weight: float = 0.1
    d_steps: int = 1
    # Warmup steps before enabling adversarial term
    starting_step: int = 0
    # Discriminator queue and batch size
    d_batch_size: int = 0  # 0 -> fallback to batch_size
    d_queue_k: int = 1     # number of recent batches to cache
    # Logging/run naming
    task_name: Optional[str] = None
    device: str = "cuda"
    log_dir: str = "runs/exp"
    # optional logging backends
    wandb: bool = False
    wandb_project: Optional[str] = None


@dataclass
class DataConfig:
    # source: "hf" or "imagefolder"
    source: str = "hf"
    # HF dataset options
    hf_name: str = "food101d"
    hf_split: str = "train"
    image_key: str = "image"
    label_key: str = "label"
    # ImageFolder root
    root: str = "data/images"
    # pivot assignment options
    mu: float = 0.5
    pivot_file: Optional[str] = None
    num_workers: int = 4


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)  # type: ignore
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @staticmethod
    def from_file(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        model_raw = raw.get("model", {})
        inter_raw = model_raw.get("intermediate", {})
        intermediate = IntermediateConfig(
            targets=inter_raw.get("targets", []) or [],
            feature_key=str(inter_raw.get("feature_key", "pooler")),
            d_source=str(inter_raw.get("d_source", "logits")),
        )

        disc_raw = model_raw.get("discriminator", {})
        discriminator = DiscriminatorConfig(
            ndf=int(disc_raw.get("ndf", 64)),
            n_layers=int(disc_raw.get("n_layers", 3)),
            use_actnorm=bool(disc_raw.get("use_actnorm", False)),
        )

        model = ModelConfig(
            provider=str(model_raw["provider"]).lower(),
            name=str(model_raw["name"]),
            pretrained=bool(model_raw.get("pretrained", True)),
            num_classes=int(model_raw.get("num_classes", 1000)),
            intermediate=intermediate,
            discriminator=discriminator,
            input_size=int(model_raw.get("input_size", 224)),
        )

        train_raw = raw.get("training", {})
        training = TrainConfig(
            optimizer=str(train_raw.get("optimizer", "adam")).lower(),
            lr=float(train_raw.get("lr", 1e-4)),
            weight_decay=float(train_raw.get("weight_decay", 0.0)),
            epochs=int(train_raw.get("epochs", 1)),
            batch_size=int(train_raw.get("batch_size", 32)),
            g_adv_weight=float(train_raw.get("g_adv_weight", 0.1)),
            d_steps=int(train_raw.get("d_steps", 1)),
            starting_step=int(train_raw.get("starting_step", 0)),
            d_batch_size=int(train_raw.get("d_batch_size", 0)),
            d_queue_k=int(train_raw.get("d_queue_k", 1)),
            task_name=(train_raw.get("task_name", None)),
            device=str(train_raw.get("device", "cuda")),
            log_dir=str(train_raw.get("log_dir", "runs")),
            wandb=bool(train_raw.get("wandb", False)),
            wandb_project=(train_raw.get("wandb_project", None)),
        )

        data_raw = raw.get("data", {})
        data = DataConfig(
            source=str(data_raw.get("source", "hf")).lower(),
            hf_name=str(data_raw.get("hf_name", "food101d")),
            hf_split=str(data_raw.get("hf_split", "train")),
            image_key=str(data_raw.get("image_key", "image")),
            label_key=str(data_raw.get("label_key", "label")),
            root=str(data_raw.get("root", "data/images")),
            mu=float(data_raw.get("mu", 0.5)),
            pivot_file=(data_raw.get("pivot_file", None)),
            num_workers=int(data_raw.get("num_workers", 4)),
        )

        return Config(model=model, training=training, data=data)
