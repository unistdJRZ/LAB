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
    device: str = "cuda"
    log_dir: str = "runs/exp"


@dataclass
class DataConfig:
    # Generic dataset root for ImageFolder; users provide data layout externally
    root: str = "data/images"
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
            device=str(train_raw.get("device", "cuda")),
            log_dir=str(train_raw.get("log_dir", "runs/exp")),
        )

        data_raw = raw.get("data", {})
        data = DataConfig(
            root=str(data_raw.get("root", "data/images")),
            num_workers=int(data_raw.get("num_workers", 4)),
        )

        return Config(model=model, training=training, data=data)
