from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import random

import yaml
from torch.utils.data import Dataset


class AutoPivoitDataset(Dataset):
    """Wrap a torchvision-style classification dataset and attach a per-sample boolean pivot label.

    - Splits each class into two subsets with ratio mu and (1-mu) in a class-balanced way.
    - Saves/loads the pivot assignment to a YAML file for reproducibility.
    - __getitem__ returns (image, class_label, d_bool) where d_bool=True -> +1 target for D, False -> -1.

    Args:
        base: underlying dataset returning (image, label)
        mu: fraction per class assigned to True (+1). Must be in (0, 1).
        pivot_file: optional path to save/load pivot YAML. If exists, load; else create and save when save_pivot=True.
        seed: RNG seed for deterministic split.
        save_pivot: if True and pivot_file provided or can be derived, save the created split.
        pivot_dir: directory to save pivot file if pivot_file is not provided.
        name: optional dataset name used for auto-naming pivot file.
    """

    def __init__(
        self,
        base: Dataset,
        mu: float = 0.5,
        pivot_file: Optional[str] = None,
        seed: int = 42,
        save_pivot: bool = True,
        pivot_dir: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        assert 0.0 < mu < 1.0, "mu must be in (0, 1)"
        self.base = base
        self.mu = float(mu)
        self.seed = int(seed)
        self._signs: List[int]

        if pivot_file and os.path.isfile(pivot_file):
            self._load_pivot(pivot_file)
            self.pivot_file = pivot_file
        else:
            labels = self._get_all_labels()
            self._signs = self._build_balanced_pivot(labels, self.mu, self.seed)
            # Determine path for saving
            self.pivot_file = pivot_file or self._default_pivot_path(pivot_dir, name)
            if save_pivot:
                self._save_pivot(self.pivot_file, labels, self._signs, self.mu, self.seed)

    # -- Dataset protocol --
    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        item = self.base[idx]
        # Expect (image, label) structure; pass through extras if present
        if isinstance(item, tuple) and len(item) >= 2:
            image, label = item[0], int(item[1])
            d_bool = bool(self._signs[idx] == 1)
            return image, label, d_bool
        # Fallback: unrecognized structure -> return as-is with d_bool
        d_bool = bool(self._signs[idx] == 1)
        return item, d_bool

    # -- Internals --
    def _get_all_labels(self) -> List[int]:
        # Try common attributes for torchvision datasets
        if hasattr(self.base, "targets"):
            return list(map(int, getattr(self.base, "targets")))  # type: ignore[arg-type]
        if hasattr(self.base, "labels"):
            return list(map(int, getattr(self.base, "labels")))  # type: ignore[arg-type]
        # As a last resort, iterate once (may be slow)
        labels: List[int] = []
        for i in range(len(self.base)):
            item = self.base[i]
            if isinstance(item, tuple) and len(item) >= 2:
                labels.append(int(item[1]))
            else:
                raise RuntimeError("Unable to infer labels from base dataset; provide a dataset with .targets or (image, label) items.")
        return labels

    @staticmethod
    def _build_balanced_pivot(labels: Sequence[int], mu: float, seed: int) -> List[int]:
        n = len(labels)
        by_class: Dict[int, List[int]] = {}
        for idx, y in enumerate(labels):
            by_class.setdefault(int(y), []).append(idx)

        rng = random.Random(seed)
        signs = [-1] * n
        for y, idxs in by_class.items():
            idxs = list(idxs)
            rng.shuffle(idxs)
            k_true = int(round(mu * len(idxs)))
            # Clamp to [0, len]
            k_true = max(0, min(len(idxs), k_true))
            for i in idxs[:k_true]:
                signs[i] = 1
            for i in idxs[k_true:]:
                signs[i] = -1
        return signs

    def _default_pivot_path(self, pivot_dir: Optional[str], name: Optional[str]) -> str:
        base_dir = pivot_dir or os.path.join(os.getcwd(), "pivots")
        os.makedirs(base_dir, exist_ok=True)
        ds_name = name or self._infer_dataset_name()
        fname = f"pivot_{ds_name}_mu{self.mu:.3f}_seed{self.seed}.yaml"
        return os.path.join(base_dir, fname)

    def _infer_dataset_name(self) -> str:
        # Try to infer a meaningful name
        if hasattr(self.base, "root"):
            root = str(getattr(self.base, "root"))
            return os.path.basename(os.path.abspath(root)) or "dataset"
        return getattr(self.base, "__class__", type(self.base)).__name__.lower()

    def _save_pivot(self, path: str, labels: Sequence[int], signs: Sequence[int], mu: float, seed: int) -> None:
        data = {
            "version": 1,
            "timestamp": int(time.time()),
            "mu": float(mu),
            "seed": int(seed),
            "size": len(signs),
            "signs": list(map(int, signs)),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=True)

    def _load_pivot(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        signs = data.get("signs", None)
        if not isinstance(signs, list) or len(signs) != len(self.base):
            raise RuntimeError("Pivot file incompatible with current dataset length")
        self._signs = [int(s) for s in signs]
        self.mu = float(data.get("mu", self.mu))
        self.seed = int(data.get("seed", self.seed))

