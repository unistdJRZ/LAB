from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable
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
        image_key: str = "image",
        label_key: str = "label",
        transform: Optional[Callable] = None,
    ) -> None:
        assert 0.0 < mu < 1.0, "mu must be in (0, 1)"
        self.base = base
        self.mu = float(mu)
        self.seed = int(seed)
        self._signs: List[int]
        # HF-style support
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

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
        d_bool = bool(self._signs[idx] == 1)
        # Case 1: tuple/list like (image, label, ...)
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            image, label = item[0], self._to_int_label(item[1])
            return image, label, d_bool
        # Case 2: dict-like sample (e.g., HuggingFace datasets)
        if isinstance(item, dict) and self.label_key in item:
            label = self._to_int_label(item[self.label_key])
            image = item.get(self.image_key, None)
            # Lazy import PIL only if needed
            try:
                from PIL import Image  # type: ignore
            except Exception:
                Image = None  # type: ignore
            # Convert common image types to PIL for transforms
            if self.transform is not None:
                if image is not None:
                    image = self._to_pil(image)
                    image = self.transform(image)
            return image, label, d_bool
        # Fallback: unknown structure; return as-is with boolean
        return item, d_bool

    # -- Internals --
    def _get_all_labels(self) -> List[int]:
        # Try common attributes for torchvision datasets
        if hasattr(self.base, "targets"):
            return list(map(int, getattr(self.base, "targets")))  # type: ignore[arg-type]
        if hasattr(self.base, "labels"):
            return list(map(int, getattr(self.base, "labels")))  # type: ignore[arg-type]
        # Try HuggingFace datasets column extraction quickly
        try:
            # datasets.Dataset supports column access by key
            col = self.label_key if hasattr(self, "label_key") else "label"
            labels = self.base[col]  # type: ignore[index]
            return [self._to_int_label(y) for y in labels]
        except Exception:
            pass
        # As a last resort, iterate once (may be slow)
        labels: List[int] = []
        for i in range(len(self.base)):
            item = self.base[i]
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                labels.append(self._to_int_label(item[1]))
            elif isinstance(item, dict):
                key = getattr(self, "label_key", "label")
                if key in item:
                    labels.append(self._to_int_label(item[key]))
                else:
                    raise RuntimeError("Label key not found in sample dict; set label_key appropriately.")
            else:
                raise RuntimeError("Unable to infer labels from base dataset; provide a dataset with .targets or (image, label) items.")
        return labels

    def _to_int_label(self, y: Any) -> int:
        # Prefer ints and bools directly
        if isinstance(y, bool):
            return int(y)
        if isinstance(y, int):
            return y
        # Strings: attempt to use HF ClassLabel if present
        if isinstance(y, str):
            try:
                features = getattr(self.base, "features", None)
                key = getattr(self, "label_key", "label")
                if features is not None and key in features and hasattr(features[key], "str2int"):
                    return int(features[key].str2int(y))
            except Exception:
                pass
            # Fallback stable hash mapping (not ideal, but deterministic)
            return int(abs(hash(y)) % (10**9))
        # Other numeric types
        try:
            return int(y)
        except Exception:
            raise TypeError(f"Unsupported label type: {type(y)}")

    def _to_pil(self, img: Any):
        # Convert common HuggingFace image representations to PIL
        try:
            from PIL import Image  # type: ignore
        except Exception:
            Image = None  # type: ignore
        # PIL already
        if Image is not None:
            import PIL
            if isinstance(img, PIL.Image.Image):
                return img
        # dict with path or bytes
        if isinstance(img, dict):
            if "path" in img and Image is not None:
                return Image.open(img["path"]).convert("RGB")
            if "bytes" in img and Image is not None:
                from io import BytesIO
                return Image.open(BytesIO(img["bytes"]))
        # numpy array
        try:
            import numpy as np  # type: ignore
            if isinstance(img, np.ndarray):
                if Image is None:
                    return img
                if img.ndim == 2:
                    mode = "L"
                    arr = img
                elif img.ndim == 3 and img.shape[-1] == 1:
                    # squeeze single-channel HxWx1 to HxW (L)
                    mode = "L"
                    arr = img[..., 0]
                else:
                    mode = "RGB"
                    arr = img
                return Image.fromarray(arr.astype(np.uint8), mode=mode)
        except Exception:
            pass
        # torch tensor CxHxW
        try:
            import torch
            if isinstance(img, torch.Tensor) and img.dim() == 3 and img.shape[0] in (1, 3):
                np_img = img.permute(1, 2, 0).contiguous().cpu().numpy()
                import numpy as np  # type: ignore
                # If single-channel, squeeze last dim to 2D for 'L'
                if np_img.shape[-1] == 1:
                    np_img = np_img[..., 0]
                    mode = "L"
                else:
                    mode = "RGB"
                np_img = (np_img * 255).astype(np.uint8)
                return Image.fromarray(np_img, mode=mode) if Image is not None else np_img
        except Exception:
            pass
        return img

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
            # Match example logic: use floor via int(len * mu)
            # rather than rounding, to decide the supervised/pivoted subset size per class.
            k_true = int(mu * len(idxs))
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


def collate_pivot(batch):
    """Collate function that ensures 3-channel contiguous tensors and returns
    (images, labels: LongTensor, d_bool: BoolTensor).

    Placed at package level so it is picklable on Windows when using
    DataLoader with num_workers > 0.
    """
    import torch

    imgs, labels, dbools = [], [], []
    for sample in batch:
        if isinstance(sample, (list, tuple)) and len(sample) >= 3:
            img, y, d = sample[0], sample[1], sample[2]
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            img, y = sample[0], sample[1]
            d = True
        elif isinstance(sample, dict):
            img, y = sample.get("image"), sample.get("label", 0)
            d = sample.get("d", True)
        else:
            return torch.utils.data.default_collate(batch)
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            img = img.contiguous()
        imgs.append(img)
        labels.append(int(y))
        dbools.append(bool(d))
    if all(isinstance(x, torch.Tensor) for x in imgs):
        images = torch.stack([x.contiguous() for x in imgs], dim=0)
    else:
        images = imgs
    labels_t = torch.tensor(labels, dtype=torch.long)
    dbools_t = torch.tensor(dbools, dtype=torch.bool)
    return images, labels_t, dbools_t
