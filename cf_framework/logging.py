from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LossLogger:
    log_dir: str
    filename: str = "loss.csv"
    write_header: bool = True

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self._path = os.path.join(self.log_dir, self.filename)
        self._fp = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fp,
            fieldnames=["step", "epoch", "loss_G", "loss_D", "cls", "g_adv", "lr"]
        )
        if self.write_header:
            self._writer.writeheader()
            self._fp.flush()

    def log(self, step: int, epoch: int, loss: float, lr: float, extra: Optional[Dict]=None):
        # Map provided loss to generator loss for clarity
        row = {"step": step, "epoch": epoch, "loss_G": float(loss), "lr": float(lr)}
        if extra:
            row.update(extra)
        # File
        # Ensure writer knows new fields if any
        unknown = [k for k in row.keys() if k not in self._writer.fieldnames]
        if unknown:
            self._writer.fieldnames.extend(unknown)  # type: ignore
        self._writer.writerow(row)
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass
