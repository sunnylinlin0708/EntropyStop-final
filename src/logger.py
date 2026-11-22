"""CSV logging helpers for the B part of EntropyStop.

The shared per-update format deliberately stays identical to the team contract:
dataset,method,seed,iter,metric_value,AUC,AP.

`iter` is a **zero-based mini-batch/update index**, not an epoch number.
AUC/AP are evaluated post hoc only, never used in the stop rule.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

STEP_FIELDS = ["dataset", "method", "seed", "iter", "metric_value", "AUC", "AP"]
BASELINE_FIELDS = [
    "dataset", "method", "seed", "lr", "k", "R_down", "extra_param",
    "stop_iter", "total_iter", "AUC", "AP", "train_time_sec",
]


class CSVStepLogger:
    def __init__(self, filename: str | Path) -> None:
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists() or self.path.stat().st_size == 0
        self.file = self.path.open("a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=STEP_FIELDS)
        if new:
            self.writer.writeheader()
        self.file.flush()

    def log(
        self, dataset: str, method: str, seed: int, iteration: int,
        metric_value: float, auc: float | None = None, ap: float | None = None,
    ) -> None:
        self.writer.writerow({
            "dataset": dataset, "method": method, "seed": seed,
            "iter": iteration, "metric_value": f"{metric_value:.12g}",
            "AUC": "" if auc is None else f"{auc:.12g}",
            "AP": "" if ap is None else f"{ap:.12g}",
        })

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()

    def __enter__(self) -> "CSVStepLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def append_result(path: str | Path, record: dict[str, Any]) -> None:
    """Append a result using A's exact column order (including total_iter)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    new = not target.exists() or target.stat().st_size == 0
    with target.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASELINE_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in BASELINE_FIELDS})
