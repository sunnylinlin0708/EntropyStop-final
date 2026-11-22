"""Reusable AE training + interchangeable early-stop interface for EntropyStop.

Integration rules:
  * Put this file into EntropyStop-final/src/common.py.
  * Import the author's Autoencoder/weights_init_normal/cal_entropy unchanged.
  * Match A's run_baseline.py RNG and minibatch-update ordering.
  * Training and stop decisions use ONLY unlabeled reconstruction loss.
  * Labels are accessed only by post-hoc AUC/AP and failure analysis.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.logger import CSVStepLogger


# Receive one scalar metric at a time; report if training should stop.
# Example replacements: fixed-step stop, BSS/TSS, EMA-smoothed entropy.
class StopRule(Protocol):
    best_iter: int

    def step(self, metric_value: float) -> bool: ...


class EntropyStopRule:
    """Strict scalar-metric translation of the author's ModelEntropyEarlyStop.

    For `metric_fn=cal_entropy` it follows the same decision logic.
    `best_iter` is 0-based; `stop_trigger_iter` is the update at which
    `k` consecutive non-improvements have accumulated.
    """

    def __init__(self, k: int = 100, R_down: float = 0.01):
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k
        self.R_down = R_down
        self.entropys: list[float] = []  # same naming as author for comparisons
        self.patience = 0
        self.best_iter = 0
        self.minEn: float | None = None
        self.G = 0.0
        self.is_stop = False
        self.stop_trigger_iter: int | None = None
        self.improved = False

    def step(self, metric_value: float) -> bool:
        self.improved = False
        if self.is_stop:
            return True
        value = float(metric_value)
        self.entropys.append(value)
        iteration = len(self.entropys) - 1
        if self.minEn is None:
            self.minEn = value
            self.improved = True
            return False
        self.G += abs(value - self.entropys[-2])
        if value < self.minEn and (self.minEn - value) / self.G > self.R_down:
            self.minEn = value
            self.best_iter = iteration
            self.patience = 0
            self.G = 0.0
            self.improved = True
        else:
            self.patience += 1
        if self.patience >= self.k:
            self.is_stop = True
            self.stop_trigger_iter = iteration
        return self.is_stop


class FixedStepStopRule:
    """Minimal example for C: stop on a predeclared number of updates."""

    def __init__(self, max_steps: int = 50):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = max_steps
        self.best_iter = 0
        self.stop_trigger_iter: int | None = None
        self.improved = False
        self.count = 0

    def step(self, metric_value: float) -> bool:
        del metric_value  # rule does not inspect metric or any labels
        self.count += 1
        self.best_iter = self.count - 1
        self.improved = True
        if self.count >= self.max_steps:
            self.stop_trigger_iter = self.count - 1
            return True
        return False


@dataclass
class TrainingResult:
    dataset: str
    method: str
    seed: int
    lr: float
    k: int | str
    R_down: float | str
    extra_param: str
    stop_iter: int | str   # A's convention: BEST snapshot/update (not trigger time)
    total_iter: int
    AUC: float
    AP: float
    train_time_sec: float
    stop_trigger_iter: int | None
    actual_stop: bool
    n_samples: int
    steps_per_epoch: int
    best_scores: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)

    def baseline_row(self) -> dict:
        # Keep the same 12 CSV columns as A's baseline_results.csv.
        return {
            "dataset": self.dataset, "method": self.method, "seed": self.seed,
            "lr": self.lr, "k": self.k, "R_down": self.R_down,
            "extra_param": self.extra_param, "stop_iter": self.stop_iter,
            "total_iter": self.total_iter, "AUC": round(self.AUC, 6),
            "AP": round(self.AP, 6), "train_time_sec": round(self.train_time_sec, 4),
        }


def read_dataset_list(filename: str | Path) -> list[str]:
    entries = []
    for line in Path(filename).read_text(encoding="utf-8").splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            entries.append(name if name.endswith(".npz") else f"{name}.npz")
    return entries


def load_dataset(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Equivalent to A's exp_utils.dataLoading (scaler fits entire dataset)."""
    with np.load(filename) as data:
        x = np.asarray(data["X"])
        y = np.asarray(data["y"]).reshape(-1)
    return StandardScaler().fit_transform(x), y


def set_seed(seed: int) -> None:
    # Match A's exp_utils.set_seed for run-to-run parity.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def run_experiment(
    dataset_path: str | Path,
    *,
    method: str = "EntropyAE",
    seed: int = 0,
    epochs: int = 250,
    batch_size: int = 1024,
    lr: float = 0.001,
    k: int = 100,
    R_down: float = 0.01,
    n_eval: int = 1024,
    metric_fn: Callable[[np.ndarray], float] | None = None,
    stop_rule: StopRule | None = None,
    full_run: bool = False,
    step_logger: CSVStepLogger | None = None,
    auc_every_step: bool = False,
) -> TrainingResult:
    """Run A-compatible AE training; plug in a new metric or stop rule.

    Baseline mode: `full_run=False`, epochs=250, method='EntropyAE'.
    Analysis mode: `full_run=True`, epochs=500, step_logger=... .

    The stop metric is computed after EVERY optimizer update, exactly like A.
    AUC/AP are optionally logged every update (expensive) or after each epoch
    (fast). For fixed-run analysis, monitoring continues until the first stop
    trigger, then training continues without affecting that selected checkpoint.
    """
    from DeepODModel.AE.SimpleAE import Autoencoder
    from EntropyEarlyStop import cal_entropy
    from exp_utils import weights_init_normal

    if epochs < 1 or batch_size < 1 or n_eval < 1:
        raise ValueError("epochs, batch_size, n_eval must all be >= 1")
    is_vanilla = method == "VanillaAE"
    if is_vanilla and (metric_fn is not None or stop_rule is not None):
        raise ValueError("VanillaAE should not receive a stop method")

    ds = Path(dataset_path)
    x, y = load_dataset(ds)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(x).type(torch.FloatTensor).to(device)
    random_index = np.arange(X.size(0))
    np.random.shuffle(random_index)
    train_X = X[random_index]

    model = Autoencoder(X.size(1)).to(device)
    model.apply(weights_init_normal)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rule = None if is_vanilla else (stop_rule or EntropyStopRule(k=k, R_down=R_down))
    metric_fn = metric_fn or cal_entropy
    if not is_vanilla:
        n_subset = min(n_eval, X.shape[0])
        eval_index = np.random.choice(X.shape[0], n_subset, replace=False)
        x_eval = X[eval_index]

    best_state = None
    total_steps = 0
    actual_stop = False
    stop_trigger = None
    steps_per_epoch = (len(x) + batch_size - 1) // batch_size
    start = time.perf_counter()

    for e in range(epochs):
        for i in range(0, X.size(0), batch_size):
            model.train()
            recon_score = model(train_X[i:i + batch_size])
            loss = torch.mean(recon_score)
            opt.zero_grad()
            loss.backward()
            opt.step()

            metric = None
            if not is_vanilla:
                with torch.no_grad():
                    model.eval()
                    eval_losses = model(x_eval).cpu().detach().numpy()
                    metric = float(metric_fn(eval_losses))
                    model.train()
                # After trigger, keep logging entropy while leaving the rule frozen.
                if not actual_stop:
                    should_stop = rule.step(metric)
                    if getattr(rule, "improved", False):
                        best_state = copy.deepcopy(model.state_dict())
                    if should_stop:
                        actual_stop = True
                        stop_trigger = total_steps
                # Avoid using labels to update the stop rule.

            if step_logger is not None:
                auc, ap = None, None
                is_epoch_end = i + batch_size >= X.size(0)
                if auc_every_step or is_epoch_end:
                    with torch.no_grad():
                        model.eval()
                        scores = model(X).cpu().detach().numpy()
                        auc = float(roc_auc_score(y, scores))
                        ap = float(average_precision_score(y, scores))
                step_logger.log(ds.name, method, seed, total_steps,
                                float("nan") if metric is None else metric, auc, ap)
            total_steps += 1
            if actual_stop and not full_run:
                break
        if actual_stop and not full_run:
            break

    train_time = time.perf_counter() - start
    if not is_vanilla:
        if best_state is None:
            raise RuntimeError("Stop rule must set `improved=True` on its first update")
        model.load_state_dict(best_state)

    with torch.no_grad():
        model.eval()
        scores = model(X).cpu().detach().numpy()
    auc = float(roc_auc_score(y, scores))
    ap = float(average_precision_score(y, scores))
    return TrainingResult(
        dataset=ds.name, method=method, seed=seed, lr=lr,
        k=k if not is_vanilla else "", R_down=R_down if not is_vanilla else "",
        extra_param=f"batch_size={batch_size};epoch={epochs}",
        stop_iter=(rule.best_iter if not is_vanilla else ""),
        total_iter=total_steps, AUC=auc, AP=ap,
        train_time_sec=train_time, stop_trigger_iter=stop_trigger,
        actual_stop=actual_stop, n_samples=len(y), steps_per_epoch=steps_per_epoch,
        best_scores=scores, labels=y,
    )


def false_normal_ratio(scores: np.ndarray, y: np.ndarray) -> float:
    """For post-hoc failure analysis ONLY. Do not call within any stop rule.

    Definition in the handoff: fraction of normal-labeled examples whose loss
    exceeds the mean loss of all anomaly-labeled examples.
    """
    normal = scores[np.asarray(y) == 0]
    abnormal = scores[np.asarray(y) == 1]
    if len(normal) == 0 or len(abnormal) == 0:
        return float("nan")
    return float(np.mean(normal > abnormal.mean()))
