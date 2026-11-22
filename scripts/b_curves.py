"""B: 500-epoch full training curves, entropy/AUC Pearson r, failure analysis.

Run SMALL first:
  python scripts/b_curves.py --datasets 18_Ionosphere.npz 14_glass.npz --seeds 0 --epochs 500
Run complete datasets × 5 seeds:
  python scripts/b_curves.py --seeds 0 1 2 3 4 --epochs 500

As in A, EntropyStop's metric is measured EACH MINIBATCH, so `iter` denotes
optimizer update. For performance, AUC/AP are sampled at the end of each epoch.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import false_normal_ratio, read_dataset_list, run_experiment
from src.logger import CSVStepLogger


def pearson_r(values_x, values_y):
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_curves(rows, result, out_path):
    steps = np.array([int(row["iter"]) for row in rows])
    entropy = np.array([float(row["metric_value"]) for row in rows])
    auc_steps = np.array([int(row["iter"]) for row in rows if row["AUC"]])
    aucs = np.array([float(row["AUC"]) for row in rows if row["AUC"]])

    fig, left = plt.subplots(figsize=(10, 5))
    left.plot(steps, entropy, color="tab:orange", linewidth=1.1, label="Loss entropy")
    left.set_xlabel("Optimizer update (zero-based)")
    left.set_ylabel("Loss entropy", color="tab:orange")
    right = left.twinx()
    right.plot(auc_steps, aucs, color="tab:blue", linewidth=1.2,
               label="AUC (epoch-end evaluation)")
    right.set_ylabel("AUC (post-hoc only)", color="tab:blue")
    left.axvline(result.stop_iter, color="tab:green", linestyle="--",
                 label=f"Selected best update {result.stop_iter}")
    if result.stop_trigger_iter is not None:
        left.axvline(result.stop_trigger_iter, color="tab:red", linestyle=":",
                     label=f"Would trigger stop {result.stop_trigger_iter}")
    left.legend(loc="best", fontsize=8)
    fig.suptitle(f"{result.dataset} | seed={result.seed} | full-run curve")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_loss_hist(result, out_path):
    normal = result.best_scores[result.labels == 0]
    anomaly = result.best_scores[result.labels == 1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Histogram is strictly post-hoc analysis; never for hyperparameter tuning.
    ax.hist(normal, bins=35, alpha=0.65, density=True, label="Normal (true label)")
    ax.hist(anomaly, bins=35, alpha=0.55, density=True, label="Anomaly (true label)")
    ax.set_xlabel("Reconstruction loss at selected checkpoint")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(f"Post-hoc loss overlap: {result.dataset}, seed {result.seed}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def categorize(r, std_auc, auc_gap):
    # These are predeclared descriptive heuristics, NOT settings selected by AUC.
    if std_auc < 0.02:
        return "AUC almost unchanged"
    if np.isfinite(r) and r <= -0.5 and auc_gap <= 0.03:
        return "Strong negative correlation; selected AUC close to peak"
    return "Weak / positive correlation or inaccurate stopping"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Default: all entries in configs/datasets.txt")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--rdown", type=float, default=0.01)
    parser.add_argument("--n-eval", type=int, default=1024)
    parser.add_argument("--auc-every-step", action="store_true",
                        help="Slow: evaluate all samples on every optimizer update")
    args = parser.parse_args()
    ds_list = args.datasets or read_dataset_list(ROOT / "configs" / "datasets.txt")
    analysis_dir = ROOT / "results" / "analysis"
    curve_dir = analysis_dir / "curves"
    figure_dir = ROOT / "figures" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for ds in ds_list:
        for seed in args.seeds:
            path = curve_dir / f"{Path(ds).stem}_seed{seed}.csv"
            if path.exists():
                path.unlink()  # Avoid duplicate samples on re-run.
            print(f"Full curves: {ds}, seed={seed}, epochs={args.epochs}", flush=True)
            with CSVStepLogger(path) as step_logger:
                result = run_experiment(
                    ROOT / "data" / ds, method="EntropyAE", seed=seed,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    k=args.k, R_down=args.rdown, n_eval=args.n_eval,
                    full_run=True, step_logger=step_logger,
                    auc_every_step=args.auc_every_step,
                )
            with path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            sampled = [r for r in rows if r["AUC"]]
            sampled_auc = np.array([float(r["AUC"]) for r in sampled])
            r = pearson_r([float(z["metric_value"]) for z in sampled], sampled_auc)
            auc_gap = max(0.0, float(sampled_auc.max()) - result.AUC)
            auc_std = float(sampled_auc.std())
            false_ratio = false_normal_ratio(result.best_scores, result.labels)
            summary.append({
                "dataset": ds, "method": "EntropyAE", "seed": seed,
                "epochs": args.epochs, "steps_per_epoch": result.steps_per_epoch,
                "total_updates": result.total_iter, "best_iter": result.stop_iter,
                "stop_trigger_iter": ("" if result.stop_trigger_iter is None else result.stop_trigger_iter),
                "would_stop_early": result.actual_stop,
                "pearson_r_entropy_auc": r,
                "auc_at_selected_checkpoint": result.AUC,
                "best_epoch_sampled_auc": float(sampled_auc.max()),
                "auc_gap_vs_sampled_peak": auc_gap,
                "auc_std_over_epochs": auc_std,
                "false_normal_ratio": false_ratio,
                "category": categorize(r, auc_std, auc_gap),
            })
            stem = Path(ds).stem + f"_seed{seed}"
            plot_curves(rows, result, figure_dir / f"{stem}_entropy_auc.png")
            plot_loss_hist(result, figure_dir / f"{stem}_loss_hist.png")
            print(f"  r={r:.4f}, selected AUC={result.AUC:.4f}, "
                  f"false-normal={false_ratio:.2%}, best_step={result.stop_iter}", flush=True)

    output = analysis_dir / "correlation_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = [float(s["pearson_r_entropy_auc"]) for s in summary]
    values = [v for v in values if np.isfinite(v)]
    if values:
        ax.hist(values, bins=np.linspace(-1, 1, 21), edgecolor="black", alpha=0.75)
    ax.set_xlabel("Pearson r (loss entropy vs. AUC; epoch-end samples)")
    ax.set_ylabel("Number of dataset × seed runs")
    ax.set_title("Distribution of entropy–AUC correlation")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "pearson_r_distribution.png", dpi=160)
    plt.close(fig)
    print(f"Summary: {output}")
    print(f"Figures: {figure_dir}")


if __name__ == "__main__":
    main()
