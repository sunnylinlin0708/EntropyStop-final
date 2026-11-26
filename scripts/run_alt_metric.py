
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import run_experiment, read_dataset_list
from src.logger import CSVStepLogger, append_result

from src.alt_metric_stop import (
    bss_metric,
    silhouette_metric
)


METRICS = {
    "BSSAE": bss_metric,
    "SilhouetteAE": silhouette_metric
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs="+", default=None)

    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[0, 1, 2, 3, 4]
    )

    parser.add_argument(
        "--methods", nargs="+",
        choices=list(METRICS),
        default=list(METRICS)
    )

    parser.add_argument("--epochs", type=int, default=250)

    parser.add_argument(
        "--out", default="results/altmetric"
    )

    args = parser.parse_args()

    datasets = args.datasets or read_dataset_list(
        ROOT / "configs" / "datasets.txt"
    )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:

        for method in args.methods:

            for seed in args.seeds:

                print(
                    f"Running {dataset} | "
                    f"{method} | seed={seed}"
                )

                log_path = (
                    out / "curves" /
                    f"{Path(dataset).stem}_{method}_seed{seed}.csv"
                )

                if log_path.exists():
                    log_path.unlink()

                with CSVStepLogger(log_path) as logger:

                    result = run_experiment(
                        ROOT / "data" / dataset,
                        method=method,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=1024,
                        lr=0.001,
                        k=100,
                        R_down=0.01,
                        n_eval=1024,
                        metric_fn=METRICS[method],
                        step_logger=logger
                    )

                row = result.baseline_row()

                row["extra_param"] += (
                    ";cluster=exact_1d_kmeans"
                )

                append_result(
                    out / "altmetric_results.csv",
                    row
                )

                print(
                    f"AUC={result.AUC:.6f}, "
                    f"AP={result.AP:.6f}, "
                    f"best_iter={result.stop_iter}, "
                    f"total_iter={result.total_iter}"
                )


if __name__ == "__main__":
    main()
