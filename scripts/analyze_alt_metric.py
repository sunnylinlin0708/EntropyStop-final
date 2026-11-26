
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import run_experiment, read_dataset_list
from src.logger import CSVStepLogger
from src.alt_metric_stop import bss_metric, silhouette_metric


METRICS = {
    "BSSAE": bss_metric,
    "SilhouetteAE": silhouette_metric
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0, 1, 2, 3, 4])
    parser.add_argument("--out",
                        default="results/altmetric/analysis")

    args = parser.parse_args()

    datasets = args.datasets or read_dataset_list(
        ROOT / "configs" / "datasets.txt"
    )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    records = []

    for dataset in datasets:

        for seed in args.seeds:

            for method, metric_fn in METRICS.items():

                print(f"Running {dataset} {method} seed={seed}")

                log_path = out / "curves" / (
                    f"{Path(dataset).stem}_{method}_seed{seed}.csv"
                )

                if log_path.exists():
                    log_path.unlink()

                # 使用 B 的共用框架
                with CSVStepLogger(log_path) as logger:

                    result = run_experiment(
                        ROOT / "data" / dataset,

                        method=method,
                        seed=seed,

                        epochs=500,
                        batch_size=1024,
                        lr=0.001,

                        k=100,
                        R_down=0.01,
                        n_eval=1024,

                        metric_fn=metric_fn,

                        full_run=True,

                        step_logger=logger,
                        auc_every_step=False
                    )

                # 讀取完整訓練紀錄
                df = pd.read_csv(log_path)

                # 只保留有 AUC 的 iteration
                valid = df.dropna(
                    subset=["metric_value", "AUC"]
                ).copy()

                # B 的框架存的是負值，因此轉回正值
                metric_values = -valid["metric_value"]
                auc_values = valid["AUC"]

                # 計算 Pearson correlation
                if (
                    len(valid) < 2
                    or metric_values.nunique() < 2
                    or auc_values.nunique() < 2
                ):
                    r = np.nan
                else:
                    r, _ = pearsonr(
                        metric_values,
                        auc_values
                    )

                records.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "pearson_r": r,
                    "best_iter": result.stop_iter,
                    "stop_trigger_iter": result.stop_trigger_iter,
                    "total_iter": result.total_iter,
                    "AUC": result.AUC,
                    "AP": result.AP
                })

                # 每完成一組就儲存
                pd.DataFrame(records).to_csv(
                    out / "correlation.csv",
                    index=False
                )

                print(f"Pearson r = {r:.6f}")

    print("All correlation experiments completed.")


if __name__ == "__main__":
    main()
