"""B acceptance test: reproduce two of A's EntropyAE runs without changing A."""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import run_experiment
from src.logger import append_result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["18_Ionosphere.npz", "14_glass.npz"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--tolerance", type=float, default=0.03,
                   help="Allowed absolute AUC/AP difference on another machine")
    return p.parse_args()


def main():
    args = parse_args()
    out = ROOT / "results" / "analysis" / "b_validation.csv"
    if out.exists():
        out.unlink()  # one validation report per invocation, no duplicate appends
    a_file = ROOT / "results" / "baseline" / "baseline_results.csv"
    with a_file.open(encoding="utf-8", newline="") as f:
        a = {(r["dataset"], r["method"], int(r["seed"])): r for r in csv.DictReader(f)}
    failures = []
    for ds in args.datasets:
        for seed in args.seeds:
            key = (ds, "EntropyAE", seed)
            if key not in a:
                raise KeyError(f"Missing A baseline: {key}")
            result = run_experiment(ROOT / "data" / ds, seed=seed,
                                    epochs=args.epochs, method="EntropyAE")
            append_result(out, result.baseline_row())
            base = a[key]
            da = result.AUC - float(base["AUC"])
            dp = result.AP - float(base["AP"])
            print(f"{ds}, seed={seed}: AUC={result.AUC:.6f} (A: {float(base['AUC']):.6f}; Δ={da:+.6f}), "
                  f"AP={result.AP:.6f} (A: {float(base['AP']):.6f}; Δ={dp:+.6f}), "
                  f"best_step={result.stop_iter}, updates={result.total_iter}")
            if abs(da) > args.tolerance or abs(dp) > args.tolerance:
                failures.append(key)
    print(f"Output: {out}")
    if failures:
        print("WARNING: large differences need examination; possible GPU/library differences.")
        raise SystemExit(1)
    print("PASS: B framework is close to A's measured AUC/AP under the set tolerance.")


if __name__ == "__main__":
    main()
