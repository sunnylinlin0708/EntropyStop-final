"""C handoff entry: a plug-in stopping at a fixed 50 optimizer updates."""

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import FixedStepStopRule, run_experiment
from src.logger import CSVStepLogger, append_result

if __name__ == "__main__":
    data = ROOT / "data" / "18_Ionosphere.npz"
    log_path = ROOT / "results" / "analysis" / "fixed50_steps.csv"
    if log_path.exists():
        log_path.unlink()
    with CSVStepLogger(log_path) as logger:
        result = run_experiment(
            data, method="Fixed50AE", seed=0,
            stop_rule=FixedStepStopRule(max_steps=50),
            epochs=250, step_logger=logger,
        )
    append_result(ROOT / "results" / "analysis" / "fixed50_results.csv",
                  result.baseline_row())
    print(f"Fixed50: best_iter={result.stop_iter}, total_iter={result.total_iter}, "
          f"AUC={result.AUC:.6f}, AP={result.AP:.6f}")
    print(f"Step log: {log_path}")
