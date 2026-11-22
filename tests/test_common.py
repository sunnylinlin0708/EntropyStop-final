"""Pure-python tests; does not require torch-geometric or the ADBench .npz files."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import EntropyStopRule, FixedStepStopRule, false_normal_ratio
from EntropyEarlyStop import EntropyEarlyStop
import numpy as np


def test_rule_identical_to_authors_scalar_stop():
    series = [8., 7.9, 7.8, 8.1, 8.2, 7.3, 7.4, 7.5, 7.6, 7.7]
    ref = EntropyEarlyStop(k=3, R_smooth=0.01)
    ours = EntropyStopRule(k=3, R_down=0.01)
    for value in series:
        x = ref.step(value)
        y = ours.step(value)
        assert x == y
        assert ref.best_epoch == ours.best_iter
        if x:
            break


def test_fixed50():
    stop = FixedStepStopRule(50)
    for i in range(49):
        assert not stop.step(0.0)
        assert stop.best_iter == i
    assert stop.step(0.0)
    assert stop.best_iter == 49
    assert stop.stop_trigger_iter == 49


def test_false_normal_ratio():
    scores = np.array([1., 10., 2., 7.])
    labels = np.array([0, 0, 0, 1])
    assert math.isclose(false_normal_ratio(scores, labels), 1.0 / 3.0)
