import numpy as np
import pandas as pd

from churn_mlops.monitoring.metrics import (
    chi2_pvalue,
    evaluate_drift,
    psi_categorical,
    psi_numeric,
)

DRIFT_CFG = {
    "psi_moderate": 0.10,
    "psi_critical": 0.25,
    "drifted_share_threshold": 0.50,
    "psi_bins": 10,
}


def test_psi_numeric_identical_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert psi_numeric(x, x) < 0.01


def test_psi_numeric_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(1.0, 1, 5000)
    assert psi_numeric(ref, cur) > 0.25


def test_psi_categorical_detects_rebalance():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.choice(["a", "b", "c"], 4000, p=[0.6, 0.3, 0.1]))
    same = pd.Series(rng.choice(["a", "b", "c"], 4000, p=[0.6, 0.3, 0.1]))
    shifted = pd.Series(rng.choice(["a", "b", "c"], 4000, p=[0.1, 0.3, 0.6]))
    assert psi_categorical(ref, same) < 0.05
    assert psi_categorical(ref, shifted) > 0.25


def test_psi_categorical_handles_unseen_category():
    ref = pd.Series(["a"] * 500 + ["b"] * 500)
    cur = pd.Series(["a"] * 300 + ["b"] * 300 + ["c"] * 400)
    assert np.isfinite(psi_categorical(ref, cur))
    assert psi_categorical(ref, cur) > 0.25


def test_chi2_pvalue_stable_vs_shifted():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.choice(["x", "y"], 3000, p=[0.7, 0.3]))
    same = pd.Series(rng.choice(["x", "y"], 500, p=[0.7, 0.3]))
    shifted = pd.Series(rng.choice(["x", "y"], 500, p=[0.3, 0.7]))
    assert chi2_pvalue(ref, same) > 0.01
    assert chi2_pvalue(ref, shifted) < 0.01


def _tiny_schema():
    return {
        "numeric_features": ["num"],
        "categorical_features": ["cat"],
        "categories": {"cat": ["a", "b"]},
    }


def _frame(num_loc: float, p_a: float, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"num": rng.normal(num_loc, 1, n), "cat": rng.choice(["a", "b"], n, p=[p_a, 1 - p_a])}
    )


def test_evaluate_drift_ok_when_same_distribution():
    report = evaluate_drift(_frame(0, 0.5, seed=1), _frame(0, 0.5, seed=2), _tiny_schema(), DRIFT_CFG)
    assert report.severity == "OK"
    assert not report.dataset_drift


def test_evaluate_drift_critical_on_strong_shift():
    report = evaluate_drift(_frame(0, 0.5, seed=1), _frame(1.5, 0.9, seed=2), _tiny_schema(), DRIFT_CFG)
    assert report.severity == "CRITICAL"
    assert report.dataset_drift
    assert report.max_psi >= 0.25
