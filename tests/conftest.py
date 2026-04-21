"""
Shared pytest configuration for the forecasting_IATI test suite.

Sets up sys.path so that all utility modules can be imported without
installing the package, and provides common fixtures.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# -- Path setup --------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / "src" / "utils"
PIPELINE_DIR = REPO_ROOT / "src" / "pipeline"

for _p in [str(UTILS_DIR), str(PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# -- Common fixtures ----------------------------------------------------------


@pytest.fixture
def small_df():
    """A small DataFrame with the columns most scripts need."""
    rng = np.random.default_rng(42)
    n = 40
    dates = pd.date_range("2005-01-01", periods=n, freq="180D")
    return pd.DataFrame(
        {
            "activity_id": [f"act_{i:03d}" for i in range(n)],
            "start_date": dates,
            "rating": rng.uniform(0, 5, n),
            "reporting_orgs": rng.choice(["FCDO", "WorldBank", "BMZ"], n),
        }
    )


@pytest.fixture
def perfect_predictions():
    """y_true == y_pred for edge-case testing."""
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    return y, y.copy()


@pytest.fixture
def known_predictions():
    """Simple known-value arrays for deterministic metric checks."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])  # constant +0.5 bias
    return y_true, y_pred


@pytest.fixture
def grouped_data():
    """y_true / y_pred / groups for within-group metric tests."""
    y_true = np.array([1, 2, 3, 4, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 2.2, 2.8, 4.1, 4.9], dtype=float)
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    return y_true, y_pred, groups


@pytest.fixture
def feature_df():
    """DataFrame with numeric features, rating target, and start_date for model tests."""
    rng = np.random.default_rng(7)
    n = 60
    dates = pd.date_range("2005-01-01", periods=n, freq="120D")
    idx = pd.RangeIndex(n)
    df = pd.DataFrame(
        {
            "start_date": dates,
            "rating": rng.uniform(0, 5, n),
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(2, 0.5, n),
            "feat_c": rng.normal(-1, 2, n),
        },
        index=idx,
    )
    # Introduce NaNs to exercise median imputation
    nan_a = rng.choice(n, 10, replace=False)
    nan_b = rng.choice(n, 5, replace=False)
    df.loc[nan_a, "feat_a"] = np.nan
    df.loc[nan_b, "feat_b"] = np.nan
    return df
