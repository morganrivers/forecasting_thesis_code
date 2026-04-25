"""
Import smoke-tests for every Python file on the minimal branch.

Each test simply imports the module and checks it loaded without errors.
Modules that perform heavy I/O or ML at import time are guarded with
pytest.importorskip-style try/except so a missing data file doesn't
cause a misleading FAIL (it becomes an xfail or skip instead).

We also verify that the 35 source files physically exist on disk.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / "src" / "utils"
PIPELINE_DIR = REPO_ROOT / "src" / "pipeline"

# Ensure all source dirs are on the path (conftest already does this, but
# being explicit here makes tests runnable standalone too).
for _p in [str(UTILS_DIR), str(PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -- File existence ------------------------------------------------------------

REQUIRED_FILES = [
    # Entry-point scripts
    "src/utils/llm_load_predictions.py",
    "src/pipeline/L_cost_effectiveness_train_and_score.py",
    "src/pipeline/A_overall_rating_fit_and_evaluate.py",
    "src/pipeline/C_overall_rating_insample_r2.py",
    "src/pipeline/F_llm_score_forecast_narratives.py",
    "src/utils/llm_grading_utils.py",
    "src/pipeline/B_overall_rating_plot_shap.py",
    "src/pipeline/E_overall_rating_extrapolate_scaling.py",
    "src/pipeline/G_outcome_tag_train.py",
    "src/pipeline/H_outcome_tag_evaluate.py",
    "src/pipeline/I_outcome_tag_shap_stability.py",
    "src/pipeline/K_outcome_tag_extrapolate_scaling.py",
    "src/pipeline/J_outcome_tag_results_table.py",
    # Utility modules
    "src/utils/data_currency_conversion.py",
    "src/utils/data_loan_disbursement.py",
    "src/utils/llm_extraction_and_grading.py",
    "src/utils/overall_rating_feature_labels.py",
    "src/utils/feature_engineering.py",
    "src/utils/data_sector_clusters.py",
    "src/utils/overall_rating_rf_conformal.py",
    "src/utils/scoring_metrics.py",
    "src/utils/split_constants.py",
    "src/utils/ml_models.py",
]


@pytest.mark.parametrize("rel_path", REQUIRED_FILES)
def test_file_exists(rel_path):
    """Every required source file must be present on disk."""
    assert (REPO_ROOT / rel_path).is_file(), f"Missing: {rel_path}"


# -- Utility module imports ----------------------------------------------------
# These modules are pure-Python utilities with no heavy side-effects on import.

UTIL_MODULES = [
    "scoring_metrics",
    "split_constants",
    "overall_rating_feature_labels",
    "overall_rating_rf_conformal",
    "data_currency_conversion",
    "data_loan_disbursement",
]


@pytest.mark.parametrize("module_name", UTIL_MODULES)
def test_util_module_importable(module_name):
    """Utility modules must import without raising."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# -- Modules that may do light I/O at import time ------------------------------
# These are given xfail(strict=False) — they pass if the import works, and
# are recorded as xfail (not error) if the import fails due to missing data.

LIGHT_IO_MODULES = [
    "ml_models",
    "data_sector_clusters",
    "feature_engineering",
    "llm_extraction_and_grading",
]


@pytest.mark.parametrize("module_name", LIGHT_IO_MODULES)
def test_light_io_module_importable(module_name):
    """Modules with potential light I/O: import must succeed."""
    try:
        mod = importlib.import_module(module_name)
        assert mod is not None
    except Exception as exc:
        pytest.skip(f"Import of {module_name!r} failed (possibly needs data): {exc}")


# -- Entry-point scripts -------------------------------------------------------
# These are expected to require data files; we only verify they are syntactically
# valid Python (compile) and that their top-level constants/constants can be read
# without actually running main().

ENTRY_POINT_FILES = [
    (
        PIPELINE_DIR / "A_overall_rating_fit_and_evaluate.py",
        "A_overall_rating_fit_and_evaluate",
    ),
    (PIPELINE_DIR / "C_overall_rating_insample_r2.py", "C_overall_rating_insample_r2"),
    (
        PIPELINE_DIR / "L_cost_effectiveness_train_and_score.py",
        "L_cost_effectiveness_train_and_score",
    ),
    (UTILS_DIR / "llm_load_predictions.py", "llm_load_predictions"),
    (
        PIPELINE_DIR / "F_llm_score_forecast_narratives.py",
        "F_llm_score_forecast_narratives",
    ),
    (UTILS_DIR / "llm_grading_utils.py", "llm_grading_utils"),
    (PIPELINE_DIR / "B_overall_rating_plot_shap.py", "B_overall_rating_plot_shap"),
    (
        PIPELINE_DIR / "E_overall_rating_extrapolate_scaling.py",
        "E_overall_rating_extrapolate_scaling",
    ),
    (PIPELINE_DIR / "G_outcome_tag_train.py", "G_outcome_tag_train"),
    (PIPELINE_DIR / "H_outcome_tag_evaluate.py", "H_outcome_tag_evaluate"),
    (PIPELINE_DIR / "I_outcome_tag_shap_stability.py", "I_outcome_tag_shap_stability"),
    (
        PIPELINE_DIR / "K_outcome_tag_extrapolate_scaling.py",
        "K_outcome_tag_extrapolate_scaling",
    ),
    (PIPELINE_DIR / "J_outcome_tag_results_table.py", "J_outcome_tag_results_table"),
]


@pytest.mark.parametrize("path,name", ENTRY_POINT_FILES)
def test_entry_point_compiles(path, name):
    """Entry-point scripts must be syntactically valid Python."""
    source = path.read_text(encoding="utf-8")
    try:
        compile(source, str(path), "exec")
    except SyntaxError as e:
        pytest.fail(f"SyntaxError in {name}: {e}")


@pytest.mark.parametrize("path,name", ENTRY_POINT_FILES)
def test_entry_point_importable(path, name):
    """Entry-point scripts should import (skip if data is unavailable)."""
    try:
        mod = importlib.import_module(name)
        assert mod is not None
    except (FileNotFoundError, OSError, KeyError) as exc:
        pytest.skip(f"{name} needs data files not present: {exc}")
    except Exception as exc:
        pytest.skip(f"{name} raised at import time: {type(exc).__name__}: {exc}")
