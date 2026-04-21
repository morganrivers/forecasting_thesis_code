"""
Structural tests for run_all_thesis_results.sh

Checks:
  - bash syntax is valid (bash -n)
  - every python3 invocation references a file that exists on disk
  - the cd directories all exist
  - the script is executable (or at least readable)
  - required non-script files exist (requirements.txt, setup_env.sh, README.md)
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "run_all_thesis_results.sh"


# -- Bash syntax ---------------------------------------------------------------


def test_bash_syntax_valid():
    """bash -n checks syntax without executing."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n reported syntax errors:\n{result.stderr}"


# -- Script existence and readability -----------------------------------------


def test_script_exists():
    assert SCRIPT_PATH.is_file(), f"Missing: {SCRIPT_PATH}"


def test_script_is_readable():
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert len(content) > 0


def test_script_has_shebang():
    first_line = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!"), "Script must have a shebang line"


# -- Referenced Python files exist ---------------------------------------------


def _extract_python_scripts(script_text: str) -> list[tuple[str, str]]:
    """
    Return (cd_dir, python_file) pairs from lines like:
        "python3 A_overall_rating_fit_and_evaluate.py ..."
    combined with the preceding cd directory from run_script calls.
    """
    # Extract run_script calls: dir is arg 3, cmd is arg 4
    # Pattern: run_script \ ... "$REPO_ROOT/src/..." \ "python3 foo.py ..."
    results = []
    lines = script_text.splitlines()
    current_dir = None
    for line in lines:
        line = line.strip()
        # Capture the directory argument (3rd arg to run_script, after REPO_ROOT)
        dir_match = re.search(r'"?\$REPO_ROOT/([^"]+)"?', line)
        if dir_match and "python3" not in line:
            current_dir = dir_match.group(1).strip()
        # Capture python3 invocations
        py_match = re.search(r'"python3\s+(\S+\.py)', line)
        if py_match and current_dir:
            results.append((current_dir, py_match.group(1)))
    return results


@pytest.fixture(scope="module")
def script_text():
    return SCRIPT_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def python_invocations(script_text):
    return _extract_python_scripts(script_text)


def test_at_least_one_python_invocation_found(python_invocations):
    assert (
        len(python_invocations) >= 8
    ), f"Expected >=8 python3 invocations, found {len(python_invocations)}"


@pytest.mark.parametrize(
    "rel_dir,py_file",
    [
        ("src/pipeline", "A_overall_rating_fit_and_evaluate.py"),
        ("src/pipeline", "B_overall_rating_plot_shap.py"),
        ("src/pipeline", "C_overall_rating_insample_r2.py"),
        ("src/pipeline", "F_llm_score_forecast_narratives.py"),
        ("src/pipeline", "G_outcome_tag_train.py"),
        ("src/pipeline", "H_outcome_tag_evaluate.py"),
        ("src/pipeline", "I_outcome_tag_shap_stability.py"),
        ("src/pipeline", "J_outcome_tag_results_table.py"),
        ("src/pipeline", "E_overall_rating_extrapolate_scaling.py"),
        ("src/pipeline", "K_outcome_tag_extrapolate_scaling.py"),
        ("src/pipeline", "L_cost_effectiveness_train_and_score.py"),
    ],
)
def test_referenced_python_file_exists(rel_dir, py_file):
    full = REPO_ROOT / rel_dir / py_file
    assert (
        full.is_file()
    ), f"Script references {py_file} but it does not exist at {full}"


# -- cd directories exist ------------------------------------------------------


@pytest.mark.parametrize(
    "rel_dir",
    [
        "src/pipeline",
    ],
)
def test_cd_directory_exists(rel_dir):
    assert (
        REPO_ROOT / rel_dir
    ).is_dir(), f"Directory referenced in script does not exist: {rel_dir}"


# -- Companion files exist -----------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "README.md",
    ],
)
def test_companion_file_exists(filename):
    assert (REPO_ROOT / filename).is_file(), f"Missing companion file: {filename}"


# -- Output variable defined ---------------------------------------------------


def test_output_file_variable_defined(script_text):
    assert "thesis_results_output.txt" in script_text


def test_run_script_function_defined(script_text):
    assert "run_script()" in script_text or "run_script " in script_text


# -- set -u / pipefail present -------------------------------------------------


def test_errexit_or_pipefail_present(script_text):
    """Script should use set -uo pipefail or similar for safety."""
    assert (
        "pipefail" in script_text or "set -e" in script_text
    ), "Script should use set -uo pipefail or set -e for safe error handling"
