"""
Utility functions for loading activity embeddings, preparing the activity
dataframe, and computing recency-weighted KNN similarity features.
"""

import statsmodels.api as sm
import json
import math
import sys
from pathlib import Path
from typing import Set, Iterable

import numpy as np
import pandas as pd

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from sklearn.metrics import r2_score
from scoring_metrics import rmse, side_accuracy, r2 as r2_metric

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERBOSE = False

ACTIVITY_SCOPES = {
    "global": 7,
    "regional": 6,
    "multi-national": 5,
    "national": 4,
    "sub-national: multi-first-level administrative areas": 3,
    "sub-national: single first-level administrative area": 2,
    "sub-national: single second-level administrative area": 1,
    "single location": 0,
}

CSV_PATH = "../../data/info_for_activity_forecasting_old_transaction_types.csv"
EMBEDDINGS_PATH = Path("../../data/activity_text_embeddings_gemini.jsonl")

# results will be written here (one CSV per recency weight)
OUT_DIR = Path("../../data/temp_knn_results")

_EMBEDDINGS_CACHE = None  # activity_id -> np.ndarray
_DF_CACHE = None  # prepared metadata dataframe


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
def load_activity_embeddings(path: Path = EMBEDDINGS_PATH) -> dict[str, np.ndarray]:
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    embs: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            aid = obj.get("activity_id")
            vec = obj.get("embedding")
            if aid is None or vec is None:
                continue
            v = np.asarray(vec, dtype="float32")
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            embs[str(aid)] = v

    _EMBEDDINGS_CACHE = embs
    return embs


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    global _DF_CACHE
    if _DF_CACHE is not None:
        return _DF_CACHE

    df = pd.read_csv(csv_path)

    df["activity_scope_norm"] = df["activity_scope"].astype(str).str.strip().str.lower()
    df["scope_code"] = df["activity_scope_norm"].map(ACTIVITY_SCOPES)

    date_cols = [
        "original_planned_start_date",
        "original_planned_close_date",
        "actual_start_date",
        "actual_end_date",
        "actual_close_date",
        "txn_first_date",
        "txn_last_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["start_date"] = df.apply(pick_start_date, axis=1)
    df["end_date"] = df.apply(pick_end_date, axis=1)

    _DF_CACHE = df
    return df


def passes_quartile_constraint(q_start, q_end, c_start, c_end) -> bool:
    if pd.isna(q_start) or pd.isna(q_end) or pd.isna(c_start) or pd.isna(c_end):
        return False
    q_25 = q_start + (q_end - q_start) * 0.25
    c_75 = c_start + (c_end - c_start) * 0.75
    return c_75 <= q_25
