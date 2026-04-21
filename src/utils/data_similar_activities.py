"""
Loads activity text embeddings and BM25 summaries to compute KNN similarity
features between IATI activities for use in the overall rating model.
"""

import json
import pprint
import math
import sys
from typing import Set
from pathlib import Path
import numpy as np
import pandas as pd

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

_EMBEDDINGS_CACHE = None  # activity_id -> np.ndarray


def load_activity_embeddings(path: Path = EMBEDDINGS_PATH) -> dict[str, np.ndarray]:
    """
    Load embeddings from JSONL into a dict: activity_id -> L2-normalised np.ndarray.
    Assumes each line is like:
      {"activity_id": "...", "embedding": [float, ...], ...}
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE

    embs: dict[str, np.ndarray] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON in {path} at line {line_no}: {e}") from e

            aid = obj.get("activity_id")
            vec = obj.get("embedding")  # key name matches your file

            if aid is None or vec is None:
                continue

            v = np.asarray(vec, dtype="float32")
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm

            embs[str(aid)] = v

    _EMBEDDINGS_CACHE = embs
    return embs


def prepare_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalise and map activity_scope to numeric code
    df["activity_scope_norm"] = df["activity_scope"].astype(str).str.strip().str.lower()
    df["scope_code"] = df["activity_scope_norm"].map(ACTIVITY_SCOPES)

    # Parse dates
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

    # Precompute generic start/end date
    df["start_date"] = df.apply(pick_start_date, axis=1)
    df["end_date"] = df.apply(pick_end_date, axis=1)

    return df


def passes_quartile_constraint(q_start, q_end, c_start, c_end) -> bool:
    """Return False if candidate 3/4 mark is after query 1/4 mark."""
    if pd.isna(q_start) or pd.isna(q_end) or pd.isna(c_start) or pd.isna(c_end):
        return False  # if missing dates, don't include
    q_25 = q_start + (q_end - q_start) * 0.25
    c_75 = c_start + (c_end - c_start) * 0.75
    return c_75 <= q_25


def find_similar_activities_semantic(
    activity_id: str,
    csv_path: str = CSV_PATH,
    top_n: int = 20,
    allowed_ids=None,
):
    """
    Semantic similarity search using text embeddings (title+description).
    Returns (result_df, query_row) where result_df has metadata + 'similarity' column, sorted desc.
    """
    df = prepare_dataframe(csv_path)

    # query item from csv
    query_row = df[df["activity_id"] == activity_id].iloc[0]

    if "activity_id" not in df.columns:
        raise ValueError("CSV must contain an 'activity_id' column")

    # Map activity_id -> row index for quick lookups
    aid_to_idx = {}
    for idx, row in df.iterrows():
        aid = row.get("activity_id")
        if isinstance(aid, str) and aid:
            aid_to_idx[aid] = idx

    if activity_id not in aid_to_idx:
        raise ValueError(f"Activity ID '{activity_id}' not found in {csv_path}")

    # Load embeddings
    embs = load_activity_embeddings()

    if activity_id not in embs:
        raise ValueError(
            f"Activity ID '{activity_id}' not found in embeddings file {EMBEDDINGS_PATH}"
        )

    q_vec = embs[activity_id]
    q_start = query_row.get("start_date")
    q_end = query_row.get("end_date")

    if allowed_ids is not None:
        allowed_ids = set(str(a) for a in allowed_ids)

    candidates = df

    if pd.notna(q_start):
        candidates_notna = candidates[(pd.notna(candidates["end_date"]))]
        cand_ids = set(candidates_notna["activity_id"])
        candidates = candidates_notna
    else:
        print("ERROR: the query start q_start is invalid!")

    candidate_ids = []
    for idx, row in candidates.iterrows():
        aid = row.get("activity_id")
        if aid not in embs.keys():
            continue

        if allowed_ids is not None:
            if aid not in allowed_ids:
                continue

        # enforce 1/2 (candidate) <= 1/2 (query)
        if not passes_quartile_constraint(
            q_start,
            q_end,
            row.get("start_date"),
            row.get("end_date"),
        ):
            continue

        candidate_ids.append(aid)

    if not candidate_ids:
        # return an empty dataframe if no candidate ids
        return (
            pd.DataFrame(columns=list(df.columns) + ["similarity"]),
            df[df["activity_id"] == activity_id].copy(),
        )

    # Precompute candidate matrix for speed
    cand_vecs = np.stack([embs[aid] for aid in candidate_ids], axis=0)  # (N, d)

    # cosine similarity = dot product because we normalised
    sims = cand_vecs @ q_vec  # (N,)

    # sort by similarity desc
    order = np.argsort(-sims)
    order = order[:top_n]

    top_ids = [candidate_ids[i] for i in order]
    top_sims = sims[order]

    # build result DataFrame
    idxs = [aid_to_idx[aid] for aid in top_ids]
    result = df.loc[idxs].copy()
    result["similarity"] = top_sims

    # reorder columns
    cols_front = [
        "activity_id",
        "similarity",
        "activity_title",
        "activity_scope",
        "country_location",
        "gdp_percap",
        "dac5",
        "reporting_orgs",
        "participating_orgs",
        "start_date",
        "end_date",
    ]
    cols_front = [c for c in cols_front if c in result.columns]
    other_cols = [c for c in result.columns if c not in cols_front]

    result = result[cols_front + other_cols]

    return result, query_row


def main(activity_id: str):
    print("\nSEMANTIC VECTOR SIMILARITY:\n\n")
    df_sim, search_item = find_similar_activities_semantic(activity_id)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit("Usage: python similar_activities.py <activity_id>")
