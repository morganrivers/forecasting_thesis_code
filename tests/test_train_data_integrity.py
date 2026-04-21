"""
Fragile, data-facing invariant tests.  Any file corruption, schema drift,
or code path that silently mishandles the actual training data will cause
these tests to raise — not warn, not log — and exit.

Design principles:
  - Every assertion uses bare `assert` or `pytest.raises`.
  - No `try/except` blocks that swallow errors.
  - No "if bad: print warning; continue" patterns.
  - Duplicate keys, out-of-range values, wrong column names → AssertionError.
  - These tests describe what IS TRUE about the current data, not what
    might be true in a hypothetical edge case.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# -- Path setup ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
UTILS_DIR = REPO_ROOT / "src" / "utils"
PIPELINE_DIR = REPO_ROOT / "src" / "pipeline"

for _p in [str(UTILS_DIR), str(PIPELINE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -- Canonical file paths -------------------------------------------------------
INFO_CSV = DATA_DIR / "info_for_activity_forecasting_old_transaction_types.csv"
RATINGS_JSONL = DATA_DIR / "merged_overall_ratings.jsonl"
APPLIED_TAGS_JSONL = DATA_DIR / "outcome_tags" / "applied_tags.jsonl"
LLM_EXPENDITURE_JSONL = DATA_DIR / "llm_planned_expenditure.jsonl"


# -- Helpers --------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def _pick_start_date(row: pd.Series) -> pd.Timestamp:
    """Mirror of feature_engineering.pick_start_date."""
    for col in ("actual_start_date", "original_planned_start_date", "txn_first_date"):
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return pd.NaT


# -- Fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def info_df() -> pd.DataFrame:
    """Load the main activity feature CSV (module-scoped to avoid re-reading)."""
    assert INFO_CSV.exists(), f"Required data file missing: {INFO_CSV}"
    return pd.read_csv(INFO_CSV)


@pytest.fixture(scope="module")
def ratings_records() -> list[dict]:
    assert RATINGS_JSONL.exists(), f"Required data file missing: {RATINGS_JSONL}"
    return _load_jsonl(RATINGS_JSONL)


@pytest.fixture(scope="module")
def applied_tags_records() -> list[dict]:
    assert (
        APPLIED_TAGS_JSONL.exists()
    ), f"Required data file missing: {APPLIED_TAGS_JSONL}"
    return _load_jsonl(APPLIED_TAGS_JSONL)


@pytest.fixture(scope="module")
def loaded_ratings() -> pd.Series:
    """Call the real load_ratings() on the real file."""
    from feature_engineering import load_ratings

    return load_ratings(str(RATINGS_JSONL))


@pytest.fixture(scope="module")
def tags_df_and_cols():
    """Return (tags_df, model_cols) from load_applied_tags on the real file."""
    from G_outcome_tag_train import load_applied_tags

    return load_applied_tags(APPLIED_TAGS_JSONL)


# -- info_for_activity_forecasting CSV -----------------------------------------


class TestInfoCSV:
    """
    Invariants about info_for_activity_forecasting_old_transaction_types.csv.
    This is the primary feature source; any schema or data corruption here
    propagates silently through the entire pipeline.
    """

    REQUIRED_COLUMNS = [
        "activity_id",
        "reporting_orgs",
        "implementing_org_type",
        "activity_scope",
        "gdp_percap",
        "cpia_score",
        "wgi_control_of_corruption_est",
        "wgi_government_effectiveness_est",
        "wgi_political_stability_est",
        "wgi_regulatory_quality_est",
        "wgi_rule_of_law_est",
        "status_code",
        "original_planned_start_date",
        "actual_start_date",
        "txn_first_date",
        "region_AFE",
        "region_AFW",
        "region_EAP",
        "region_ECA",
        "region_LAC",
        "region_MENA",
        "region_SAS",
    ]

    # IATI activity status codes we have ever seen in the data:
    # 2=Implementation, 3=Finalisation, 4=Closed, 5=Cancelled, 6=Suspended
    VALID_STATUS_CODES = {2, 3, 4, 5, 6}

    # implementing_org_type values present in the current data:
    VALID_ORG_TYPES = {"govermental", "ngo", "other", "academic"}

    def test_file_exists(self):
        assert INFO_CSV.exists(), f"Missing data file: {INFO_CSV}"

    def test_required_columns_present(self, info_df):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in info_df.columns]
        assert not missing, f"Required columns absent from {INFO_CSV.name}: {missing}"

    def test_no_duplicate_activity_ids(self, info_df):
        dup_ids = info_df[info_df["activity_id"].duplicated(keep=False)][
            "activity_id"
        ].tolist()
        assert not dup_ids, (
            f"{INFO_CSV.name}: found {len(dup_ids)} duplicate activity_id values — "
            f"activity_id must be unique primary key.  Duplicates: {dup_ids[:10]}"
        )

    def test_activity_id_no_nulls(self, info_df):
        n_null = int(info_df["activity_id"].isna().sum())
        assert n_null == 0, (
            f"{INFO_CSV.name}: {n_null} null activity_id values — "
            "every row must have a unique non-null identifier"
        )

    def test_reporting_orgs_no_nulls(self, info_df):
        n_null = int(info_df["reporting_orgs"].isna().sum())
        assert n_null == 0, f"{INFO_CSV.name}: {n_null} null reporting_orgs values"

    def test_status_code_only_known_values(self, info_df):
        """status_code must contain only codes we understand; unknown codes silently corrupt the pipeline."""
        observed = set(info_df["status_code"].dropna().astype(int).unique())
        unknown = observed - self.VALID_STATUS_CODES
        assert not unknown, (
            f"{INFO_CSV.name}: unknown status_code values {unknown} — "
            f"expected subset of {self.VALID_STATUS_CODES}"
        )

    def test_implementing_org_type_only_known_values(self, info_df):
        """Unknown org types map to 'other' in the pipeline; new values silently dropped."""
        observed = set(
            info_df["implementing_org_type"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )
        unknown = observed - self.VALID_ORG_TYPES
        assert not unknown, (
            f"{INFO_CSV.name}: unknown implementing_org_type values {unknown} — "
            f"expected subset of {self.VALID_ORG_TYPES}"
        )

    def test_wgi_values_within_range(self, info_df):
        """WGI estimates are bounded: values outside [-4, 4] indicate a bad merge."""
        wgi_cols = [
            "wgi_control_of_corruption_est",
            "wgi_government_effectiveness_est",
            "wgi_political_stability_est",
            "wgi_regulatory_quality_est",
            "wgi_rule_of_law_est",
        ]
        for col in wgi_cols:
            vals = info_df[col].dropna()
            lo, hi = float(vals.min()), float(vals.max())
            assert lo >= -4.0 and hi <= 4.0, (
                f"{INFO_CSV.name}: {col} has values outside [-4, 4]: "
                f"min={lo:.3f}, max={hi:.3f}"
            )

    def test_region_columns_are_binary(self, info_df):
        """Region indicator columns must be 0/1 only — they are used as one-hot features."""
        region_cols = [c for c in info_df.columns if c.startswith("region_")]
        assert region_cols, f"No region_ columns found in {INFO_CSV.name}"
        for col in region_cols:
            vals = set(info_df[col].dropna().astype(int).unique())
            assert vals <= {
                0,
                1,
            }, f"{INFO_CSV.name}: {col} contains non-binary values {vals}"

    def test_cpia_score_within_range(self, info_df):
        """CPIA scores are 1–6; values outside this range indicate a bad merge or unit change."""
        vals = info_df["cpia_score"].dropna()
        lo, hi = float(vals.min()), float(vals.max())
        assert (
            lo >= 1.0 and hi <= 6.0
        ), f"{INFO_CSV.name}: cpia_score outside [1, 6]: min={lo:.3f}, max={hi:.3f}"

    def test_gdp_percap_positive(self, info_df):
        """GDP per capita cannot be non-positive."""
        vals = info_df["gdp_percap"].dropna()
        assert (
            float(vals.min()) > 0
        ), f"{INFO_CSV.name}: gdp_percap has non-positive values (min={float(vals.min()):.3f})"

    def test_noisy_feature_wgi_and_cpia_columns_present(self, info_df):
        """
        NOISY_FEATURE_GROUPS includes WGI/CPIA features loaded directly from the CSV.
        The 11 missingness indicators are computed at runtime so are NOT expected in the CSV.
        """
        from G_outcome_tag_train import NOISY_FEATURE_GROUPS

        # Only the raw WGI and CPIA features come directly from the CSV
        csv_features = [
            f
            for f in NOISY_FEATURE_GROUPS
            if (f.startswith("wgi_") and not f.endswith("_missing"))
            or f == "cpia_score"
        ]
        missing = [f for f in csv_features if f not in info_df.columns]
        assert (
            not missing
        ), f"NOISY_FEATURE_GROUPS references columns not in {INFO_CSV.name}: {missing}"


# -- merged_overall_ratings.jsonl ----------------------------------------------


class TestRatingsJSONL:
    """
    Invariants about merged_overall_ratings.jsonl and the load_ratings() pipeline.
    """

    def test_file_exists(self):
        assert RATINGS_JSONL.exists(), f"Missing: {RATINGS_JSONL}"

    def test_all_records_have_activity_id(self, ratings_records):
        """Every record must have a non-empty activity_id."""
        missing = [
            i
            for i, r in enumerate(ratings_records)
            if not str(r.get("activity_id") or "").strip()
        ]
        assert not missing, (
            f"{RATINGS_JSONL.name}: {len(missing)} records have no activity_id "
            f"(line indices: {missing[:10]})"
        )

    def test_loaded_ratings_all_in_range(self, loaded_ratings):
        """load_ratings() must never produce a value outside the 0–5 canonical scale."""
        lo = float(loaded_ratings.min())
        hi = float(loaded_ratings.max())
        assert lo >= 0.0, (
            f"load_ratings() returned rating < 0: min={lo:.4f} — "
            "check get_success_measure_from_rating_value clipping"
        )
        assert hi <= 5.0, (
            f"load_ratings() returned rating > 5: max={hi:.4f} — "
            "check get_success_measure_from_rating_value clipping"
        )

    def test_loaded_ratings_no_nan(self, loaded_ratings):
        """The function must never yield NaN — it filters or converts."""
        n_nan = int(loaded_ratings.isna().sum())
        assert n_nan == 0, (
            f"load_ratings() produced {n_nan} NaN values — "
            "the function should skip bad records, not include NaN ones"
        )

    def test_loaded_ratings_minimum_count(self, loaded_ratings):
        """Sanity floor: if far fewer ratings loaded than expected, something is broken."""
        assert len(loaded_ratings) >= 1000, (
            f"load_ratings() returned only {len(loaded_ratings)} ratings; "
            "expected >= 1000 — check RATINGS_JSONL path and parsing"
        )

    def test_loaded_ratings_index_is_string_activity_ids(self, loaded_ratings):
        """Index must contain string activity_ids (used for DataFrame.join)."""
        assert loaded_ratings.index.dtype == object, (
            f"load_ratings() index dtype = {loaded_ratings.index.dtype}; "
            "expected object (str) so .join() on activity_id works"
        )

    def test_loaded_ratings_no_duplicate_index(self, loaded_ratings):
        """Duplicate index entries in the ratings Series will corrupt downstream joins."""
        dup_ids = loaded_ratings.index[loaded_ratings.index.duplicated()].tolist()
        assert not dup_ids, (
            f"load_ratings() produced duplicate index entries — "
            f"last-write-wins silently: {dup_ids[:10]}"
        )


# -- applied_tags.jsonl --------------------------------------------------------


class TestAppliedTagsJSONL:
    """
    Invariants about data/outcome_tags/applied_tags.jsonl.
    Duplicate activity_ids or inconsistent tag keys here will produce silently
    wrong binary labels and corrupt all 14 tag models.
    """

    def test_file_exists(self):
        assert APPLIED_TAGS_JSONL.exists(), f"Missing: {APPLIED_TAGS_JSONL}"

    def test_no_duplicate_activity_ids(self, applied_tags_records):
        """Duplicate activity_ids → the second record silently overwrites the first."""
        ids = [r["activity_id"] for r in applied_tags_records]
        counts = Counter(ids)
        dups = {k: v for k, v in counts.items() if v > 1}
        assert not dups, (
            f"{APPLIED_TAGS_JSONL.name}: {len(dups)} duplicate activity_ids — "
            f"duplicates: {list(dups.items())[:10]}"
        )

    def test_all_records_have_activity_id(self, applied_tags_records):
        missing = [
            i
            for i, r in enumerate(applied_tags_records)
            if not str(r.get("activity_id") or "").strip()
        ]
        assert (
            not missing
        ), f"{APPLIED_TAGS_JSONL.name}: records at indices {missing[:10]} have no activity_id"

    def test_all_records_have_unsigned_or_signed_tags(self, applied_tags_records):
        """load_applied_tags filters out records without tag dicts — they would silently vanish."""
        missing = [
            r["activity_id"]
            for r in applied_tags_records
            if "unsigned_tags" not in r and "signed_tags" not in r
        ]
        assert not missing, (
            f"{APPLIED_TAGS_JSONL.name}: {len(missing)} records have neither "
            f"unsigned_tags nor signed_tags: {missing[:10]}"
        )

    def test_unsigned_tag_keys_consistent_across_all_records(
        self, applied_tags_records
    ):
        """
        load_applied_tags discovers column names from records[0].
        If any later record has different unsigned_tags keys, those columns silently
        get value 0 (key missing → .get() returns None → row[col] = 0).
        """
        first_keys = frozenset(applied_tags_records[0].get("unsigned_tags", {}).keys())
        inconsistent = [
            (i, r["activity_id"], frozenset(r.get("unsigned_tags", {}).keys()))
            for i, r in enumerate(applied_tags_records)
            if frozenset(r.get("unsigned_tags", {}).keys()) != first_keys
        ]
        assert not inconsistent, (
            f"{APPLIED_TAGS_JSONL.name}: {len(inconsistent)} records have different "
            f"unsigned_tag keys from record 0.  "
            f"First mismatch at index {inconsistent[0][0]}, id={inconsistent[0][1]}"
        )

    def test_signed_tag_keys_consistent_across_all_records(self, applied_tags_records):
        """Same as above for signed_tags."""
        first_keys = frozenset(applied_tags_records[0].get("signed_tags", {}).keys())
        inconsistent = [
            (i, r["activity_id"], frozenset(r.get("signed_tags", {}).keys()))
            for i, r in enumerate(applied_tags_records)
            if frozenset(r.get("signed_tags", {}).keys()) != first_keys
        ]
        assert not inconsistent, (
            f"{APPLIED_TAGS_JSONL.name}: {len(inconsistent)} records have different "
            f"signed_tag keys from record 0.  "
            f"First mismatch at index {inconsistent[0][0]}, id={inconsistent[0][1]}"
        )

    def test_unsigned_tag_values_are_bool_or_null(self, applied_tags_records):
        """Unsigned tag values must be True/False/None — anything else corrupts binary labels."""
        for rec in applied_tags_records:
            aid = rec["activity_id"]
            for tag, val in rec.get("unsigned_tags", {}).items():
                assert val is True or val is False or val is None, (
                    f"{APPLIED_TAGS_JSONL.name}: activity={aid}, tag={tag} has "
                    f"non-boolean value {val!r}"
                )

    def test_signed_tag_values_are_bool_or_null(self, applied_tags_records):
        """Signed tag values must be True/False/None."""
        for rec in applied_tags_records:
            aid = rec["activity_id"]
            for tag, val in rec.get("signed_tags", {}).items():
                assert val is True or val is False or val is None, (
                    f"{APPLIED_TAGS_JSONL.name}: activity={aid}, tag={tag} has "
                    f"non-boolean value {val!r}"
                )


# -- load_applied_tags() integration -------------------------------------------


class TestLoadAppliedTags:
    """
    Test the load_applied_tags() function on the real file.
    These catch bugs in column naming, dtype, and DataFrame integrity.
    """

    def test_returns_dataframe_and_list(self, tags_df_and_cols):
        df, cols = tags_df_and_cols
        assert isinstance(
            df, pd.DataFrame
        ), "load_applied_tags must return (DataFrame, list)"
        assert isinstance(
            cols, list
        ), "load_applied_tags second return value must be list"

    def test_no_duplicate_columns(self, tags_df_and_cols):
        df, _ = tags_df_and_cols
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        assert (
            not dup_cols
        ), f"load_applied_tags() produced duplicate columns: {dup_cols}"

    def test_no_duplicate_activity_ids_in_dataframe(self, tags_df_and_cols):
        """DataFrame index is activity_id; duplicates corrupt all label lookups."""
        df, _ = tags_df_and_cols
        dup_idx = df.index[df.index.duplicated()].tolist()
        assert (
            not dup_idx
        ), f"load_applied_tags() produced DataFrame with duplicate index values: {dup_idx[:10]}"

    def test_no_duplicate_model_cols(self, tags_df_and_cols):
        _, cols = tags_df_and_cols
        counts = Counter(cols)
        dups = {k: v for k, v in counts.items() if v > 1}
        assert (
            not dups
        ), f"load_applied_tags() produced duplicate model_cols entries: {dups}"

    def test_all_hardcoded_14_tags_in_model_cols(self, tags_df_and_cols):
        """Every HARDCODED_14_TAGS entry must be in model_cols; missing = untrained tag."""
        from G_outcome_tag_train import HARDCODED_14_TAGS

        _, cols = tags_df_and_cols
        cols_set = set(cols)
        missing = [t for t in HARDCODED_14_TAGS if t not in cols_set]
        assert not missing, (
            f"load_applied_tags() is missing HARDCODED_14_TAGS entries: {missing} — "
            "the applied_tags.jsonl schema no longer matches the hardcoded tag list"
        )

    def test_all_hardcoded_14_tags_are_dataframe_columns(self, tags_df_and_cols):
        """The tags must also be actual DataFrame columns (not just in model_cols list)."""
        from G_outcome_tag_train import HARDCODED_14_TAGS

        df, _ = tags_df_and_cols
        missing = [t for t in HARDCODED_14_TAGS if t not in df.columns]
        assert (
            not missing
        ), f"HARDCODED_14_TAGS entries absent from tags_df columns: {missing}"

    def test_tag_values_are_binary_or_nan(self, tags_df_and_cols):
        """
        After load_applied_tags, every cell in a tag column must be 0, 1, or NaN.
        Any other value is a code bug in the tag-loading logic.
        """
        from G_outcome_tag_train import HARDCODED_14_TAGS

        df, _ = tags_df_and_cols
        for tag in HARDCODED_14_TAGS:
            col = df[tag]
            valid_mask = col.isna() | col.isin([0, 1])
            bad = col[~valid_mask]
            assert bad.empty, (
                f"Tag column '{tag}' contains non-binary, non-NaN values: "
                f"{bad.unique().tolist()[:10]}"
            )

    def test_model_cols_match_dataframe_columns_exactly(self, tags_df_and_cols):
        """model_cols and df.columns must be identical sets (no phantom cols in either)."""
        df, cols = tags_df_and_cols
        cols_set = set(cols)
        df_cols_set = set(df.columns)
        in_list_not_df = cols_set - df_cols_set
        in_df_not_list = df_cols_set - cols_set
        assert (
            not in_list_not_df
        ), f"model_cols entries not in DataFrame: {in_list_not_df}"
        assert (
            not in_df_not_list
        ), f"DataFrame columns not in model_cols: {in_df_not_list}"


# -- Train-set label count sanity ----------------------------------------------


class TestTrainLabelCounts:
    """
    For each of the HARDCODED_14_TAGS, verify that the training split has
    at least MIN_TAG_TRAIN_COUNT positive examples.  Fewer positives than
    the threshold means the code path that gates on MIN_TAG_TRAIN_COUNT
    would silently drop the tag model.
    """

    @pytest.fixture(scope="class")
    def train_tags(self, tags_df_and_cols):
        """
        Return the subset of tags_df corresponding to the training split.
        Uses the same date cascade as pick_start_date + split_latest_by_date_with_cutoff.
        """
        from split_constants import split_latest_by_date_with_cutoff

        df_tags, _ = tags_df_and_cols

        # Load start dates from the activity CSV
        info_df = pd.read_csv(
            INFO_CSV,
            usecols=[
                "activity_id",
                "actual_start_date",
                "original_planned_start_date",
                "txn_first_date",
            ],
        )
        info_df = info_df.set_index("activity_id")

        # Build start_date via the same priority cascade as pick_start_date
        start = info_df["actual_start_date"].copy()
        mask = start.isna()
        start[mask] = info_df.loc[mask, "original_planned_start_date"]
        mask = start.isna()
        start[mask] = info_df.loc[mask, "txn_first_date"]
        start = pd.to_datetime(start, errors="coerce")
        start.name = "start_date"

        # Merge tags and dates on activity_id
        merged = df_tags.join(start, how="inner")

        # Reset to positional index for split_latest_by_date_with_cutoff
        merged_reset = merged.reset_index()  # brings activity_id back as column
        train_idx, _, _ = split_latest_by_date_with_cutoff(merged_reset, "start_date")
        return merged_reset.loc[train_idx]

    def test_each_hardcoded_tag_has_enough_train_positives(self, train_tags):
        from G_outcome_tag_train import HARDCODED_14_TAGS, MIN_TAG_TRAIN_COUNT

        failures = []
        for tag in HARDCODED_14_TAGS:
            n_pos = int(train_tags[tag].sum())
            if n_pos < MIN_TAG_TRAIN_COUNT:
                failures.append(f"{tag}: {n_pos} < {MIN_TAG_TRAIN_COUNT}")
        assert not failures, (
            "The following HARDCODED_14_TAGS have fewer positives in TRAIN than "
            f"MIN_TAG_TRAIN_COUNT={MIN_TAG_TRAIN_COUNT}:\n  " + "\n  ".join(failures)
        )

    def test_no_hardcoded_tag_is_all_zeros_in_train(self, train_tags):
        """A tag with zero positives in train silently trains on noise — it must not happen."""
        from G_outcome_tag_train import HARDCODED_14_TAGS

        all_zero = [t for t in HARDCODED_14_TAGS if int(train_tags[t].sum()) == 0]
        assert (
            not all_zero
        ), f"HARDCODED_14_TAGS with ZERO positives in TRAIN: {all_zero}"

    def test_no_hardcoded_tag_is_all_ones_in_train(self, train_tags):
        """A tag that is always 1 in train cannot be predicted — it must not happen."""
        from G_outcome_tag_train import HARDCODED_14_TAGS

        n_train = len(train_tags)
        all_one = [t for t in HARDCODED_14_TAGS if int(train_tags[t].sum()) == n_train]
        assert not all_one, f"HARDCODED_14_TAGS that are ALL 1 in TRAIN: {all_one}"


# -- llm_planned_expenditure.jsonl ---------------------------------------------


class TestLLMExpenditureJSONL:
    """
    load_llm_planned_expenditure() stores results in a dict keyed by activity_id.
    Duplicate activity_ids are silently overwritten by last-record-wins.
    """

    @pytest.fixture(scope="class")
    def expenditure_records(self):
        assert LLM_EXPENDITURE_JSONL.exists(), f"Missing: {LLM_EXPENDITURE_JSONL}"
        return _load_jsonl(LLM_EXPENDITURE_JSONL)

    def test_no_duplicate_activity_ids(self, expenditure_records):
        ids = [r["activity_id"] for r in expenditure_records]
        counts = Counter(ids)
        dups = {k: v for k, v in counts.items() if v > 1}
        assert not dups, (
            f"{LLM_EXPENDITURE_JSONL.name}: {len(dups)} duplicate activity_ids — "
            f"duplicates: {list(dups.items())[:10]}"
        )

    def test_all_records_have_required_keys(self, expenditure_records):
        """Each record must have activity_id and planned_expenditure_usd."""
        bad = [
            (i, r.get("activity_id"))
            for i, r in enumerate(expenditure_records)
            if "activity_id" not in r or "planned_expenditure_usd" not in r
        ]
        assert (
            not bad
        ), f"{LLM_EXPENDITURE_JSONL.name}: {len(bad)} records missing required keys: {bad[:10]}"

    def test_positive_expenditure_values_are_positive(self, expenditure_records):
        """
        load_llm_planned_expenditure stores log(usd) only for v > 0.
        Any record where planned_expenditure_usd is present but <= 0
        would be silently skipped — test that this is intentional absence, not a bug.
        """
        non_positive = [
            (r["activity_id"], r["planned_expenditure_usd"])
            for r in expenditure_records
            if r.get("planned_expenditure_usd") is not None
            and r["planned_expenditure_usd"] <= 0
        ]
        # Non-positive values are silently skipped — but that means the feature will be NaN.
        # This test alerts if there are suspiciously many such records.
        pct = len(non_positive) / max(len(expenditure_records), 1)
        assert pct < 0.10, (
            f"{LLM_EXPENDITURE_JSONL.name}: {len(non_positive)} records "
            f"({pct:.1%}) have non-positive planned_expenditure_usd — "
            "they will be silently dropped.  Investigate if this is unexpected."
        )
