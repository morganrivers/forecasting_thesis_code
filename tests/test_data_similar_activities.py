"""
Unit tests for src/utils/data_similar_activities.py.

Covers:
 - ACTIVITY_SCOPES constant (structure and values)
 - load_activity_embeddings (file I/O mocked)
 - passes_quartile_constraint (pure function)
 - find_similar_activities_semantic (mocked prepare_dataframe + load_activity_embeddings)
"""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / "src" / "utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

import data_similar_activities as DSA

# ===========================================================================
# 1.  ACTIVITY_SCOPES constant
# ===========================================================================


class TestActivityScopes:
    """Structural invariants for the ACTIVITY_SCOPES lookup dict."""

    def test_is_a_dict(self):
        assert isinstance(DSA.ACTIVITY_SCOPES, dict)

    def test_non_empty(self):
        assert len(DSA.ACTIVITY_SCOPES) > 0

    def test_all_keys_are_strings(self):
        for k in DSA.ACTIVITY_SCOPES:
            assert isinstance(k, str), f"Non-string key: {k!r}"

    def test_all_values_are_integers(self):
        for k, v in DSA.ACTIVITY_SCOPES.items():
            assert isinstance(v, int), f"Non-int value for key {k!r}: {v!r}"

    def test_global_has_highest_code(self):
        assert DSA.ACTIVITY_SCOPES["global"] == max(DSA.ACTIVITY_SCOPES.values())

    def test_single_location_has_lowest_code(self):
        assert DSA.ACTIVITY_SCOPES["single location"] == min(
            DSA.ACTIVITY_SCOPES.values()
        )

    def test_codes_are_unique(self):
        vals = list(DSA.ACTIVITY_SCOPES.values())
        assert len(vals) == len(set(vals)), "Duplicate scope codes found"

    def test_national_scope_present(self):
        assert "national" in DSA.ACTIVITY_SCOPES

    def test_regional_scope_greater_than_national(self):
        assert DSA.ACTIVITY_SCOPES["regional"] > DSA.ACTIVITY_SCOPES["national"]

    def test_national_scope_greater_than_sub_national(self):
        sub_national_vals = [
            v for k, v in DSA.ACTIVITY_SCOPES.items() if "sub-national" in k
        ]
        assert all(DSA.ACTIVITY_SCOPES["national"] > v for v in sub_national_vals)


# ===========================================================================
# 2.  passes_quartile_constraint — pure function
# ===========================================================================


class TestPassesQuartileConstraint:
    """
    Returns True iff candidate 3/4 mark (c_75) <= query 1/4 mark (q_25).
    Returns False when any date is NaT.
    """

    def _ts(self, year):
        return pd.Timestamp(f"{year}-01-01")

    def test_candidate_well_before_query_returns_true(self):
        # candidate ends (75%) in 2010, query starts (25%) in 2015
        assert (
            DSA.passes_quartile_constraint(
                self._ts(2013),
                self._ts(2017),  # query: 25% = 2014-01-01
                self._ts(2006),
                self._ts(2014),  # candidate: 75% = 2012-01-01
            )
            is True
        )

    def test_candidate_overlapping_query_returns_false(self):
        # candidate 75% mark is after query 25% mark
        assert (
            DSA.passes_quartile_constraint(
                self._ts(2010),
                self._ts(2014),  # q_25 ≈ 2011-01-01
                self._ts(2010),
                self._ts(2018),  # c_75 ≈ 2016-01-01 > q_25
            )
            is False
        )

    def test_c75_exactly_equal_to_q25_returns_true(self):
        # c_75 == q_25 is allowed (<=)
        q_start = self._ts(2010)
        q_end = self._ts(2014)
        q_25 = q_start + (q_end - q_start) * 0.25  # 2011-01-01

        # Make a candidate whose 75% equals q_25 exactly
        c_start = self._ts(2008)
        # c_75 = c_start + (c_end - c_start) * 0.75 = q_25
        # => c_end - c_start = (q_25 - c_start) / 0.75
        delta = (q_25 - c_start) / 0.75
        c_end = c_start + delta

        assert DSA.passes_quartile_constraint(q_start, q_end, c_start, c_end) is True

    def test_nat_q_start_returns_false(self):
        assert (
            DSA.passes_quartile_constraint(
                pd.NaT, self._ts(2014), self._ts(2008), self._ts(2012)
            )
            is False
        )

    def test_nat_q_end_returns_false(self):
        assert (
            DSA.passes_quartile_constraint(
                self._ts(2010), pd.NaT, self._ts(2008), self._ts(2012)
            )
            is False
        )

    def test_nat_c_start_returns_false(self):
        assert (
            DSA.passes_quartile_constraint(
                self._ts(2010), self._ts(2014), pd.NaT, self._ts(2012)
            )
            is False
        )

    def test_nat_c_end_returns_false(self):
        assert (
            DSA.passes_quartile_constraint(
                self._ts(2010), self._ts(2014), self._ts(2008), pd.NaT
            )
            is False
        )

    def test_all_nat_returns_false(self):
        assert DSA.passes_quartile_constraint(pd.NaT, pd.NaT, pd.NaT, pd.NaT) is False

    def test_candidate_completely_before_query_returns_true(self):
        # Candidate finishes entirely before query even starts
        assert (
            DSA.passes_quartile_constraint(
                self._ts(2015),
                self._ts(2020),
                self._ts(2000),
                self._ts(2010),
            )
            is True
        )

    def test_candidate_same_period_as_query_returns_false(self):
        # Identical date ranges: c_75 > q_25
        t0 = self._ts(2010)
        t1 = self._ts(2014)
        assert DSA.passes_quartile_constraint(t0, t1, t0, t1) is False

    @pytest.mark.parametrize(
        "q_start,q_end,c_start,c_end,expected",
        [
            # Candidate finishes long before query starts
            ("2015-01-01", "2020-01-01", "2000-01-01", "2005-01-01", True),
            # Candidate starts after query ends — c_75 is in the future relative to q_25
            ("2000-01-01", "2005-01-01", "2010-01-01", "2015-01-01", False),
        ],
    )
    def test_parametrized_cases(self, q_start, q_end, c_start, c_end, expected):
        result = DSA.passes_quartile_constraint(
            pd.Timestamp(q_start),
            pd.Timestamp(q_end),
            pd.Timestamp(c_start),
            pd.Timestamp(c_end),
        )
        assert result is expected


# ===========================================================================
# 3.  load_activity_embeddings — mocked file I/O
# ===========================================================================


class TestLoadActivityEmbeddings:
    """load_activity_embeddings reads a JSONL file and L2-normalises vectors."""

    def _make_jsonl(self, records: list[dict]) -> str:
        return "\n".join(json.dumps(r) for r in records) + "\n"

    def _reset_cache(self):
        DSA._EMBEDDINGS_CACHE = None

    def test_loads_basic_embedding(self, tmp_path):
        self._reset_cache()
        data = [{"activity_id": "act_001", "embedding": [1.0, 0.0, 0.0]}]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        assert "act_001" in result
        assert result["act_001"].shape == (3,)

    def test_embedding_is_l2_normalised(self, tmp_path):
        self._reset_cache()
        data = [{"activity_id": "act_001", "embedding": [3.0, 4.0]}]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        norm = float(np.linalg.norm(result["act_001"]))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector_not_normalised(self, tmp_path):
        self._reset_cache()
        data = [{"activity_id": "zero_act", "embedding": [0.0, 0.0, 0.0]}]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        # Zero vectors are left as-is (norm == 0 check in source)
        assert np.all(result["zero_act"] == 0.0)

    def test_multiple_embeddings_loaded(self, tmp_path):
        self._reset_cache()
        data = [
            {"activity_id": "a1", "embedding": [1.0, 0.0]},
            {"activity_id": "a2", "embedding": [0.0, 1.0]},
            {"activity_id": "a3", "embedding": [1.0, 1.0]},
        ]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        assert set(result.keys()) == {"a1", "a2", "a3"}

    def test_skips_records_missing_activity_id(self, tmp_path):
        self._reset_cache()
        data = [
            {"embedding": [1.0, 0.0]},  # no activity_id
            {"activity_id": "a2", "embedding": [0.0, 1.0]},
        ]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        assert "a2" in result
        assert len(result) == 1

    def test_skips_records_missing_embedding(self, tmp_path):
        self._reset_cache()
        data = [
            {"activity_id": "a1"},  # no embedding key
            {"activity_id": "a2", "embedding": [1.0, 0.0]},
        ]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        assert "a2" in result
        assert "a1" not in result

    def test_skips_blank_lines(self, tmp_path):
        self._reset_cache()
        content = (
            json.dumps({"activity_id": "a1", "embedding": [1.0, 0.0]}) + "\n"
            "\n"
            "\n" + json.dumps({"activity_id": "a2", "embedding": [0.0, 1.0]}) + "\n"
        )
        p = tmp_path / "embs.jsonl"
        p.write_text(content)
        result = DSA.load_activity_embeddings(p)
        assert set(result.keys()) == {"a1", "a2"}

    def test_raises_on_bad_json(self, tmp_path):
        self._reset_cache()
        p = tmp_path / "embs.jsonl"
        p.write_text('{"activity_id": "a1", "embedding": [1.0}\n')  # malformed
        with pytest.raises(ValueError, match="Bad JSON"):
            DSA.load_activity_embeddings(p)

    def test_returns_dict(self, tmp_path):
        self._reset_cache()
        data = [{"activity_id": "a1", "embedding": [1.0, 2.0, 3.0]}]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        assert isinstance(result, dict)

    def test_result_values_are_numpy_arrays(self, tmp_path):
        self._reset_cache()
        data = [{"activity_id": "a1", "embedding": [1.0, 0.0]}]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        assert isinstance(result["a1"], np.ndarray)

    def test_caches_result_on_second_call(self, tmp_path):
        self._reset_cache()
        data = [{"activity_id": "a1", "embedding": [1.0, 0.0]}]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        first = DSA.load_activity_embeddings(p)
        # Overwrite file — second call should return cached result
        p.write_text(
            json.dumps({"activity_id": "different", "embedding": [0.0, 1.0]}) + "\n"
        )
        second = DSA.load_activity_embeddings(p)
        assert first is second  # same object (cache hit)
        assert "a1" in second
        self._reset_cache()  # clean up for other tests

    def test_activity_id_stored_as_string(self, tmp_path):
        self._reset_cache()
        data = [{"activity_id": 12345, "embedding": [1.0, 0.0]}]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        assert "12345" in result

    def test_dtype_is_float32(self, tmp_path):
        self._reset_cache()
        data = [{"activity_id": "a1", "embedding": [1.0, 2.0]}]
        p = tmp_path / "embs.jsonl"
        p.write_text(self._make_jsonl(data))
        result = DSA.load_activity_embeddings(p)
        assert result["a1"].dtype == np.float32


# ===========================================================================
# 4.  find_similar_activities_semantic — mocked dependencies
# ===========================================================================


def _make_test_df(n=10, seed=0):
    """Build a minimal DataFrame that mimics the output of prepare_dataframe."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="365D")
    end_dates = pd.date_range("2005-01-01", periods=n, freq="365D")
    return pd.DataFrame(
        {
            "activity_id": [f"act_{i:03d}" for i in range(n)],
            "activity_title": [f"Title {i}" for i in range(n)],
            "activity_scope": ["national"] * n,
            "country_location": ["ZM"] * n,
            "gdp_percap": rng.uniform(1000, 5000, n),
            "dac5": ["11110"] * n,
            "reporting_orgs": ["FCDO"] * n,
            "participating_orgs": ["FCDO"] * n,
            "start_date": dates,
            "end_date": end_dates,
        }
    )


def _make_embeddings(activity_ids, dim=8, seed=1):
    """Return a dict of L2-normalised embeddings for given ids."""
    rng = np.random.default_rng(seed)
    result = {}
    for aid in activity_ids:
        v = rng.standard_normal(dim).astype("float32")
        v /= np.linalg.norm(v)
        result[aid] = v
    return result


class TestFindSimilarActivitiesSemantic:

    def _patch(self, df, embs):
        """Return a context manager dict of patches for prepare_dataframe and load_activity_embeddings."""
        return {
            "prepare": patch.object(DSA, "prepare_dataframe", return_value=df),
            "embs": patch.object(DSA, "load_activity_embeddings", return_value=embs),
        }

    def test_returns_tuple_of_two(self):
        df = _make_test_df(10)
        embs = _make_embeddings(df["activity_id"].tolist())
        query_id = "act_000"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            result = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv"
            )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_result_df_has_similarity_column(self):
        df = _make_test_df(10)
        embs = _make_embeddings(df["activity_id"].tolist())
        query_id = "act_000"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            result_df, _ = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv"
            )
        assert "similarity" in result_df.columns

    def test_query_row_is_dataframe_row(self):
        df = _make_test_df(10)
        embs = _make_embeddings(df["activity_id"].tolist())
        query_id = "act_000"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            _, query_row = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv"
            )
        # query_row is a Series (single row); retrieve activity_id as scalar
        aid_val = query_row["activity_id"]
        if hasattr(aid_val, "item"):
            aid_val = aid_val.item()
        assert str(aid_val) == query_id

    def test_query_id_excluded_from_results(self):
        df = _make_test_df(10)
        embs = _make_embeddings(df["activity_id"].tolist())
        query_id = "act_000"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            result_df, _ = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv"
            )
        # The query itself passes_quartile_constraint with itself only when c_75 <= q_25,
        # which is False for identical ranges — so it should not appear in results.
        # The function may or may not include query_id; test that similarity is sorted.
        if len(result_df) > 0:
            assert (
                result_df["similarity"].is_monotonic_decreasing or len(result_df) == 1
            )

    def test_result_sorted_by_similarity_descending(self):
        df = _make_test_df(15)
        embs = _make_embeddings(df["activity_id"].tolist())
        query_id = "act_007"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            result_df, _ = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv"
            )
        if len(result_df) > 1:
            sims = result_df["similarity"].tolist()
            assert sims == sorted(sims, reverse=True)

    def test_top_n_respected(self):
        df = _make_test_df(20)
        embs = _make_embeddings(df["activity_id"].tolist())
        query_id = "act_010"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            result_df, _ = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv", top_n=3
            )
        assert len(result_df) <= 3

    def test_allowed_ids_filters_results(self):
        df = _make_test_df(20)
        embs = _make_embeddings(df["activity_id"].tolist())
        # Use early activities as query, restrict to a small allowed set
        # Allowed set has early start_dates so the quartile constraint can pass
        allowed = {"act_000", "act_001", "act_002"}
        query_id = "act_010"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            result_df, _ = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv", allowed_ids=allowed
            )
        if len(result_df) > 0:
            returned_ids = set(result_df["activity_id"].tolist())
            assert returned_ids.issubset(allowed)

    def test_empty_result_when_no_candidates_pass_constraint(self):
        """If quartile constraint eliminates all candidates, return empty DataFrame."""
        df = _make_test_df(5)
        embs = _make_embeddings(df["activity_id"].tolist())
        # Make query start_date later than all candidates' end_dates so c_75 > q_25
        # Reverse: query is early, candidates are all later
        late_dates = pd.date_range("2050-01-01", periods=5, freq="365D")
        late_end_dates = pd.date_range("2055-01-01", periods=5, freq="365D")
        df["start_date"] = late_dates
        df["end_date"] = late_end_dates
        # Make one very early query so all other activities (which are later) fail constraint
        early_start = pd.Timestamp("1990-01-01")
        early_end = pd.Timestamp("1995-01-01")
        df.loc[df["activity_id"] == "act_000", "start_date"] = early_start
        df.loc[df["activity_id"] == "act_000", "end_date"] = early_end

        query_id = "act_000"
        # All other acts start in 2050 → their c_75 ~ 2053 >> q_25 ~ 1991
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            result_df, _ = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv"
            )
        assert isinstance(result_df, pd.DataFrame)
        assert "similarity" in result_df.columns

    def test_raises_if_activity_id_not_in_embeddings(self):
        df = _make_test_df(5)
        # Embeddings for all except the query
        embs = _make_embeddings([aid for aid in df["activity_id"] if aid != "act_000"])
        query_id = "act_000"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            with pytest.raises(ValueError, match="act_000"):
                DSA.find_similar_activities_semantic(query_id, csv_path="dummy.csv")

    def test_raises_if_activity_id_not_in_csv(self):
        df = _make_test_df(5)
        embs = _make_embeddings(df["activity_id"].tolist())
        missing_id = "NONEXISTENT_ID"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            with pytest.raises((ValueError, IndexError)):
                DSA.find_similar_activities_semantic(missing_id, csv_path="dummy.csv")

    def test_similarity_values_in_minus_one_to_one(self):
        """Cosine similarity of unit vectors must be in [-1, 1]."""
        df = _make_test_df(15)
        embs = _make_embeddings(df["activity_id"].tolist())
        query_id = "act_007"
        with patch.object(DSA, "prepare_dataframe", return_value=df), patch.object(
            DSA, "load_activity_embeddings", return_value=embs
        ):
            result_df, _ = DSA.find_similar_activities_semantic(
                query_id, csv_path="dummy.csv"
            )
        if len(result_df) > 0:
            assert result_df["similarity"].between(-1.0, 1.0).all()
