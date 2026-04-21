"""
Tests for src/utils/llm_load_predictions.py.

Covers:
  - load_predictions_from_jsonl: ChatGPT-style, Gemini-style, missing fields,
    empty input, blank lines, parser returning None, malformed JSON.
  - get_llm_prediction_configs: return shape and field types.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / "src" / "utils"

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

import llm_load_predictions as llp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _identity_parser(content, record):
    """Return the raw content string unchanged (for simple tests)."""
    return content


def _none_parser(content, record):
    """Always return None — simulates a failed parse."""
    return None


# ---------------------------------------------------------------------------
# load_predictions_from_jsonl
# ---------------------------------------------------------------------------


class TestLoadPredictionsFromJsonl:
    """Tests for load_predictions_from_jsonl()."""

    def test_chatgpt_style_content_key(self, tmp_path):
        """ChatGPT-style records with response.content are parsed."""
        records = [
            {"activity_id": "act_001", "response": {"content": "hello world"}},
        ]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert "act_001" in result.index
        assert result["act_001"] == "hello world"

    def test_chatgpt_style_text_key(self, tmp_path):
        """ChatGPT-style records with response.text are parsed."""
        records = [
            {"activity_id": "act_002", "response": {"text": "some text"}},
        ]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert result["act_002"] == "some text"

    def test_gemini_style_response_text(self, tmp_path):
        """Gemini-style records with response_text are parsed."""
        records = [
            {"activity_id": "act_003", "response_text": "gemini output"},
        ]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert result["act_003"] == "gemini output"

    def test_chatgpt_content_takes_precedence_over_response_text(self, tmp_path):
        """When both response.content and response_text exist, content wins."""
        records = [
            {
                "activity_id": "act_004",
                "response": {"content": "from_content"},
                "response_text": "from_response_text",
            },
        ]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert result["act_004"] == "from_content"

    def test_missing_activity_id_skipped(self, tmp_path):
        """Records without activity_id are silently skipped."""
        records = [
            {"response": {"content": "no id here"}},
            {"activity_id": "act_005", "response": {"content": "has id"}},
        ]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert len(result) == 1
        assert "act_005" in result.index

    def test_missing_content_skipped(self, tmp_path):
        """Records where no content field resolves are skipped."""
        records = [
            {"activity_id": "act_006"},  # no response at all
            {
                "activity_id": "act_007",
                "response": {},
            },  # response dict but no content/text
            {
                "activity_id": "act_008",
                "response": "a string",
            },  # response is not a dict
            {"activity_id": "act_009", "response": {"content": "ok"}},
        ]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert list(result.index) == ["act_009"]

    def test_parser_returning_none_skips_record(self, tmp_path):
        """If the parser returns None the record is excluded from the Series."""
        records = [
            {"activity_id": "act_010", "response": {"content": "something"}},
        ]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _none_parser, "test")
        assert len(result) == 0

    def test_empty_file_returns_empty_series(self, tmp_path):
        """An empty JSONL file yields an empty Series."""
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_blank_lines_skipped(self, tmp_path):
        """Blank lines in the file do not cause errors and are ignored."""
        record = {"activity_id": "act_011", "response": {"content": "val"}}
        p = tmp_path / "blanks.jsonl"
        p.write_text("\n\n" + json.dumps(record) + "\n\n")
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert "act_011" in result.index

    def test_series_name_is_set(self, tmp_path):
        """The returned Series carries the requested name."""
        records = [
            {"activity_id": "act_012", "response_text": "x"},
        ]
        p = tmp_path / "named.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "my_series")
        assert result.name == "my_series"

    def test_multiple_records_all_loaded(self, tmp_path):
        """All valid records in a multi-record file are loaded."""
        records = [
            {"activity_id": f"act_{i:03d}", "response": {"content": f"val_{i}"}}
            for i in range(10)
        ]
        p = tmp_path / "multi.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert len(result) == 10

    def test_parser_receives_full_record(self, tmp_path):
        """The parser is called with the full record dict as second argument."""
        received_records = []

        def capturing_parser(content, record):
            received_records.append(record)
            return content

        records = [
            {"activity_id": "act_013", "response": {"content": "data"}, "extra": 42},
        ]
        p = tmp_path / "capture.jsonl"
        _write_jsonl(p, records)
        llp.load_predictions_from_jsonl(p, capturing_parser, "test")
        assert len(received_records) == 1
        assert received_records[0]["extra"] == 42

    def test_file_not_found_raises(self, tmp_path):
        """A non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            llp.load_predictions_from_jsonl(
                tmp_path / "missing.jsonl", _identity_parser, "test"
            )

    def test_malformed_json_raises(self, tmp_path):
        """A line with invalid JSON raises json.JSONDecodeError."""
        p = tmp_path / "bad.jsonl"
        p.write_text("{not valid json}\n")
        with pytest.raises(json.JSONDecodeError):
            llp.load_predictions_from_jsonl(p, _identity_parser, "test")

    def test_response_dict_with_none_content_falls_back_to_response_text(
        self, tmp_path
    ):
        """response.content is None → falls back to response_text."""
        records = [
            {
                "activity_id": "act_014",
                "response": {"content": None},
                "response_text": "fallback_text",
            },
        ]
        p = tmp_path / "fallback.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert result["act_014"] == "fallback_text"

    def test_duplicate_activity_id_last_write_wins(self, tmp_path):
        """When the same activity_id appears twice, the second value overwrites the first."""
        records = [
            {"activity_id": "act_dup", "response": {"content": "first"}},
            {"activity_id": "act_dup", "response": {"content": "second"}},
        ]
        p = tmp_path / "dup.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert result["act_dup"] == "second"

    def test_returns_pandas_series(self, tmp_path):
        """Return type is always pd.Series regardless of input size."""
        records = [{"activity_id": "act_015", "response_text": "v"}]
        p = tmp_path / "type_check.jsonl"
        _write_jsonl(p, records)
        result = llp.load_predictions_from_jsonl(p, _identity_parser, "test")
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# get_llm_prediction_configs
# ---------------------------------------------------------------------------


class TestGetLlmPredictionConfigs:
    """Tests for get_llm_prediction_configs()."""

    @pytest.fixture
    def configs(self):
        return llp.get_llm_prediction_configs()

    def test_returns_a_list(self, configs):
        assert isinstance(configs, list)

    def test_length_matches_variant_paths(self, configs):
        assert len(configs) == len(llp.VARIANT_PATHS)

    def test_each_entry_is_4_tuple(self, configs):
        for entry in configs:
            assert len(entry) == 4, f"Expected 4-tuple, got length {len(entry)}"

    def test_name_is_string(self, configs):
        for name, _path, _parser, _label in configs:
            assert isinstance(name, str)

    def test_path_is_path_object(self, configs):
        for _name, path, _parser, _label in configs:
            assert isinstance(path, Path)

    def test_parser_is_callable(self, configs):
        for _name, _path, parser, _label in configs:
            assert callable(parser)

    def test_label_is_string(self, configs):
        for _name, _path, _parser, label in configs:
            assert isinstance(label, str)

    def test_name_contains_fewshot_variant(self, configs):
        """Each config name starts with 'fewshot_variant_'."""
        for name, _path, _parser, _label in configs:
            assert name.startswith(
                "fewshot_variant_"
            ), f"Name '{name}' does not start with 'fewshot_variant_'"

    def test_parser_is_parse_last_line_label_after_forecast(self, configs):
        """The parser used must be the module-level parse_last_line_label_after_forecast."""
        from feature_engineering import (
            parse_last_line_label_after_forecast,
        )

        for _name, _path, parser, _label in configs:
            assert parser is parse_last_line_label_after_forecast

    def test_paths_point_into_data_dir(self, configs):
        """Each variant path should be inside the repo's data directory."""
        data_dir = llp.DATA_DIR
        for _name, path, _parser, _label in configs:
            assert str(data_dir) in str(
                path
            ), f"Path '{path}' does not appear to be under DATA_DIR '{data_dir}'"

    def test_no_duplicate_names(self, configs):
        names = [name for name, *_ in configs]
        assert len(names) == len(set(names)), "Config names must be unique"

    def test_no_duplicate_paths(self, configs):
        paths = [str(path) for _, path, *_ in configs]
        assert len(paths) == len(set(paths)), "Config paths must be unique"
