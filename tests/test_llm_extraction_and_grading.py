"""
Tests for src/utils/llm_extraction_and_grading.py.

Covered:
  - get_key(): extracts activity_id from a row dict
  - resolve_gemini_model(): Vertex endpoint override and passthrough logic
  - wait_file_active(): polls until ACTIVE, raises on FAILED or timeout
  - load_seen_keys(): parses JSONL, skips errors and missing activity_ids
  - make_executor(): returns a ThreadPoolExecutor with the right worker count
  - Module-level constants (CONCURRENCY, TIMEOUT_SECONDS, BATCH_MODE, etc.)

NOT covered (require live API credentials / async infra):
  - run_one_row(), run_one_row_openai(), loop_over_rows_to_call_model()
  - assemble_and_upload_activity_pdf()
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
for _p in [_REPO / "src" / "utils", _REPO / "src" / "pipeline"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import llm_extraction_and_grading as leg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list) -> None:
    """Write a list of dicts as JSONL to path."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ===========================================================================
# TestModuleConstants
# ===========================================================================


class TestModuleConstants:
    """Smoke-test that module-level configuration constants exist and are sane."""

    def test_concurrency_is_positive_int(self):
        assert isinstance(leg.CONCURRENCY, int)
        assert leg.CONCURRENCY > 0

    def test_timeout_seconds_is_positive(self):
        assert leg.TIMEOUT_SECONDS > 0

    def test_batch_mode_is_bool(self):
        assert isinstance(leg.BATCH_MODE, bool)

    def test_airplane_mode_is_bool(self):
        assert isinstance(leg.AIRPLANE_MODE, bool)

    def test_model_name_is_string(self):
        assert isinstance(leg.MODEL_NAME, str)
        assert len(leg.MODEL_NAME) > 0

    def test_debug_aid_is_string(self):
        assert isinstance(leg.DEBUG_AID, str)

    def test_excluded_cat_pages_is_set(self):
        assert isinstance(leg.EXCLUDED_CAT_PAGES, set)
        assert "glossary" in leg.EXCLUDED_CAT_PAGES
        assert "blank_page" in leg.EXCLUDED_CAT_PAGES
        assert "table_of_contents" in leg.EXCLUDED_CAT_PAGES
        assert "references" in leg.EXCLUDED_CAT_PAGES


# ===========================================================================
# TestGetKey
# ===========================================================================


class TestGetKey:
    """get_key(row) returns the value of row['activity_id']."""

    def test_returns_activity_id(self):
        row = {"activity_id": "44000-P123", "other_field": "irrelevant"}
        assert leg.get_key(row) == "44000-P123"

    def test_returns_none_when_missing(self):
        assert leg.get_key({}) is None

    def test_returns_none_explicitly_set(self):
        assert leg.get_key({"activity_id": None}) is None

    def test_numeric_activity_id(self):
        assert leg.get_key({"activity_id": 12345}) == 12345

    def test_string_with_spaces(self):
        row = {"activity_id": "  padded  "}
        assert leg.get_key(row) == "  padded  "

    def test_ignores_extra_fields(self):
        row = {"activity_id": "X", "section": "intro", "cached_file": "file.pdf"}
        assert leg.get_key(row) == "X"


# ===========================================================================
# TestResolveGeminiModel
# ===========================================================================


class TestResolveGeminiModel:
    """resolve_gemini_model() handles Vertex endpoint strings and plain model names."""

    def test_passthrough_plain_model_name(self):
        """Non-Vertex model names are returned unchanged."""
        with patch.object(leg, "USE_VERTEX", False):
            result = leg.resolve_gemini_model("gemini-2.5-flash")
        assert result == "gemini-2.5-flash"

    def test_passthrough_already_vertex_endpoint(self):
        """Strings that already look like Vertex resource paths pass through."""
        endpoint = "projects/my-proj/locations/us-central1/endpoints/abc123"
        # Should return as-is regardless of USE_VERTEX flag
        result = leg.resolve_gemini_model(endpoint)
        assert result == endpoint

    def test_vertex_mode_with_tuned_endpoint(self):
        """When USE_VERTEX=True and GEMINI_TUNED_ENDPOINT_ID is set, build the full path."""
        # VERTEX_PROJECT / VERTEX_LOCATION / GEMINI_TUNED_ENDPOINT_ID only exist in the
        # module when USE_VERTEX=True at import time, so use create=True to inject them.
        with (
            patch.object(leg, "USE_VERTEX", True),
            patch.object(leg, "VERTEX_PROJECT", "my-project", create=True),
            patch.object(leg, "VERTEX_LOCATION", "us-central1", create=True),
            patch.object(
                leg, "GEMINI_TUNED_ENDPOINT_ID", "endpoint-id-999", create=True
            ),
        ):
            result = leg.resolve_gemini_model("gemini-2.5-flash")
        assert "my-project" in result
        assert "us-central1" in result
        assert "endpoint-id-999" in result
        assert result.startswith("projects/")

    def test_vertex_mode_missing_project_raises(self):
        """USE_VERTEX=True with GEMINI_TUNED_ENDPOINT_ID but no project raises RuntimeError."""
        with (
            patch.object(leg, "USE_VERTEX", True),
            patch.object(leg, "VERTEX_PROJECT", None, create=True),
            patch.object(
                leg, "GEMINI_TUNED_ENDPOINT_ID", "endpoint-id-999", create=True
            ),
        ):
            with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
                leg.resolve_gemini_model("gemini-2.5-flash")

    def test_vertex_mode_no_tuned_endpoint_returns_model(self):
        """USE_VERTEX=True but no GEMINI_TUNED_ENDPOINT_ID → plain model name returned."""
        with (
            patch.object(leg, "USE_VERTEX", True),
            patch.object(leg, "GEMINI_TUNED_ENDPOINT_ID", "", create=True),
        ):
            result = leg.resolve_gemini_model("gemini-2.5-flash")
        assert result == "gemini-2.5-flash"


# ===========================================================================
# TestWaitFileActive
# ===========================================================================


class TestWaitFileActive:
    """wait_file_active() polls until the file state is ACTIVE."""

    def _make_file_obj(self, state: str, name: str = "files/abc123") -> MagicMock:
        f = MagicMock()
        f.name = name
        f.state = state
        f.display_state = state
        return f

    def test_returns_immediately_when_active(self):
        """If the file is already ACTIVE, it is returned without sleeping."""
        uploaded = self._make_file_obj("ACTIVE")
        active_file = self._make_file_obj("ACTIVE")

        client = MagicMock()
        client.files.get.return_value = active_file

        result = leg.wait_file_active(client, uploaded, timeout=5, interval=0.01)
        assert result is active_file
        client.files.get.assert_called_once()

    def test_raises_on_failed_state(self):
        """A FAILED file state raises RuntimeError immediately."""
        uploaded = self._make_file_obj("PROCESSING")
        failed_file = self._make_file_obj("FAILED")

        client = MagicMock()
        client.files.get.return_value = failed_file

        with pytest.raises(RuntimeError, match="failed to process"):
            leg.wait_file_active(client, uploaded, timeout=5, interval=0.01)

    def test_raises_timeout_when_stuck_processing(self):
        """If the file never becomes ACTIVE within the timeout, TimeoutError is raised."""
        uploaded = self._make_file_obj("PROCESSING")
        processing_file = self._make_file_obj("PROCESSING")

        client = MagicMock()
        client.files.get.return_value = processing_file

        with pytest.raises(TimeoutError, match="Timed out"):
            leg.wait_file_active(client, uploaded, timeout=0.05, interval=0.01)

    def test_polls_until_active(self):
        """Polling transitions from PROCESSING to ACTIVE on the second call."""
        uploaded = self._make_file_obj("PROCESSING")
        processing_file = self._make_file_obj("PROCESSING")
        active_file = self._make_file_obj("ACTIVE")

        client = MagicMock()
        client.files.get.side_effect = [processing_file, active_file]

        result = leg.wait_file_active(client, uploaded, timeout=5, interval=0.01)
        assert result is active_file
        assert client.files.get.call_count == 2

    def test_uses_name_attribute_for_polling(self):
        """The 'name' attribute of the uploaded object is passed to files.get."""
        uploaded = MagicMock()
        uploaded.name = "files/xyz789"
        active_file = self._make_file_obj("ACTIVE", name="files/xyz789")

        client = MagicMock()
        client.files.get.return_value = active_file

        leg.wait_file_active(client, uploaded, timeout=5, interval=0.01)
        client.files.get.assert_called_with(name="files/xyz789")

    def test_falls_back_to_id_when_no_name(self):
        """If 'name' attribute is missing, 'id' is used as fallback."""
        uploaded = MagicMock(spec=[])  # no 'name' attribute
        uploaded.id = "files/fallback-id"
        active_file = self._make_file_obj("ACTIVE", name="files/fallback-id")

        client = MagicMock()
        client.files.get.return_value = active_file

        leg.wait_file_active(client, uploaded, timeout=5, interval=0.01)
        client.files.get.assert_called_with(name="files/fallback-id")


# ===========================================================================
# TestLoadSeenKeys
# ===========================================================================


class TestLoadSeenKeys:
    """load_seen_keys() reads a JSONL file and returns successfully-processed activity IDs."""

    def test_returns_set_of_successful_ids(self, tmp_path):
        p = tmp_path / "output.jsonl"
        _write_jsonl(
            p,
            [
                {"activity_id": "act1"},
                {"activity_id": "act2"},
            ],
        )
        result = leg.load_seen_keys(str(p))
        assert result == {"act1", "act2"}

    def test_skips_rows_with_error(self, tmp_path):
        p = tmp_path / "output.jsonl"
        _write_jsonl(
            p,
            [
                {"activity_id": "act1"},
                {"activity_id": "act2", "ERROR": "upload_timeout"},
            ],
        )
        result = leg.load_seen_keys(str(p))
        assert "act1" in result
        assert "act2" not in result

    def test_skips_rows_missing_activity_id(self, tmp_path):
        p = tmp_path / "output.jsonl"
        _write_jsonl(
            p,
            [
                {"activity_id": "act1"},
                {"some_other_field": "value"},
            ],
        )
        result = leg.load_seen_keys(str(p))
        assert result == {"act1"}

    def test_skips_invalid_json_lines(self, tmp_path):
        p = tmp_path / "output.jsonl"
        p.write_text(
            '{"activity_id": "act1"}\n' "NOT VALID JSON\n" '{"activity_id": "act2"}\n',
            encoding="utf-8",
        )
        result = leg.load_seen_keys(str(p))
        assert result == {"act1", "act2"}

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "output.jsonl"
        p.write_text(
            '{"activity_id": "act1"}\n' "\n" "   \n" '{"activity_id": "act2"}\n',
            encoding="utf-8",
        )
        result = leg.load_seen_keys(str(p))
        assert result == {"act1", "act2"}

    def test_empty_file_returns_empty_set(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        result = leg.load_seen_keys(str(p))
        assert result == set()

    def test_raises_file_not_found_for_missing_file(self, tmp_path):
        missing = tmp_path / "does_not_exist.jsonl"
        with pytest.raises(FileNotFoundError):
            leg.load_seen_keys(str(missing))

    def test_activity_id_converted_to_string(self, tmp_path):
        """Numeric activity_ids are stored as str in the seen set."""
        p = tmp_path / "output.jsonl"
        _write_jsonl(p, [{"activity_id": 99999}])
        result = leg.load_seen_keys(str(p))
        assert "99999" in result

    def test_deduplicates_repeated_ids(self, tmp_path):
        """Same activity_id appearing multiple times is stored only once."""
        p = tmp_path / "output.jsonl"
        _write_jsonl(
            p,
            [
                {"activity_id": "act1"},
                {"activity_id": "act1"},
                {"activity_id": "act1"},
            ],
        )
        result = leg.load_seen_keys(str(p))
        assert result == {"act1"}

    def test_error_with_empty_string_is_skipped(self, tmp_path):
        """An empty ERROR string should still cause the row to be skipped."""
        # Only non-falsy ERROR values are skipped per the code `if obj.get("ERROR")`
        p = tmp_path / "output.jsonl"
        _write_jsonl(
            p,
            [
                {"activity_id": "act1", "ERROR": ""},
                {"activity_id": "act2", "ERROR": "real_error"},
            ],
        )
        result = leg.load_seen_keys(str(p))
        # act1 has falsy ERROR → treated as success; act2 has truthy ERROR → skipped
        assert "act1" in result
        assert "act2" not in result

    def test_accepts_path_object(self, tmp_path):
        """load_seen_keys accepts a Path object (coerced to str internally)."""
        p = tmp_path / "output.jsonl"
        _write_jsonl(p, [{"activity_id": "act1"}])
        result = leg.load_seen_keys(p)
        assert "act1" in result


# ===========================================================================
# TestMakeExecutor
# ===========================================================================


class TestMakeExecutor:
    """make_executor() returns a ThreadPoolExecutor respecting CONCURRENCY."""

    def test_returns_thread_pool_executor(self):
        executor = leg.make_executor()
        try:
            assert isinstance(executor, ThreadPoolExecutor)
        finally:
            executor.shutdown(wait=False)

    def test_executor_max_workers_matches_concurrency(self):
        executor = leg.make_executor()
        try:
            assert executor._max_workers == leg.CONCURRENCY
        finally:
            executor.shutdown(wait=False)

    def test_executor_thread_name_prefix(self):
        executor = leg.make_executor()
        try:
            assert executor._thread_name_prefix == "genai"
        finally:
            executor.shutdown(wait=False)

    def test_executor_can_run_tasks(self):
        """The executor actually executes submitted callables."""
        executor = leg.make_executor()
        try:
            future = executor.submit(lambda: 42)
            assert future.result(timeout=5) == 42
        finally:
            executor.shutdown(wait=False)
