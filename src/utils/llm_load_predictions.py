"""
LLM prediction loading module.

Loads the 3 current RAG-based forecast JSONL files and exposes helpers for use
by A_overall_rating_fit_and_evaluate.py.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Set up utils path so this module can be imported standalone
UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from feature_engineering import parse_last_line_label_after_forecast

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

VARIANT_PATHS = [
    DATA_DIR
    / "rag_prompts_and_responses/outputs_onlysummary_no_knn_no_rag_onlysummary_no_knn_no_rag_s3_call_1.jsonl",
]


def load_predictions_from_jsonl(filepath, parser, series_name):
    """
    Generic loader: JSONL -> Series(activity_id -> parsed rating).

    Supports:
      - ChatGPT-style records: {"response": {"content": "..."}}
      - Gemini-style records: {"response_text": "..."}
    """
    preds = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            activity_id = record.get("activity_id")
            if activity_id is None:
                continue

            # Try ChatGPT-style first
            content = None
            resp = record.get("response")
            if isinstance(resp, dict):
                content = resp.get("content") or resp.get("text")

            # Fall back to plain response_text
            if not content:
                content = record.get("response_text")

            if not content:
                continue

            value = parser(content, record)
            if value is None:
                continue

            preds[activity_id] = value

    return pd.Series(preds, name=series_name)


def get_llm_prediction_configs():
    """
    Returns list of (name, path, parser, label) for the 3 current VARIANT_PATHS.
    """
    configs = []
    for p in VARIANT_PATHS:
        variant = p.stem.replace("outputs_forecast_ensemble_variant_", "")
        name = f"fewshot_variant_{variant}"
        label = f"fewshot variant {variant}"
        configs.append((name, p, parse_last_line_label_after_forecast, label))
    return configs
