"""
Print feature lists for all 4 forecasting models.

Overall rating and outcome tag features are saved by their training scripts (A, G).
Cost effectiveness features are saved by L. LLM forecasting features are static.

Run from repo root or any directory; paths are resolved relative to this file.
"""

import json
from pathlib import Path

FEATURE_LISTS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "feature_lists"
)

MODELS = [
    "overall_rating_features.json",
    "outcome_tag_nolimits_features.json",
    "cost_effectiveness_features.json",
    "llm_forecasting_features.json",
]


def load_feature_list(filename: str) -> dict | None:
    path = FEATURE_LISTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    for filename in MODELS:
        data = load_feature_list(filename)
        if data is None:
            print(f"\n{'=' * 60}")
            print(f"MODEL: {filename.replace('_features.json', '')}")
            print(f"  [NOT FOUND] Run the training script first to generate {filename}")
            continue

        model = data.get("model", filename)
        features = data.get("features", [])
        n = data.get("n_features", len(features))
        note = data.get("note")

        print(f"\n{'=' * 60}")
        print(f"MODEL: {model}  ({n} features)")
        if note:
            print(f"  NOTE: {note}")
        print(f"  Features:")
        for i, feat in enumerate(features, 1):
            print(f"    {i:3d}. {feat}")


if __name__ == "__main__":
    main()
