"""
Print the number of activities in each evaluation set used by the thesis,
broken down by train/val/test, and show which sets are subsets of each other.

This script reads CSV files saved by each evaluation script at the point of
the train/val/test split, rather than recreating their filtering logic.

The CSV files are written by:
  overall_ratings    -> src/pipeline/A_overall_rating_fit_and_evaluate.py  -> data/eval_set_sizes/overall_ratings_splits.csv
  outcome_tags       -> src/pipeline/G_outcome_tag_train.py                -> data/eval_set_sizes/outcome_tags_splits.csv
  cost_effectiveness -> src/pipeline/L_cost_effectiveness_train_and_score.py -> data/eval_set_sizes/cost_effectiveness_splits.csv
  ai_forecasting     -> src/pipeline/F_llm_score_forecast_narratives.py    -> data/eval_set_sizes/ai_forecasting_splits_set_a.csv

Evaluation sets:
  1. Overall ratings   (A_overall_rating_fit_and_evaluate.py)
       Activities from merged_overall_ratings.jsonl restricted to 4 reporting orgs,
       requiring is_completed==1 and a valid start_date.
       Core forecasting dataset for the rating prediction model.

  2. Outcome tags      (G_outcome_tag_train.py)
       Same 4-org, rated-activity universe as overall ratings.
       Tags are features/targets joined onto the activity; lack of a tag for an
       activity does not exclude it from the dataset.

  3. Cost-effectiveness outcomes  (L_cost_effectiveness_train_and_score.py)
       Any activity in the info CSV with either (a) a wanted quantitative outcome
       (BCR, yield, RoR, area protected, CO2, etc.) OR (b) an overall rating.
       Restricted to the same 4 orgs, requires start_date and is_completed==1.
       The ZAGG aggregate z-score is the primary cost-effectiveness target.

  4. AI / narrative forecasting   (F_llm_score_forecast_narratives.py, 2-pane plot)
       The strict intersection of: val-split activities (from overall_ratings split)
       that also have an LLM forecast, a parseable forecast rating, and a graded
       outcome. Only the val split was run through the LLM due to API cost.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
EVAL_DIR = DATA_DIR / "eval_set_sizes"

SOURCES = {
    "overall_ratings": EVAL_DIR / "overall_ratings_splits.csv",
    "outcome_tags": EVAL_DIR / "outcome_tags_splits.csv",
    "cost_effectiveness": EVAL_DIR / "cost_effectiveness_splits.csv",
    "ai_forecasting": EVAL_DIR / "ai_forecasting_splits_set_a.csv",
}


def load_split(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  MISSING: {path}")
        print("    Run the corresponding script first to generate this file.")
        return None
    df = pd.read_csv(path, dtype={"activity_id": str})
    df["activity_id"] = df["activity_id"].str.strip()
    return df


def split_counts(df: pd.DataFrame) -> dict:
    counts = df["split"].value_counts().to_dict()
    return {
        "train": counts.get("train", 0),
        "val": counts.get("val", 0),
        "test": counts.get("test", 0),
        "total": len(df),
    }


def ids(df: pd.DataFrame) -> set:
    return set(df["activity_id"])


def main():
    print("=" * 70)
    print("Loading saved split CSVs from data/eval_set_sizes/")
    print("=" * 70)

    dfs = {}
    for name, path in SOURCES.items():
        print(f"\n  {name}: {path.name}")
        df = load_split(path, name)
        dfs[name] = df

    missing = [k for k, v in dfs.items() if v is None]
    if missing:
        print(f"\nCannot proceed: {len(missing)} file(s) missing: {missing}")
        return

    # -- Summary table --------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"\n  {'Set':<40}  {'train':>6}  {'val':>6}  {'test':>6}  {'total':>6}")
    print(f"  {'-'*40}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    counts = {}
    for name, df in dfs.items():
        c = split_counts(df)
        counts[name] = c
        if name == "ai_forecasting":
            print(f"  {name:<40}  {'--':>6}  {c['val']:>6}  {'--':>6}  {c['total']:>6}")
        else:
            print(
                f"  {name:<40}  {c['train']:>6}  {c['val']:>6}  {c['test']:>6}  {c['total']:>6}"
            )

    # -- Subset relationships -------------------------------------------------
    print()
    print("=" * 70)
    print("SUBSET RELATIONSHIPS")
    print("=" * 70)

    s_rat = ids(dfs["overall_ratings"])
    s_tag = ids(dfs["outcome_tags"])
    s_ce = ids(dfs["cost_effectiveness"])
    s_ai = ids(dfs["ai_forecasting"])

    # val-split IDs for overall_ratings, for checking ai_forecasting subset
    val_rat = set(
        dfs["overall_ratings"][dfs["overall_ratings"]["split"] == "val"]["activity_id"]
    )

    print("\n  overall_ratings == outcome_tags?  ", end="")
    if s_rat == s_tag:
        print("YES -- identical activity sets")
    else:
        only_rat = s_rat - s_tag
        only_tag = s_tag - s_rat
        print("NO")
        print(f"    In overall_ratings only: {len(only_rat)}")
        print(f"    In outcome_tags only:    {len(only_tag)}")

    print("\n  overall_ratings subset cost_effectiveness?  ", end="")
    if s_rat <= s_ce:
        print("YES")
    else:
        print(f"NO -- {len(s_rat - s_ce)} rated activities not in cost_effectiveness")
        print(
            "    (these are likely rated but not marked is_completed in the outcomes script)"
        )

    print("\n  cost_effectiveness subset overall_ratings?  ", end="")
    if s_ce <= s_rat:
        print("YES")
    else:
        ce_only = s_ce - s_rat
        print(
            f"NO -- {len(ce_only)} cost_effectiveness activities not in overall_ratings"
        )
        print("    (these have quantitative outcomes but no overall rating)")

    print("\n  ai_forecasting subset val_split(overall_ratings)?  ", end="")
    if s_ai <= val_rat:
        print("YES")
    else:
        ai_not_val = s_ai - val_rat
        print(f"NO -- {len(ai_not_val)} AI activities not in overall_ratings val split")

    n_removed_llm = len(val_rat) - len(s_ai)
    print(f"\n  Val activities NOT covered by ai_forecasting: {len(val_rat - s_ai)}")
    print("\n  LLM FILTERING SUMMARY:")
    print(f"    Overall-ratings val split:        {len(val_rat):>4}")
    print(f"    Removed (no forecast/grade/parse):{n_removed_llm:>4}")
    print(f"    AI forecasting set (kept):        {len(s_ai):>4}")

    # -- LaTeX summary table --------------------------------------------------
    DISPLAY = {
        "overall_ratings": "Overall ratings",
        "outcome_tags": "Outcome tags",
        "cost_effectiveness": "Cost-effectiveness",
        "ai_forecasting": "AI forecasting",
    }
    print()
    print("% requires: \\usepackage{booktabs}")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(
        r"\caption{Sizes of the four evaluation datasets used in this thesis, broken down by train/validation/test split. The AI forecasting set uses only the validation split (the LLM was run on validation activities only due to API Sizes of the four evaluation datasets used in this thesis, broken down by train/validation/test split. The AI forecasting set shows only the validation split (due to API cost and temporal leakage on the test set).}"
    )
    print(r"\label{tab:eval_set_sizes}")
    print(r"\begin{tabular}{lrrrr}")
    print(r"\toprule")
    print(r"Dataset & Train & Val & Test & Total \\")
    print(r"\midrule")
    for name, _df in dfs.items():
        c = counts[name]
        label = DISPLAY[name]
        if name == "ai_forecasting":
            print(f"{label} & -- & {c['val']} & -- & {c['total']} \\\\")
        else:
            print(
                f"{label} & {c['train']} & {c['val']} & {c['test']} & {c['total']} \\\\"
            )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # -- Pairwise overlap counts -----------------------------------------------
    print()
    print("=" * 70)
    print("PAIRWISE OVERLAPS")
    print("=" * 70)
    sets = {
        "overall_ratings": s_rat,
        "outcome_tags": s_tag,
        "cost_effectiveness": s_ce,
        "ai_forecasting": s_ai,
    }
    names = list(sets.keys())
    print(f"\n  {'':30}", end="")
    for n in names:
        print(f"  {n[:18]:>18}", end="")
    print()
    for n1 in names:
        print(f"  {n1[:30]:30}", end="")
        for n2 in names:
            overlap = len(sets[n1] & sets[n2])
            print(f"  {overlap:>18}", end="")
        print()


if __name__ == "__main__":
    main()
