"""
Utilities to classify activities as 'loan' or 'disbursement'.

Primary entrypoint for other scripts:
    lod_df = load_loan_or_disbursement()

Which returns a pandas DataFrame indexed by activity_id with columns:
    - finance_label: 'loan' / 'disbursement' / None
    - finance_is_loan: 1.0 (loan), 0.0 (disbursement), NaN (unknown)
    - finance_class_method: 'misc' / 'finance_type' / 'transactions' / None
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd  # NEW

BASE = Path(__file__).resolve().parent.parent.parent / "data"

MERGED_RANKINGS = BASE / "merged_overall_ratings.jsonl"
OUTPUTS_MISC = BASE / "outputs_misc.jsonl"
DEFAULT_FINANCE_TYPES_CSV = BASE / "default_finance_types.csv"
TXN_SUMMARY = BASE / "iati_transactions_summary.csv"


# ---- Loaders (unchanged logic) ----


def load_merged_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            aid = (obj.get("activity_id") or "").strip()
            if aid:
                ids.add(aid)
    return ids


def load_outputs_misc(path: Path) -> dict[str, dict]:
    """
    Map activity_id -> dict with optional disbursement_total/loan_total and units.
    Uses the first record per activity_id (Baseline etc.).
    """
    by_aid: dict[str, dict] = {}
    if not path.exists():
        return by_aid

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            aid = (row.get("activity_id") or "").strip()
            if not aid:
                continue
            if aid in by_aid:
                continue

            rt_raw = row.get("response_text")
            if not rt_raw:
                continue
            try:
                rt = json.loads(rt_raw)
            except Exception:
                continue

            disb = rt.get("disbursement_total")
            loan = rt.get("loan_total")
            disb_units = rt.get("disbursement_units")
            loan_units = rt.get("loan_units")

            if disb is None and loan is None:
                continue

            by_aid[aid] = {
                "disbursement_total": disb,
                "loan_total": loan,
                "disbursement_units": disb_units,
                "loan_units": loan_units,
            }
    return by_aid


def load_default_finance_types(path: Path) -> dict[str, str]:
    """
    Map activity_id -> default-finance-type/@code (string code, e.g. '110', '421').
    Reads from pre-extracted CSV (activity_id, default_finance_type_code).
    """
    if not path.exists():
        raise FileNotFoundError(f"Default finance types CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["activity_id"], df["default_finance_type_code"]))


# Finance Type classification: loan vs disbursement vs other
LOAN_FT_CODES = {
    # classic loans and related instruments
    "410",
    "411",
    "412",
    "413",
    "414",
    "421",
    "422",
    "431",
    "423",
    "424",
    "425",
    "451",
    "452",
    "453",
    "810",
    "811",
    "910",
    "911",
    "912",
    "913",
    "1100",
}

DISB_FT_CODES = {
    "110",
    "111",
    "210",
    "211",
    "610",
    "611",
    "612",
    "613",
    "614",
    "615",
    "616",
    "617",
    "618",
    "620",
    "621",
    "622",
    "623",
    "624",
    "625",
    "626",
    "627",
    "630",
    "631",
    "632",
    "633",
    "634",
    "635",
    "636",
    "637",
    "638",
    "639",
}


def classify_from_finance_type(code: str) -> str | None:
    if code in LOAN_FT_CODES:
        return "loan"
    if code in DISB_FT_CODES:
        return "disbursement"
    return None


def load_txn_summary(path: Path) -> dict[str, dict[str, Decimal]]:
    """
    Map activity_id -> {transaction_type_code -> total_amount (Decimal)}.
    """
    by_aid: dict[str, dict[str, Decimal]] = {}
    if not path.exists():
        raise FileNotFoundError(
            f"Required transaction summary file not found: {path}\n"
            f"Copy it from forecasting_iati: cp /home/dmrivers/Code/forecasting_iati/data/iati_transactions_summary.csv {path}"
        )

    with path.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            aid = (row.get("activity_id") or "").strip()
            if not aid:
                continue
            code = (row.get("transaction_type_code") or "").strip()
            if not code:
                continue
            amt_str = (row.get("total_amount") or "").strip()
            if not amt_str:
                continue
            amt = Decimal(amt_str)

            d = by_aid.setdefault(aid, {})
            d[code] = d.get(code, Decimal("0")) + amt

    return by_aid


# ---- Classification heuristics ----


def classify_from_misc(misc_entry: dict) -> str | None:
    """
    Use disbursement_total / loan_total from outputs_misc.jsonl.
    """
    disb = misc_entry.get("disbursement_total")
    loan = misc_entry.get("loan_total")
    u_disb = misc_entry.get("disbursement_units")
    u_loan = misc_entry.get("loan_units")

    def _to_decimal(x):
        if x is None:
            return None
        try:
            return Decimal(str(x))
        except Exception:
            return None

    disb_dec = _to_decimal(disb)
    loan_dec = _to_decimal(loan)

    has_disb = disb_dec is not None and disb_dec > 0
    has_loan = loan_dec is not None and loan_dec > 0

    if has_loan and not has_disb:
        return "loan"
    if has_disb and not has_loan:
        return "disbursement"

    if has_disb and has_loan:
        u_disb = (u_disb or "").strip()
        u_loan = (u_loan or "").strip()
        if u_disb and u_disb == u_loan:
            if loan_dec > disb_dec:
                return "loan"
            if disb_dec > loan_dec:
                return "disbursement"
            return "loan"  # tie-breaker
        return None

    return None


def classify_from_txns(txns: dict[str, Decimal]) -> str | None:
    """
    Use transaction types to infer loan vs disbursement.

      - If any of {5,6,10} > 0 -> loan.
      - Else if any of {3,4,7} > 0 -> disbursement.
    """
    loanish_codes = {"5", "6", "10"}
    disb_codes = {"3", "4", "7"}

    for c in loanish_codes:
        if txns.get(c, Decimal("0")) > 0:
            return "loan"

    for c in disb_codes:
        if txns.get(c, Decimal("0")) > 0:
            return "disbursement"

    return None


# ---- Public helper: returns a DataFrame ----


def load_loan_or_disbursement() -> pd.DataFrame:
    """
    Classify each activity in merged_overall_rankings.jsonl as
    'loan' or 'disbursement' where possible.

    Returns a DataFrame indexed by activity_id with columns:
        - finance_label: 'loan' / 'disbursement' / None
        - finance_is_loan: 1.0 (loan), 0.0 (disbursement), NaN (unknown)
        - finance_class_method: 'misc' / 'finance_type' / 'transactions' / None
    """
    merged_ids = load_merged_ids(MERGED_RANKINGS)
    misc_by_aid = load_outputs_misc(OUTPUTS_MISC)
    finance_type_by_aid = load_default_finance_types(DEFAULT_FINANCE_TYPES_CSV)
    txns_by_aid = load_txn_summary(TXN_SUMMARY)

    records = []

    for aid in merged_ids:
        label = None
        method = None

        misc_entry = misc_by_aid.get(aid)
        if misc_entry:
            label = classify_from_misc(misc_entry)
            if label:
                method = "misc"

        if label is None:
            ft_code = finance_type_by_aid.get(aid)
            if ft_code:
                label = classify_from_finance_type(ft_code)
                if label:
                    method = "finance_type"

        if label is None:
            tx = txns_by_aid.get(aid)
            if tx:
                label = classify_from_txns(tx)
                if label:
                    method = "transactions"

        if label == "loan":
            is_loan = 1.0
        elif label == "disbursement":
            is_loan = 0.0
        else:
            is_loan = float("nan")

        records.append(
            {
                "activity_id": aid,
                "finance_label": label,
                "finance_is_loan": is_loan,
                "finance_class_method": method,
            }
        )

    df = pd.DataFrame.from_records(records).set_index("activity_id").sort_index()
    return df


# ---- Optional CLI summary ----


def main() -> None:
    df = load_loan_or_disbursement()

    total = len(df)
    known = df["finance_label"].notna().sum()
    n_loan = (df["finance_label"] == "loan").sum()
    n_disb = (df["finance_label"] == "disbursement").sum()
    unknown = total - known

    print("=== Loan vs Disbursement classification (merged_overall_rankings set) ===")
    print(f"Total unique activity_ids: {total}")
    print(f"Categorised (any method): {known}")
    print(f"  Loan:         {n_loan}")
    print(f"  Disbursement: {n_disb}")
    print(f"Uncategorised:  {unknown}")


if __name__ == "__main__":
    main()
