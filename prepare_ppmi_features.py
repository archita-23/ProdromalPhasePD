import argparse
import sys
from pathlib import Path

import pandas as pd

from ae_common import FEATURE_COLUMNS, FEATURES_CSV

KEY = ["PATNO", "EVENT_ID"]

PART1_COLUMNS = [
    "NP1COG",
    "NP1HALL",
    "NP1DPRS",
    "NP1ANXS",
    "NP1APAT",
]

TREMOR_COLUMNS = [
    "NP3PTRMR",
    "NP3PTRML",
    "NP3KTRMR",
    "NP3KTRML",
    "NP3RTARU",
    "NP3RTALU",
    "NP3RTARL",
    "NP3RTALL",
    "NP3RTALJ",
]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    return pd.read_csv(path, low_memory=False)


def clean_scores(df, columns):
    """Convert scores to numeric and mark anything outside 0–4 as missing."""
    result = df[columns].apply(pd.to_numeric, errors="coerce")
    return result.where(result.apply(lambda col: col.between(0, 4)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part1", required=True)
    parser.add_argument("--part3", required=True)
    parser.add_argument("--out", default=str(FEATURES_CSV))
    args = parser.parse_args()

    print("Loading PPMI files...")
    part1 = read_csv(args.part1)
    part3 = read_csv(args.part3)

    for name, df in [("Part I", part1), ("Part III", part3)]:
        missing_keys = [col for col in KEY if col not in df.columns]
        if missing_keys:
            sys.exit(f"{name} is missing key columns: {missing_keys}")

    missing_part1 = [c for c in PART1_COLUMNS if c not in part1.columns]
    missing_part3 = [c for c in TREMOR_COLUMNS if c not in part3.columns]

    if missing_part1:
        sys.exit(f"Missing Part I columns: {missing_part1}")
    if missing_part3:
        sys.exit(f"Missing Part III columns: {missing_part3}")

    # ---------------------------------------------------------
    # 1. Part I: one row per participant visit
    # ---------------------------------------------------------
    print("\nProcessing Part I...")

    p1 = part1[KEY + PART1_COLUMNS].copy()
    p1[PART1_COLUMNS] = clean_scores(p1, PART1_COLUMNS)

    # Collapse any repeated Part I records for the same visit.
    p1 = p1.groupby(KEY, as_index=False)[PART1_COLUMNS].mean()

    p1["cognition"] = p1["NP1COG"]
    p1["hallucinations"] = p1["NP1HALL"]
    p1["mood"] = p1[["NP1DPRS", "NP1ANXS"]].mean(axis=1)
    p1["apathy"] = p1["NP1APAT"]

    p1 = p1[KEY + ["cognition", "hallucinations", "mood", "apathy"]]

    # ---------------------------------------------------------
    # 2. Part III: choose one medication-state record per visit
    # ---------------------------------------------------------
    print("\nProcessing Part III...")

    p3 = part3[KEY + ["PDSTATE"] + TREMOR_COLUMNS].copy()
    p3[TREMOR_COLUMNS] = clean_scores(p3, TREMOR_COLUMNS)

    # Count available tremor measurements in each examination.
    p3["available_tremor"] = p3[TREMOR_COLUMNS].notna().sum(axis=1)

    # Prefer OFF, then ON, then unknown medication state.
    # Within the same state, prefer the record with more available scores.
    state_priority = {"OFF": 0, "ON": 1}
    p3["state_priority"] = (
        p3["PDSTATE"].astype(str).str.upper().map(state_priority).fillna(2)
    )

    p3 = p3.sort_values(
        KEY + ["state_priority", "available_tremor"],
        ascending=[True, True, True, False],
    )

    # Keep one examination for each participant visit.
    p3 = p3.drop_duplicates(KEY, keep="first").copy()

    # Maximum valid tremor score across the selected examination.
    p3["tremor"] = p3[TREMOR_COLUMNS].max(axis=1)
    p3 = p3[KEY + ["tremor"]]

    # ---------------------------------------------------------
    # 3. Merge Part I and Part III
    # ---------------------------------------------------------
    print("\nMerging participant visits...")

    merged = p1.merge(p3, on=KEY, how="inner")

    print(f"Part I visits: {len(p1):,}")
    print(f"Part III visits after state selection: {len(p3):,}")
    print(f"Matched visits: {len(merged):,}")
    print(f"Unique participants: {merged['PATNO'].nunique():,}")

    # ---------------------------------------------------------
    # 4. Missing-value report
    # ---------------------------------------------------------
    print("\nMissing values before row filtering:")
    print(merged[FEATURE_COLUMNS].isna().sum().to_string())

    before = len(merged)
    merged = merged.dropna(subset=FEATURE_COLUMNS).copy()

    print(f"\nRemoved visits with missing features: {before - len(merged):,}")
    print(f"Remaining visits: {len(merged):,}")

    # ---------------------------------------------------------
    # 5. Add visit numbers and check duplicates
    # ---------------------------------------------------------
    merged = merged.sort_values(KEY).reset_index(drop=True)
    merged["visit_num"] = merged.groupby("PATNO").cumcount()

    duplicate_count = merged.duplicated(KEY).sum()
    print(f"\nDuplicate participant visits remaining: {duplicate_count}")

    if duplicate_count:
        sys.exit("Duplicate visits remain. Stopping rather than saving ambiguous data.")

    # No cohort labels are assigned here. They must come from a real
    # PPMI participant status/cohort file, not guessed from anomaly scores.
    merged["cohort"] = "UNKNOWN"

    output_columns = (
        KEY + ["visit_num", "cohort"] + FEATURE_COLUMNS
    )

    merged[output_columns].to_csv(args.out, index=False)

    print("\nFeature summary (raw 0–4 scores):")
    print(merged[FEATURE_COLUMNS].describe().round(3).to_string())

    print(f"\nSaved {len(merged):,} visits to {args.out}")
    print("\nFeature order:")
    for i, feature in enumerate(FEATURE_COLUMNS, start=1):
        print(f"{i}. {feature}")

    print(
        "\nIMPORTANT: cohort is UNKNOWN. Obtain genuine participant cohort "
        "labels before running train_autoencoder.py."
    )


if __name__ == "__main__":
    main()