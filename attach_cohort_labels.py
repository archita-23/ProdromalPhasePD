
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

features_file = BASE_DIR / "ppmi_ae_features.csv"
cohort_file = BASE_DIR / "participant_cohort.csv"
output_file = BASE_DIR / "ppmi_ae_features_labeled.csv"

print("Loading files...")

features = pd.read_csv(features_file, dtype={"PATNO": str})
cohorts = pd.read_csv(cohort_file, dtype={"PATNO": str})

# Clean column names
features.columns = features.columns.str.strip().str.upper()
cohorts.columns = cohorts.columns.str.strip().str.upper()

# Check required columns
required = {"PATNO", "COHORT"}

if not required.issubset(cohorts.columns):
    raise ValueError("Cohort file must contain PATNO and COHORT columns.")

# Keep participant ID and cohort
cohorts = cohorts[["PATNO", "COHORT"]].copy()

# Clean values
cohorts["PATNO"] = cohorts["PATNO"].str.strip()
cohorts["COHORT"] = pd.to_numeric(
    cohorts["COHORT"], errors="coerce"
)

# Remove invalid rows
cohorts = cohorts.dropna(subset=["PATNO", "COHORT"])

# Check for conflicting cohort assignments
conflicts = cohorts.groupby("PATNO")["COHORT"].nunique()
conflicts = conflicts[conflicts > 1]

if not conflicts.empty:
    print("Conflicting cohort assignments found:")
    print(conflicts)
    raise ValueError("Resolve conflicting participant labels first.")

cohorts = cohorts.drop_duplicates(subset="PATNO")

# Map PPMI cohort codes
cohort_mapping = {
    1: "PD",
    2: "HC",
    3: "SWEDD",
    4: "PRODROMAL"
}

cohorts["COHORT_LABEL"] = cohorts["COHORT"].map(cohort_mapping)

# Merge participant labels into visit-level dataset
merged = features.merge(
    cohorts[["PATNO", "COHORT_LABEL"]],
    on="PATNO",
    how="left",
    validate="many_to_one"
)

merged["cohort"] = merged["COHORT_LABEL"].fillna("UNKNOWN")
merged = merged.drop(columns=["COHORT_LABEL"])

print("\nCohort distribution:")
print(merged["cohort"].value_counts(dropna=False))

print("\nUnique participants by cohort:")
print(
    merged.groupby("cohort")["PATNO"].nunique()
)

print("\nUnmatched participants:")
unmatched = merged.loc[merged["cohort"] == "UNKNOWN", "PATNO"].nunique()
print(unmatched)

print("\nTotal visits:", len(merged))
print("Total participants:", merged["PATNO"].nunique())

merged.to_csv(output_file, index=False)

print("\nSaved:", output_file)
