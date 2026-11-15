
import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent

features = pd.read_csv(base / "ppmi_ae_features.csv")
cohorts = pd.read_csv(base / "participant_cohort.csv")

def clean_id(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype("string")
    )

features["PATNO"] = clean_id(features["PATNO"])
cohorts["PATNO"] = clean_id(cohorts["PATNO"])

feature_ids = set(features["PATNO"].dropna().unique())
cohort_ids = set(cohorts["PATNO"].dropna().unique())

matched = feature_ids & cohort_ids

print("FEATURE FILE")
print("Unique participants:", len(feature_ids))

print("\nCOHORT FILE")
print("Rows:", len(cohorts))
print("Unique participants:", len(cohort_ids))
print("Cohort counts:")
print(cohorts["COHORT"].value_counts(dropna=False))

print("\nMATCHING")
print("Matched participants:", len(matched))
print("Unmatched feature participants:", len(feature_ids - cohort_ids))

print("\nSample feature IDs:", sorted(feature_ids)[:15])
print("Sample cohort IDs:", sorted(cohort_ids)[:15])

print("\nSample unmatched IDs:", sorted(feature_ids - cohort_ids)[:20])
