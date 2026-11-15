"""
compare_models.py - STEP 4: compare the Autoencoder with your existing Isolation Forest.

Run:
  python compare_models.py                       # compare on the held-out TEST patients
  python compare_models.py --db submissions.db   # also score the check-ins stored in your app database

Your Isolation Forest files (isolation_forest_model.pkl, feature_scaler.pkl) are only READ.

What you get
  1. ROC-AUC for HC vs PD, HC vs prodromal, HC vs both  - for the Isolation Forest, the
     Autoencoder, and a trivial "total symptom score" baseline. If a fancy model does not
     beat the baseline, it is not adding much.
  2. How often each model flags each cohort at its own threshold.
  3. Agreement between the two models (both / only one / neither, Cohen's kappa,
     Spearman correlation).
  4. model_comparison.png and model_comparison_results.csv

Direction of the scores (this is a common source of confusion):
  Isolation Forest score_samples : LOWER  = more unusual   (app.py shows this raw value)
  Autoencoder reconstruction err : HIGHER = more unusual
  For AUC we flip the Isolation Forest score (-score) so "higher = more unusual" for all.
"""

import argparse
import pickle
import sqlite3
import warnings

import numpy as np
import pandas as pd

from ae_common import (
    BASE_DIR, FEATURES_CSV, FEATURE_COLUMNS, FEATURE_MAX, IF_MODEL_PATH, IF_SCALER_PATH, SPLITS_CSV,
)
from predict_autoencoder import load_autoencoder, score_autoencoder

warnings.filterwarnings("ignore", category=UserWarning)


def score_both(raw_df, iso, iso_scaler, ae_bundle):
    """Add IF and AE scores/flags to a DataFrame holding the five raw 0-4 features."""

    out = raw_df.copy()
    X = raw_df[FEATURE_COLUMNS].astype(float)

    scaled = iso_scaler.transform(X)
    out["if_score"] = iso.score_samples(scaled)                # lower = more unusual
    out["if_unusualness"] = -out["if_score"]                   # higher = more unusual
    out["if_flag"] = iso.predict(scaled) == -1                 # uses the model's own contamination cut-off

    ae = score_autoencoder(ae_bundle, X.to_numpy())
    out["ae_error"] = ae["error"]
    out["ae_flag"] = ae["flag"]

    out["burden"] = X.sum(axis=1) / (FEATURE_MAX * len(FEATURE_COLUMNS))  # trivial baseline
    return out


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")


def agreement(df):
    from sklearn.metrics import cohen_kappa_score
    both = int((df.if_flag & df.ae_flag).sum())
    only_if = int((df.if_flag & ~df.ae_flag).sum())
    only_ae = int((~df.if_flag & df.ae_flag).sum())
    neither = int((~df.if_flag & ~df.ae_flag).sum())
    kappa = cohen_kappa_score(df.if_flag, df.ae_flag) if df.if_flag.nunique() > 1 and df.ae_flag.nunique() > 1 else float("nan")
    rho = df["if_unusualness"].corr(df["ae_error"], method="spearman")
    return both, only_if, only_ae, neither, kappa, rho


def print_agreement(df, label):
    both, only_if, only_ae, neither, kappa, rho = agreement(df)
    print(f"\nAgreement on {label} ({len(df):,} rows)")
    print(f"  both flag        : {both:>6,}")
    print(f"  only IsoForest   : {only_if:>6,}")
    print(f"  only Autoencoder : {only_ae:>6,}")
    print(f"  neither flags    : {neither:>6,}")
    print(f"  Cohen's kappa    : {kappa:.3f}   (1 = identical decisions, 0 = no better than chance)")
    print(f"  Spearman rho     : {rho:.3f}   (do they RANK rows similarly?)")


def compare_on_ppmi(iso, iso_scaler, ae_bundle):
    if not FEATURES_CSV.exists() or not SPLITS_CSV.exists():
        print("PPMI comparison skipped: run prepare_ppmi_features.py and train_autoencoder.py first.")
        return

    data = pd.read_csv(FEATURES_CSV).merge(pd.read_csv(SPLITS_CSV)[["PATNO", "split"]], on="PATNO", how="inner")
    test = score_both(data[data["split"] == "test"].reset_index(drop=True), iso, iso_scaler, ae_bundle)
    hc = test[test["cohort"] == "HC"]

    print("=" * 70)
    print("COMPARISON ON HELD-OUT TEST PATIENTS")
    print("=" * 70)
    print(test.groupby("cohort").size().rename("test visits").to_string())

    rows = []
    for name, col in (("Isolation Forest", "if_unusualness"), ("Autoencoder", "ae_error"), ("Baseline: total symptom score", "burden")):
        row = {"model": name}
        for label, cohorts in (("HC vs PD", ["PD"]), ("HC vs PRODROMAL", ["PRODROMAL"]), ("HC vs PD+PRODROMAL", ["PD", "PRODROMAL"])):
            pos = test[test["cohort"].isin(cohorts)]
            pair = pd.concat([hc, pos])
            row[label] = auc((pair["cohort"] != "HC").astype(int), pair[col])
        rows.append(row)
    auc_table = pd.DataFrame(rows).set_index("model").round(3)
    print("\nROC-AUC (0.5 = coin flip, 1.0 = perfect separation of healthy vs the other group)")
    print(auc_table.to_string())

    flag_rows = []
    for cohort, group in test.groupby("cohort"):
        flag_rows.append({
            "cohort": cohort,
            "IsoForest flagged": f"{group.if_flag.mean():.1%}",
            "Autoencoder flagged": f"{group.ae_flag.mean():.1%}",
        })
    print("\nShare of visits flagged as unusual (HC row = false-positive rate)")
    print(pd.DataFrame(flag_rows).set_index("cohort").to_string())

    print_agreement(test, "all test visits")
    print_agreement(hc, "healthy-control test visits only")

    auc_table.to_csv(BASE_DIR / "model_comparison_results.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"HC": "#245B52", "PRODROMAL": "#E9B96E", "PD": "#E98C83"}
    rng = np.random.default_rng(0)
    for cohort, group in test.groupby("cohort"):
        jitter = rng.normal(0, 0.002, len(group))
        ax.scatter(group["if_score"], group["ae_error"] + jitter, s=10, alpha=0.5, label=cohort, color=colors.get(cohort))
    ax.axvline(iso.offset_, color="grey", linestyle="--", linewidth=1)
    ax.axhline(ae_bundle["threshold"], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Isolation Forest score (lower = more unusual)")
    ax.set_ylabel("Autoencoder error (higher = more unusual)")
    ax.set_title("Two models on the same test visits"); ax.legend()
    fig.tight_layout(); fig.savefig(BASE_DIR / "model_comparison.png", dpi=150); plt.close(fig)
    print("\nSaved model_comparison_results.csv and model_comparison.png")


def compare_on_database(db_path, iso, iso_scaler, ae_bundle):
    with sqlite3.connect(db_path) as conn:
        db = pd.read_sql_query("SELECT id, user_id, date, " + ", ".join(FEATURE_COLUMNS) + " FROM submissions", conn)
    if db.empty:
        print("\nDatabase has no check-ins yet.")
        return

    scored = score_both(db, iso, iso_scaler, ae_bundle)
    print("\n" + "=" * 70)
    print(f"COMPARISON ON YOUR APP CHECK-INS ({db_path})")
    print("=" * 70)
    print(scored[["id", "user_id"] + FEATURE_COLUMNS + ["if_score", "if_flag", "ae_error", "ae_flag"]]
          .round(4).to_string(index=False))
    print_agreement(scored, "app check-ins")
    different = int((scored["sleep_duration"] != scored["sleep_quality"]).sum())
    print(f"\nCheck-ins where sleep_duration != sleep_quality: {different} of {len(scored)}")
    scored.to_csv(BASE_DIR / "model_comparison_checkins.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Compare Isolation Forest and Autoencoder.")
    parser.add_argument("--db", default=None, help="Optional SQLite database of app check-ins")
    args = parser.parse_args()

    with open(IF_MODEL_PATH, "rb") as f:
        iso = pickle.load(f)
    with open(IF_SCALER_PATH, "rb") as f:
        iso_scaler = pickle.load(f)
    ae_bundle = load_autoencoder()

    compare_on_ppmi(iso, iso_scaler, ae_bundle)
    if args.db:
        compare_on_database(args.db, iso, iso_scaler, ae_bundle)


if __name__ == "__main__":
    main()
