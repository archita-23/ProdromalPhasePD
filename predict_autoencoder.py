"""
predict_autoencoder.py - STEP 3: score new check-ins with the trained Autoencoder.

Two ways to use it:

1) From the command line
     python predict_autoencoder.py --values 1 1 2 0 0
        (order: sleep_duration sleep_quality mood fatigue tremor, each 0-4)

     python predict_autoencoder.py --csv my_checkins.csv --out scored.csv
        (the CSV needs the five feature columns)

2) From Python / Streamlit (this is what the app does)
     from predict_autoencoder import load_autoencoder, score_autoencoder
     bundle = load_autoencoder()
     result = score_autoencoder(bundle, [[1, 1, 2, 0, 0]])
     result["error"], result["flag"], result["threshold"]

TensorFlow is imported only inside load_autoencoder(), so simply importing this
file is fast and never crashes an app that does not have TensorFlow installed.
"""

import argparse
import json

import numpy as np
import pandas as pd

from ae_common import (
    CONFIG_PATH, FEATURE_COLUMNS, FEATURE_MAX, FEATURE_MIN, MODEL_PATH, SCALER_PATH,
)


def load_autoencoder(model_path=MODEL_PATH, scaler_path=SCALER_PATH, config_path=CONFIG_PATH):
    """Load the model, its scaler and its saved threshold. Raises a clear error if a file is missing."""

    import joblib
    from tensorflow import keras

    for path in (model_path, scaler_path, config_path):
        if not path.exists():
            raise FileNotFoundError(
                f"'{path.name}' not found. Run train_autoencoder.py first and keep the "
                "Autoencoder files in the same folder as this script."
            )

    config = json.loads(config_path.read_text())
    if config["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Saved model was trained with a different feature order than ae_common.FEATURE_COLUMNS.")

    return {
        "model": keras.models.load_model(model_path, compile=False),
        "scaler": joblib.load(scaler_path),
        "config": config,
        "threshold": float(config["threshold"]),
    }


def score_autoencoder(bundle, values):
    """
    values: one row [s1, s2, m, f, t] or a 2-D array of rows, raw 0-4 scores.

    Returns a dict:
        error             reconstruction error per row (higher = more unusual)
        flag              True where error > threshold
        threshold         the saved threshold
        ratio             error / threshold  (1.0 = exactly at the threshold)
        feature_error     per-feature squared error (which item was hardest to rebuild)
        top_feature       name of the feature with the largest error, per row
    """

    X = np.asarray(values, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} values per row ({FEATURE_COLUMNS}), got {X.shape[1]}.")
    if not np.isfinite(X).all():
        raise ValueError("Input contains missing or non-numeric values.")
    if (X < FEATURE_MIN).any() or (X > FEATURE_MAX).any():
        raise ValueError(f"All values must be between {FEATURE_MIN:g} and {FEATURE_MAX:g}.")

    scaled = bundle["scaler"].transform(X)
    rebuilt = bundle["model"].predict(scaled, verbose=0)
    feature_error = (scaled - rebuilt) ** 2
    error = feature_error.mean(axis=1)
    threshold = bundle["threshold"]

    return {
        "error": error,
        "flag": error > threshold,
        "threshold": threshold,
        "ratio": error / threshold if threshold > 0 else np.full(len(error), np.nan),
        "feature_error": feature_error,
        "top_feature": [FEATURE_COLUMNS[i] for i in feature_error.argmax(axis=1)],
    }


def main():
    parser = argparse.ArgumentParser(description="Score check-ins with the Autoencoder.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
    "--values",
    nargs=5,
    type=float,
    metavar=("COGNITION", "HALLUCINATIONS", "MOOD", "APATHY", "TREMOR"),
)
    group.add_argument("--csv", help="CSV with the five feature columns")
    parser.add_argument("--out", help="Where to save scored rows (only with --csv)")
    args = parser.parse_args()

    bundle = load_autoencoder()

    if args.values:
        r = score_autoencoder(bundle, args.values)
        print(f"reconstruction error : {r['error'][0]:.6f}")
        print(f"threshold            : {r['threshold']:.6f}")
        print(f"error / threshold    : {r['ratio'][0]:.2f}")
        print(f"result               : {'ABOVE threshold (unusual vs healthy pattern)' if r['flag'][0] else 'within the healthy range'}")
        print(f"hardest item to rebuild: {r['top_feature'][0]}")
        return

    data = pd.read_csv(args.csv)
    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}")
    r = score_autoencoder(bundle, data[FEATURE_COLUMNS].to_numpy())
    data["ae_error"] = r["error"]
    data["ae_flag"] = r["flag"].astype(int)
    data["ae_error_over_threshold"] = r["ratio"]
    data["ae_top_feature"] = r["top_feature"]
    if args.out:
        data.to_csv(args.out, index=False)
        print(f"Saved {len(data)} scored rows to {args.out}")
    else:
        print(data.to_string(index=False))


if __name__ == "__main__":
    main()
