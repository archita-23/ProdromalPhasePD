
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

from ae_common import (
    FEATURE_COLUMNS,
    make_scaler,
    build_autoencoder,
    reconstruction_errors,
    FEATURES_CSV,
    MODEL_PATH,
    SCALER_PATH,
    CONFIG_PATH,
    HISTORY_CSV,
    BASE_DIR
)

from tensorflow import keras


# -----------------------------
# Configuration
# -----------------------------

EPOCHS = 100
BATCH_SIZE = 64
THRESHOLD_PERCENTILE = 95

print("Loading PPMI dataset...")

df = pd.read_csv(FEATURES_CSV)

required = ["PATNO"] + FEATURE_COLUMNS

for column in required:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

df = df.dropna(subset=required).copy()

X = df[FEATURE_COLUMNS].astype(float).values
groups = df["PATNO"].values

print("Total visits:", len(df))
print("Unique participants:", df["PATNO"].nunique())


# -----------------------------
# Participant-level split
# -----------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, groups=groups)
)

X_train = X[train_idx]
X_val = X[val_idx]

print("\nTraining visits:", len(X_train))
print("Validation visits:", len(X_val))

print(
    "Training participants:",
    len(np.unique(groups[train_idx]))
)

print(
    "Validation participants:",
    len(np.unique(groups[val_idx]))
)


# -----------------------------
# Scaling
# -----------------------------

scaler = make_scaler()

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)


# -----------------------------
# Build Autoencoder
# -----------------------------

print("\nBuilding Autoencoder...")

model = build_autoencoder(
    hidden=8,
    bottleneck=3,
    seed=42
)

model.summary()


# -----------------------------
# Train
# -----------------------------

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("\nTraining Autoencoder...")

history = model.fit(
    X_train_scaled,
    X_train_scaled,
    validation_data=(X_val_scaled, X_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)


# -----------------------------
# Reconstruction errors
# -----------------------------

print("\nCalculating reconstruction errors...")

train_errors, _ = reconstruction_errors(
    model,
    X_train_scaled
)

val_errors, _ = reconstruction_errors(
    model,
    X_val_scaled
)

threshold = float(
    np.percentile(val_errors, THRESHOLD_PERCENTILE)
)

print("\nValidation results:")
print("Mean reconstruction error:", np.mean(val_errors))
print("Threshold:", threshold)


# -----------------------------
# Save model and artifacts
# -----------------------------

model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

config = {
    "feature_columns": FEATURE_COLUMNS,
    "threshold": threshold,
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "training_type": "unsupervised",
    "training_visits": len(X_train),
    "validation_visits": len(X_val)
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=4)

pd.DataFrame(history.history).to_csv(
    HISTORY_CSV,
    index=False
)


# -----------------------------
# Score all PPMI visits
# -----------------------------

all_scaled = scaler.transform(X)

all_errors, feature_errors = reconstruction_errors(
    model,
    all_scaled
)

results = df[["PATNO", "EVENT_ID"]].copy()

results["reconstruction_error"] = all_errors

results["is_anomaly"] = (
    all_errors > threshold
).astype(int)

for i, feature in enumerate(FEATURE_COLUMNS):
    results[f"{feature}_error"] = feature_errors[:, i]

results.to_csv(
    BASE_DIR / "autoencoder_predictions.csv",
    index=False
)


# -----------------------------
# Final report
# -----------------------------

print("\n" + "=" * 45)
print("AUTOENCODER TRAINING COMPLETED")
print("=" * 45)

print("Final training loss:", history.history["loss"][-1])
print("Final validation loss:", history.history["val_loss"][-1])

print("Threshold:", threshold)

print(
    "Anomalies detected:",
    results["is_anomaly"].sum()
)

print(
    "Anomaly percentage:",
    round(results["is_anomaly"].mean() * 100, 2),
    "%"
)

print("\nSaved files:")
print("1. autoencoder_model.keras")
print("2. autoencoder_scaler.pkl")
print("3. autoencoder_config.json")
print("4. autoencoder_training_history.csv")
print("5. autoencoder_predictions.csv")
 