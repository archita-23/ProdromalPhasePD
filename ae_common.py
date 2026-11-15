from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# Keep this order consistent across preparation, training and prediction.
FEATURE_COLUMNS = [
    "cognition",
    "hallucinations",
    "mood",
    "apathy",
    "tremor",
]

FEATURE_MIN = 0.0
FEATURE_MAX = 4.0

# Autoencoder artifacts only
FEATURES_CSV = BASE_DIR / "ppmi_ae_features.csv"
SPLITS_CSV = BASE_DIR / "ae_patient_splits.csv"
MODEL_PATH = BASE_DIR / "autoencoder_model.keras"
SCALER_PATH = BASE_DIR / "autoencoder_scaler.pkl"
CONFIG_PATH = BASE_DIR / "autoencoder_config.json"
HISTORY_CSV = BASE_DIR / "autoencoder_training_history.csv"

# Existing Isolation Forest files: read-only
IF_MODEL_PATH = BASE_DIR / "isolation_forest_model.pkl"
IF_SCALER_PATH = BASE_DIR / "feature_scaler.pkl"


def make_scaler():
    """Scale valid MDS-UPDRS scores from 0–4 into 0–1."""
    from sklearn.preprocessing import MinMaxScaler

    n = len(FEATURE_COLUMNS)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(np.array([[FEATURE_MIN] * n, [FEATURE_MAX] * n]))
    return scaler


def build_autoencoder(hidden=8, bottleneck=3, seed=42):
    """Build a small dense Autoencoder."""
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(seed)

    n = len(FEATURE_COLUMNS)

    inputs = keras.Input(shape=(n,), name="features")
    x = layers.Dense(hidden, activation="relu", name="encoder_hidden")(inputs)
    code = layers.Dense(bottleneck, activation="linear", name="bottleneck")(x)
    x = layers.Dense(hidden, activation="relu", name="decoder_hidden")(code)
    outputs = layers.Dense(n, activation="sigmoid", name="reconstruction")(x)

    model = keras.Model(inputs, outputs, name="dense_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


def reconstruction_errors(model, scaled_values):
    """Return row-level and feature-level reconstruction errors."""
    rebuilt = model.predict(scaled_values, verbose=0)
    feature_error = (scaled_values - rebuilt) ** 2
    return feature_error.mean(axis=1), feature_error