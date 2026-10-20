
"""Train and evaluate the NumPy neural network for house-price prediction.

The Random Forest and neural network share the same held-out test set.
Feature and target scaling are fitted on the neural network's training
partition only to prevent data leakage.

The trained model and its evaluation metrics are saved separately.
The metrics file is linked to the exact model artifact by SHA-256.

Run from the repository root:
    python train_neural_network.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from model_metrics import save_model_metrics
from neural_network import (
    FEATURES,
    HIDDEN_UNITS,
    RANDOM_STATE,
    TARGET,
    NeuralNetworkRegressor,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/neural_network.npz")
METRICS_PATH = Path("models/neural_network_metrics.json")

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Ensure data/train.csv exists in the repository."
    )

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {missing_columns}"
    )

data = df[required_columns].copy()

for column in required_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

if data.isna().any().any():
    missing_counts = data.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].to_dict()
    raise ValueError(
        "Selected features and target must not contain missing or "
        f"non-numeric values. Affected columns: {missing_counts}"
    )

if not np.isfinite(data.to_numpy(dtype=np.float64)).all():
    raise ValueError("Dataset contains non-finite values.")

if (data[TARGET] <= 0).any():
    raise ValueError("SalePrice must be positive for every row.")

X = data[FEATURES]
y = data[TARGET]

print(f"Dataset loaded: {len(data)} rows")
print(f"Features: {', '.join(FEATURES)}")
print(f"Target: {TARGET}")


# ============================================================
# 3. CREATE THE SHARED TEST SPLIT
# ============================================================

# Match train.py so both models use the same held-out records,
# provided the dataset row order has not changed.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

# Create the validation set from training data only.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
)

print(f"\nTraining samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples:    {len(X_test)}")


# ============================================================
# 4. SCALE FEATURES AND TARGET
# ============================================================

# Save scaling parameters so inference uses the same transformations.
x_mean = X_train.mean(axis=0).to_numpy(dtype=np.float64)
x_scale = X_train.std(axis=0, ddof=0).to_numpy(dtype=np.float64).copy()

# Avoid division by zero for constant features.
x_scale[x_scale == 0] = 1.0

y_mean = np.array([y_train.mean()], dtype=np.float64)
y_scale = np.array([y_train.std(ddof=0)], dtype=np.float64)

if not np.isfinite(x_mean).all() or not np.isfinite(x_scale).all():
    raise ValueError("Feature-scaling parameters are not finite.")

if not np.isfinite(y_mean).all() or not np.isfinite(y_scale).all():
    raise ValueError("Target-scaling parameters are not finite.")

if y_scale[0] <= 0:
    raise ValueError("Target values must have a positive standard deviation.")

X_train_scaled = (
    X_train.to_numpy(dtype=np.float64) - x_mean
) / x_scale

X_val_scaled = (
    X_val.to_numpy(dtype=np.float64) - x_mean
) / x_scale

X_test_scaled = (
    X_test.to_numpy(dtype=np.float64) - x_mean
) / x_scale

y_train_scaled = (
    y_train.to_numpy(dtype=np.float64).reshape(-1, 1) - y_mean
) / y_scale

y_val_scaled = (
    y_val.to_numpy(dtype=np.float64).reshape(-1, 1) - y_mean
) / y_scale


# ============================================================
# 5. TRAIN THE NEURAL NETWORK
# ============================================================

model = NeuralNetworkRegressor(
    input_size=len(FEATURES),
    hidden_units=HIDDEN_UNITS,
    seed=RANDOM_STATE,
)

print("\nTraining neural network...")

model.fit(
    X_train_scaled,
    y_train_scaled,
    X_val_scaled,
    y_val_scaled,
)


# ============================================================
# 6. EVALUATE ON THE SHARED TEST SET
# ============================================================

scaled_predictions = model.predict(X_test_scaled)

predictions = (
    scaled_predictions * y_scale + y_mean
).reshape(-1)

y_test_values = y_test.to_numpy(dtype=np.float64)

if not np.isfinite(predictions).all():
    raise ValueError(
        "The neural network produced non-finite test predictions."
    )

mae = float(mean_absolute_error(y_test_values, predictions))
rmse = float(np.sqrt(mean_squared_error(y_test_values, predictions)))
r2 = float(r2_score(y_test_values, predictions))

# Compare against a baseline that predicts the training-target mean.
baseline = DummyRegressor(strategy="mean")
baseline.fit(
    np.zeros((len(y_train), 1), dtype=np.float64),
    y_train.to_numpy(dtype=np.float64),
)
baseline_predictions = baseline.predict(
    np.zeros((len(y_test), 1), dtype=np.float64)
)

baseline_mae = mean_absolute_error(y_test_values, baseline_predictions)
baseline_rmse = float(
    np.sqrt(mean_squared_error(y_test_values, baseline_predictions))
)
baseline_r2 = r2_score(y_test_values, baseline_predictions)

print("\n" + "=" * 48)
print("NEURAL NETWORK — TEST PERFORMANCE")
print("=" * 48)
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:   {r2:.4f}")

print("\n" + "=" * 48)
print("MEAN BASELINE — SAME TEST SET")
print("=" * 48)
print(f"MAE:  ${baseline_mae:,.2f}")
print(f"RMSE: ${baseline_rmse:,.2f}")
print(f"R²:   {baseline_r2:.4f}")


# ============================================================
# 7. SAVE MODEL AND SCALING PARAMETERS
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

np.savez(
    MODEL_PATH,
    W1=model.W1,
    b1=model.b1,
    W2=model.W2,
    b2=model.b2,
    W3=model.W3,
    b3=model.b3,
    x_mean=x_mean,
    x_scale=x_scale,
    y_mean=y_mean,
    y_scale=y_scale,
    features=np.asarray(FEATURES),
    target=np.asarray(TARGET),
    hidden_units=np.asarray(HIDDEN_UNITS, dtype=np.int64),
)

print(f"\nNeural-network model saved to: {MODEL_PATH}")


# ============================================================
# 8. SAVE EVALUATION METRICS
# ============================================================

save_model_metrics(
    METRICS_PATH,
    MODEL_PATH,
    model_name="neural_network",
    target=TARGET,
    features=list(FEATURES),
    metrics={
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    },
    test_samples=len(X_test),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"Evaluation metrics saved to: {METRICS_PATH}")
print("\nNeural-network training completed successfully.")
