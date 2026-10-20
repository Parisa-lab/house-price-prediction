
"""Train, evaluate, and save the Random Forest house-price model."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_metrics import save_model_metrics


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/random_forest_metrics.json")
PLOT_PATH = Path("feature_importance.png")

TARGET = "SalePrice"
TEST_SIZE = 0.20
RANDOM_STATE = 42

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Download the Kaggle House Prices dataset and place "
        "train.csv inside the data directory."
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

X = df[FEATURES]
y = pd.to_numeric(df[TARGET], errors="coerce")

if y.isna().any() or not (y > 0).all():
    raise ValueError("SalePrice must contain valid positive numeric values.")

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# 4. BUILD AND TRAIN THE MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessing", SimpleImputer(strategy="median")),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

print("\nTraining Random Forest model...")
model.fit(X_train, y_train)
print("Training complete.")


# ============================================================
# 5. EVALUATE THE MODEL
# ============================================================

predictions = model.predict(X_test)

mae = float(mean_absolute_error(y_test, predictions))
rmse = float(mean_squared_error(y_test, predictions) ** 0.5)
r2 = float(r2_score(y_test, predictions))

print("\n" + "=" * 40)
print("RANDOM FOREST — TEST PERFORMANCE")
print("=" * 40)
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

regressor = model.named_steps["regressor"]

feature_importance = (
    pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": regressor.feature_importances_,
        }
    )
    .sort_values("Importance", ascending=False)
)

print("\n" + "=" * 40)
print("FEATURE IMPORTANCE")
print("=" * 40)
print(feature_importance.to_string(index=False))


# ============================================================
# 7. SAVE FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nFeature importance plot saved to: {PLOT_PATH}")


# ============================================================
# 8. SAVE MODEL AND EVALUATION METRICS
# ============================================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

with MODEL_PATH.open("wb") as file:
    pickle.dump(model, file)

print(f"Model saved to: {MODEL_PATH}")

save_model_metrics(
    METRICS_PATH,
    MODEL_PATH,
    model_name="random_forest",
    target=TARGET,
    features=FEATURES,
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
print("\nRandom Forest training completed successfully.")
