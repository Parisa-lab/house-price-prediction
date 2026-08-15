from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# =========================
# 1. CONFIGURATION
# =========================

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/model.pkl")
PLOT_PATH = Path("feature_importance.png")

TARGET = "SalePrice"

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]


# =========================
# 2. LOAD DATA
# =========================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "data/train.csv was not found. "
        "Please download the Kaggle House Prices dataset "
        "and place train.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)


# =========================
# 3. CHECK REQUIRED COLUMNS
# =========================

required_columns = FEATURES + [TARGET]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing "
        f"from the dataset: {missing_columns}"
    )


# =========================
# 4. SELECT FEATURES / TARGET
# =========================

X = df[FEATURES]
y = df[TARGET]


# =========================
# 5. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# =========================
# 6. PREPROCESSING
# =========================

preprocessor = SimpleImputer(
    strategy="median"
)


# =========================
# 7. MODEL
# =========================

model = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )
        ),
    ]
)


# =========================
# 8. TRAIN MODEL
# =========================

print("\nTraining Random Forest model...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# =========================
# 9. MAKE PREDICTIONS
# =========================

predictions = model.predict(
    X_test
)


# =========================
# 10. EVALUATE MODEL
# =========================

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 40)
print("MODEL PERFORMANCE")
print("=" * 40)

print(
    f"RMSE: ${rmse:,.2f}"
)

print(
    f"R² Score: {r2:.4f}"
)


# =========================
# 11. FEATURE IMPORTANCE
# =========================

regressor = model.named_steps[
    "regressor"
]

importances = regressor.feature_importances_


feature_importance = (
    pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": importances,
        }
    )
    .sort_values(
        by="Importance",
        ascending=False,
    )
)


print("\n" + "=" * 40)
print("FEATURE IMPORTANCE")
print("=" * 40)

print(
    feature_importance.to_string(
        index=False
    )
)


# =========================
# 12. FEATURE IMPORTANCE PLOT
# =========================

plt.figure(
    figsize=(10, 6)
)

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
)

plt.title(
    "Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()


plt.savefig(
    PLOT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print(
    f"\nFeature importance plot saved to: "
    f"{PLOT_PATH}"
)


# =========================
# 13. SAVE MODEL
# =========================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    MODEL_PATH,
    "wb"
) as file:
    pickle.dump(
        model,
        file
    )


print(
    f"Model saved to: {MODEL_PATH}"
)

print("\nTraining pipeline completed successfully.")