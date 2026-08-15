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

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]

TARGET = "SalePrice"


# =========================
# 2. LOAD DATA
# =========================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please download train.csv and place it inside data/."
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# =========================
# 3. SELECT FEATURES
# =========================

X = df[FEATURES]
y = df[TARGET]


# =========================
# 4. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# =========================
# 5. PREPROCESSING
# =========================

preprocessor = SimpleImputer(strategy="median")


# =========================
# 6. MODEL
# =========================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# =========================
# 7. TRAIN MODEL
# =========================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")


# =========================
# 8. EVALUATE MODEL
# =========================

predictions = model.predict(X_test)

rmse = mean_squared_error(
    y_test,
    predictions,
) ** 0.5

r2 = r2_score(y_test, predictions)

print("\n" + "=" * 40)
print("MODEL PERFORMANCE")
print("=" * 40)

print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")


# =========================
# 9. FEATURE IMPORTANCE
# =========================

regressor = model.named_steps["regressor"]

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

print("\nFeature importance:")
print(feature_importance)


# =========================
# 10. CREATE FEATURE
#     IMPORTANCE PLOT
# =========================

plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
)

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.show()


# =========================
# 11. SAVE MODEL
# =========================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Plot saved to: {PLOT_PATH}")