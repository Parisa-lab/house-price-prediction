from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# 1. CONFIGURATION
# =========================

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/model.pkl")
PLOT_PATH = Path("feature_importance.png")

TARGET = "SalePrice"


# =========================
# 2. LOAD DATA
# =========================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "data/train.csv was not found. "
        "Download the Kaggle House Prices dataset "
        "and place train.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)


# =========================
# 3. REMOVE ID
# =========================

if "Id" in df.columns:
    df = df.drop(columns=["Id"])


# =========================
# 4. FEATURES / TARGET
# =========================

X = df.drop(columns=[TARGET])
y = df[TARGET]


# =========================
# 5. IDENTIFY COLUMN TYPES
# =========================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print(f"Numerical features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# =========================
# 6. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================
# 7. PREPROCESSING
# =========================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# =========================
# 8. MODEL
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
                n_jobs=-1
            )
        )
    ]
)


# =========================
# 9. TRAIN
# =========================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")


# =========================
# 10. EVALUATE
# =========================

predictions = model.predict(X_test)

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

print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")


# =========================
# 11. FEATURE IMPORTANCE
# =========================

regressor = model.named_steps["regressor"]

importances = regressor.feature_importances_

# Get feature names after preprocessing
preprocessing = model.named_steps["preprocessing"]

feature_names = preprocessing.get_feature_names_out()

feature_importance = (
    pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    )
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("\nTop 15 important features:")

print(
    feature_importance.head(15).to_string(
        index=False
    )
)


# =========================
# 12. FEATURE IMPORTANCE PLOT
# =========================

plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Feature Importances"
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nFeature importance plot saved to "
    f"{PLOT_PATH}"
)


# =========================
# 13. SAVE MODEL
# =========================

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    MODEL_PATH,
    "wb"
) as file:
    pickle.dump(model, file)

print(
    f"Model saved to {MODEL_PATH}"
)