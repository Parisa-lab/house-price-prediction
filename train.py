import pickle

import matplotlib.pyplot as plt
import numpy as np
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

DATA_PATH = "data/train.csv"
MODEL_PATH = "models/model.pkl"

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

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]


# =========================
# 3. IDENTIFY COLUMN TYPES
# =========================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# =========================
# 4. PREPROCESSING
# =========================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# =========================
# 5. MODEL
# =========================

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
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
# 6. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# =========================
# 7. TRAIN
# =========================

model.fit(X_train, y_train)


# =========================
# 8. EVALUATE
# =========================

predictions = model.predict(X_test)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("=" * 40)
print("MODEL PERFORMANCE")
print("=" * 40)
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")


# =========================
# 9. FEATURE IMPORTANCE
# =========================

regressor = model.named_steps["regressor"]

importances = regressor.feature_importances_

feature_importance = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Importance": importances,
    }
).sort_values(
    by="Importance",
    ascending=False,
)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature",
)

plt.title("Feature Importance")
plt.tight_layout()
plt.show()


# =========================
# 10. SAVE MODEL
# =========================

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print(f"\nModel saved to: {MODEL_PATH}")
