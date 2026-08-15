# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("data/train.csv")

# Drop ID column
df.drop(columns=["Id"], inplace=True)

# =========================
# 3. TARGET
# =========================
target = "SalePrice"

# =========================
# 4. HANDLE MISSING VALUES
# =========================

# Separate numeric and categorical
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object"]).columns

# Fill numeric with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical with "None"
df[cat_cols] = df[cat_cols].fillna("None")

# =========================
# 5. SPLIT FEATURES / TARGET
# =========================
X = df.drop(columns=[target])
y = df[target]

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. PREPROCESSING PIPELINE
# =========================

# OneHot for categorical
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

# =========================
# 8. MODEL PIPELINE
# =========================
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# =========================
# 9. TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# 10. EVALUATION
# =========================
preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# =========================
# 11. FEATURE IMPORTANCE
# =========================

# Get feature names after encoding
ohe = model.named_steps["preprocessing"].named_transformers_["cat"]
encoded_cat_features = ohe.get_feature_names_out(cat_cols)

all_features = list(encoded_cat_features) + list(num_cols)

importances = model.named_steps["regressor"].feature_importances_

feat_importance = pd.DataFrame({
    "Feature": all_features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Plot top 15 features
plt.figure(figsize=(10,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=feat_importance.head(15)
)
plt.title("Top 15 Important Features")
plt.show()

# =========================
# 12. SAVE MODEL
# =========================
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)