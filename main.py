# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# =========================
# 2. LOAD DATA
# =========================
df = pd.read_csv("data/train.csv")

print(df.head())
print(df.info())
print(df.describe())

# =========================
# 3. DATA CLEANING
# =========================

# Handle missing values
df = df.dropna(axis=1, thresh=0.7 * len(df))  # drop columns with too many NaNs
df = df.fillna(df.median(numeric_only=True))

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# =========================
# 4. FEATURE SELECTION
# =========================
target = 'SalePrice'

X = df.drop(columns=[target])
y = df[target]

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 6. MODEL TRAINING
# =========================

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Random Forest (better)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# =========================
# 7. EVALUATION
# =========================

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return rmse, r2

lr_rmse, lr_r2 = evaluate(lr, X_test, y_test)
rf_rmse, rf_r2 = evaluate(rf, X_test, y_test)

print("Linear Regression RMSE:", lr_rmse, "R2:", lr_r2)
print("Random Forest RMSE:", rf_rmse, "R2:", rf_r2)

# =========================
# 8. VISUALIZATION
# =========================

plt.figure(figsize=(10,6))
sns.histplot(y, kde=True)
plt.title("Sale Price Distribution")
plt.show()

# Correlation heatmap (top features)
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()





import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(rf, f)