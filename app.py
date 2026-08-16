from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# =========================
# 1. PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)


# =========================
# 2. MODEL CONFIGURATION
# =========================

MODEL_PATH = Path("models/model.pkl")

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]

FEATURE_LABELS = {
    "OverallQual": "Overall Quality",
    "GrLivArea": "Living Area",
    "GarageCars": "Garage Capacity",
    "TotalBsmtSF": "Basement Area",
    "FullBath": "Full Bathrooms",
    "YearBuilt": "Year Built",
}

R2_SCORE = 0.8889
RMSE = 29190


# =========================
# 3. LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


if not MODEL_PATH.exists():
    st.error(
        "The trained model could not be found. "
        "Please make sure models/model.pkl exists."
    )
    st.stop()


model = load_model()


# =========================
# 4. VALIDATE MODEL
# =========================

if "regressor" not in model.named_steps:
    st.error(
        "The saved model does not contain the expected "
        "Random Forest regressor."
    )
    st.stop()


regressor = model.named_steps["regressor"]

if not hasattr(regressor, "feature_importances_"):
    st.error(
        "The saved model does not provide feature importance."
    )
    st.stop()


if len(regressor.feature_importances_) != len(FEATURES):
    st.error(
        "The number of model features does not match "
        "the expected feature set."
    )
    st.stop()


# =========================
# 5. EXTRACT FEATURE IMPORTANCE
# =========================

feature_importance = pd.DataFrame(
    {
        "Feature": FEATURES,
        "Importance": regressor.feature_importances_,
    }
)

feature_importance["Feature"] = (
    feature_importance["Feature"]
    .map(FEATURE_LABELS)
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=True,
)


# =========================
# 6. HEADER
# =========================

st.title("🏠 House Price Predictor")

st.write(
    "Estimate the sale price of a house using "
    "a Random Forest regression model trained "
    "on the Ames Housing dataset."
)

st.info(
    "The model uses six selected property features "
    "to generate an estimated sale price."
)


# =========================
# 7. PROPERTY INPUTS
# =========================

st.header("Property Details")

col1, col2 = st.columns(2)


with col1:

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "Overall material and finish quality "
            "of the house."
        ),
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=10000,
        value=1500,
        step=50,
    )

    garage_cars = st.number_input(
        "Garage Capacity",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
    )


with col2:

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=800,
        step=50,
    )

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1872,
        max_value=2025,
        value=2000,
        step=1,
    )


# =========================
# 8. CREATE INPUT DATA
# =========================

input_data = pd.DataFrame(
    [
        {
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars,
            "TotalBsmtSF": total_bsmt_sf,
            "FullBath": full_bath,
            "YearBuilt": year_built,
        }
    ]
)


# =========================
# 9. PREDICTION
# =========================

st.divider()

if st.button(
    "Predict House Price",
    use_container_width=True,
):

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Sale Price: ${prediction:,.0f}"
    )

    st.metric(
        label="Estimated Sale Price",
        value=f"${prediction:,.0f}",
    )


# =========================
# 10. MODEL PERFORMANCE
# =========================

st.divider()

st.header("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "R² Score",
        f"{R2_SCORE:.4f}",
    )

with col2:
    st.metric(
        "RMSE",
        f"${RMSE:,.0f}",
    )

st.write(
    f"The model explains approximately "
    f"**{R2_SCORE * 100:.1f}%** of the variation "
    f"in house sale prices on the test set."
)


# =========================
# 11. FEATURE IMPORTANCE
# =========================

st.header("Feature Importance")

st.bar_chart(
    feature_importance.set_index("Feature")
)

most_important_feature = feature_importance.iloc[-1]["Feature"]

st.write(
    f"**{most_important_feature}** is the most influential "
    "feature in the trained Random Forest model."
)


# =========================
# 12. ABOUT THE MODEL
# =========================

st.divider()

st.header("About This Project")

st.write(
    """
This project uses a Random Forest regression model
to estimate residential property sale prices.

The model uses six selected features:

- Overall Quality
- Living Area
- Garage Capacity
- Basement Area
- Full Bathrooms
- Year Built
"""
)


# =========================
# 13. DISCLAIMER
# =========================

st.divider()

st.caption(
    "For educational and demonstration purposes only. "
    "The predicted price is not a professional "
    "real-estate valuation."
)