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

FEATURE_IMPORTANCE = {
    "Overall Quality": 0.585865,
    "Living Area": 0.198440,
    "Basement Area": 0.111882,
    "Year Built": 0.058477,
    "Garage Capacity": 0.031743,
    "Full Bathrooms": 0.013594,
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
# 4. HEADER
# =========================

st.title("🏠 House Price Predictor")

st.write(
    "Estimate the sale price of a house using "
    "a machine learning model trained on the "
    "Ames Housing dataset."
)

st.info(
    "The model uses six important property features "
    "to generate an estimated sale price."
)


# =========================
# 5. PROPERTY INPUTS
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
# 6. CREATE INPUT DATA
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
# 7. PREDICTION
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
# 8. MODEL PERFORMANCE
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
# 9. FEATURE IMPORTANCE
# =========================

st.header("Feature Importance")

importance_df = pd.DataFrame(
    {
        "Feature": FEATURE_IMPORTANCE.keys(),
        "Importance": FEATURE_IMPORTANCE.values(),
    }
)

importance_df = importance_df.sort_values(
    "Importance",
    ascending=True,
)

st.bar_chart(
    importance_df.set_index("Feature")
)


st.write(
    "Overall Quality is the most influential "
    "feature in the model, followed by Living Area "
    "and Basement Area."
)


# =========================
# 10. ABOUT THE MODEL
# =========================

st.divider()

st.header("About This Project")

st.write(
    """
This project uses a Random Forest regression model
to estimate residential property sale prices.

The model was trained using six selected features:

- Overall Quality
- Living Area
- Garage Capacity
- Basement Area
- Full Bathrooms
- Year Built

The dataset is based on the Ames Housing dataset
from the Kaggle House Prices competition.
"""
)


# =========================
# 11. DISCLAIMER
# =========================

st.divider()

st.caption(
    "For educational and demonstration purposes only. "
    "The predicted price is not a professional "
    "real-estate valuation."
)