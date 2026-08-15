from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# =========================
# LOAD MODEL
# =========================

MODEL_PATH = Path("models/model.pkl")


@st.cache_resource
def load_model():

    with open(
        MODEL_PATH,
        "rb"
    ) as file:
        return pickle.load(file)


if not MODEL_PATH.exists():

    st.error(
        "The trained model was not found."
    )

    st.stop()


model = load_model()


# =========================
# HEADER
# =========================

st.title(
    "🏠 House Price Predictor"
)

st.write(
    "Estimate a house's sale price "
    "using a Random Forest regression model."
)


# =========================
# INPUTS
# =========================

st.subheader(
    "Property Details"
)

col1, col2 = st.columns(2)


with col1:

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "Overall material and finish "
            "quality of the house."
        )
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=10_000,
        value=1_500,
        step=50
    )

    garage_cars = st.number_input(
        "Garage Capacity",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )


with col2:

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5_000,
        value=800,
        step=50
    )

    full_bath = st.number_input(
        "Full Bathrooms",
        min_value=0,
        max_value=5,
        value=2,
        step=1
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1872,
        max_value=2025,
        value=2000,
        step=1
    )


# =========================
# CREATE INPUT DATA
# =========================

input_data = pd.DataFrame(
    [
        {
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars,
            "TotalBsmtSF": total_bsmt_sf,
            "FullBath": full_bath,
            "YearBuilt": year_built
        }
    ]
)


# =========================
# PREDICTION
# =========================

st.divider()


if st.button(
    "Predict House Price",
    use_container_width=True
):

    prediction = model.predict(
        input_data
    )[0]

    st.success(
        f"Estimated Sale Price: "
        f"${prediction:,.0f}"
    )

    st.caption(
        "Prediction generated using a "
        "Random Forest regression model."
    )


# =========================
# DISCLAIMER
# =========================

st.divider()

st.caption(
    "For educational purposes only. "
    "This prediction is not a professional "
    "real-estate valuation."
)