import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))

# =========================
# APP TITLE
# =========================
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction")
st.markdown("Enter property details to estimate the house price.")

# =========================
# USER INPUTS
# =========================

st.subheader("Property Features")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 1)

with col2:
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 800)
    full_bath = st.slider("Full Bathrooms", 0, 4, 1)
    year_built = st.number_input("Year Built", 1900, 2025, 2000)

# =========================
# CREATE INPUT DATAFRAME
# =========================
input_data = pd.DataFrame({
    "OverallQual": [overall_qual],
    "GrLivArea": [gr_liv_area],
    "GarageCars": [garage_cars],
    "TotalBsmtSF": [total_bsmt_sf],
    "FullBath": [full_bath],
    "YearBuilt": [year_built]
})

# =========================
# PREDICTION
# =========================
if st.button("Predict Price"):

    try:
        prediction = model.predict(input_data)[0]

        st.success(f"💰 Estimated House Price: ${prediction:,.0f}")

    except Exception as e:
        st.error("⚠️ Error in prediction. Make sure model matches input features.")