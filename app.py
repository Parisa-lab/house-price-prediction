import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD TRAINED PIPELINE MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction App")
st.write("Enter property details to estimate the house price using a trained ML model.")

# =========================
# INPUT SECTION
# =========================
st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
    garage_cars = st.slider("Garage Cars", 0, 4, 1)

with col2:
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=5000, value=800)
    full_bath = st.slider("Full Bathrooms", 0, 4, 1)
    year_built = st.number_input("Year Built", 1900, 2025, 2000)

# =========================
# CREATE INPUT DATAFRAME
# IMPORTANT: column names MUST match training data
# =========================
input_df = pd.DataFrame([{
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "GarageCars": garage_cars,
    "TotalBsmtSF": total_bsmt_sf,
    "FullBath": full_bath,
    "YearBuilt": year_built
}])

# =========================
# PREDICTION
# =========================
st.divider()

if st.button("Predict House Price"):

    try:
        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated Price: ${prediction:,.0f}")

        st.info("Prediction is generated using a trained Machine Learning model (Random Forest Pipeline).")

    except Exception as e:
        st.error("Prediction failed. Please check if model and input features match.")