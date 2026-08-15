import streamlit as st
import pandas as pd
import pickle

# Load trained model (you should save it after training)
model = pickle.load(open("model.pkl", "rb"))

st.title("House Price Prediction")

# Example inputs (customize based on your dataset)
area = st.number_input("Area (sq ft)")
bedrooms = st.number_input("Number of Bedrooms")
bathrooms = st.number_input("Number of Bathrooms")

if st.button("Predict"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms]],
                              columns=["GrLivArea", "BedroomAbvGr", "FullBath"])

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ${prediction:,.2f}")