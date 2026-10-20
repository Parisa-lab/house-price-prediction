
"""Streamlit app for Random Forest and neural-network house-price prediction."""

from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

from neural_network import FEATURES, load_model as load_neural_network_model
from neural_network import predict_price as predict_neural_network_price


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_FOREST_PATH = Path("models/model.pkl")
NEURAL_NETWORK_PATH = Path("models/neural_network.npz")

FEATURE_LABELS = {
    "OverallQual": "Overall Quality",
    "GrLivArea": "Living Area",
    "GarageCars": "Garage Capacity",
    "TotalBsmtSF": "Basement Area",
    "FullBath": "Full Bathrooms",
    "YearBuilt": "Year Built",
}

# Existing Random Forest metrics. Their test split has not yet been
# verified against the neural network's shared test split.
RANDOM_FOREST_R2 = 0.8889
RANDOM_FOREST_RMSE = 29190.0

# Metrics from the neural-network training run in this repository.
NEURAL_NETWORK_R2 = 0.8678
NEURAL_NETWORK_RMSE = 31844.77


# ============================================================
# 3. MODEL LOADING
# ============================================================

@st.cache_resource
def load_random_forest(model_path: str, modified_time: float):
    """Load the existing serialized Random Forest pipeline."""
    del modified_time  # Included in the cache key to detect file changes.

    with open(model_path, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_neural_network(model_path: str, modified_time: float):
    """Load and validate the saved NumPy neural-network artifact."""
    del modified_time  # Included in the cache key to detect file changes.

    return load_neural_network_model(model_path)


# ============================================================
# 4. VALIDATE RANDOM FOREST ARTIFACT
# ============================================================

random_forest = None
random_forest_error = None

if RANDOM_FOREST_PATH.is_file():
    try:
        random_forest = load_random_forest(
            str(RANDOM_FOREST_PATH),
            RANDOM_FOREST_PATH.stat().st_mtime,
        )

        if not hasattr(random_forest, "named_steps"):
            raise ValueError(
                "The saved Random Forest model is not a pipeline."
            )

        if "regressor" not in random_forest.named_steps:
            raise ValueError(
                "The saved pipeline does not contain the expected regressor."
            )

        rf_regressor = random_forest.named_steps["regressor"]

        if not hasattr(rf_regressor, "feature_importances_"):
            raise ValueError(
                "The Random Forest does not provide feature importance."
            )

        if len(rf_regressor.feature_importances_) != len(FEATURES):
            raise ValueError(
                "The Random Forest feature count does not match "
                "the expected feature set."
            )

    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        random_forest = None
        random_forest_error = str(exc)
else:
    random_forest_error = (
        f"Random Forest model not found: {RANDOM_FOREST_PATH}"
    )


# ============================================================
# 5. VALIDATE NEURAL-NETWORK ARTIFACT
# ============================================================

neural_network = None
neural_network_error = None

if NEURAL_NETWORK_PATH.is_file():
    try:
        neural_network = load_neural_network(
            str(NEURAL_NETWORK_PATH),
            NEURAL_NETWORK_PATH.stat().st_mtime,
        )
    except (OSError, ValueError) as exc:
        neural_network_error = str(exc)
else:
    neural_network_error = (
        f"Neural-network model not found: {NEURAL_NETWORK_PATH}"
    )


# ============================================================
# 6. HEADER AND MODEL SELECTION
# ============================================================

st.title("House Price Predictor")

st.write(
    "Estimate residential sale prices using two machine-learning "
    "approaches trained on the Ames Housing dataset."
)

available_models = []

if random_forest is not None:
    available_models.append("Random Forest")

if neural_network is not None:
    available_models.append("Neural Network")

if not available_models:
    st.error(
        "No valid trained model is available. Check the model files "
        "and training scripts."
    )

    if random_forest_error:
        st.error(f"Random Forest: {random_forest_error}")

    if neural_network_error:
        st.error(f"Neural network: {neural_network_error}")

    st.stop()

selected_model = st.radio(
    "Choose a prediction model",
    options=available_models,
    horizontal=True,
)

if selected_model == "Random Forest":
    st.info(
        "Random Forest uses an ensemble of decision trees to estimate "
        "the sale price."
    )
else:
    st.info(
        "The neural network uses two fully connected hidden layers "
        "with ReLU activation and saved feature/target scaling."
    )


# ============================================================
# 7. PROPERTY INPUTS
# ============================================================

st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5,
        help="Overall material and finish quality of the house.",
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
    ],
    columns=FEATURES,
)


# ============================================================
# 8. PREDICTION
# ============================================================

st.divider()

if st.button("Predict House Price", use_container_width=True):
    try:
        if selected_model == "Random Forest":
            prediction = float(random_forest.predict(input_data)[0])
        else:
            prediction = predict_neural_network_price(
                input_data.to_numpy(dtype=float),
                neural_network,
            )

        if not pd.notna(prediction):
            raise ValueError("The model returned an invalid prediction.")

        st.success(f"Estimated Sale Price: ${prediction:,.0f}")

        st.metric(
            label="Estimated Sale Price",
            value=f"${prediction:,.0f}",
        )

    except (ValueError, TypeError, FloatingPointError) as exc:
        st.error(f"Could not generate a prediction: {exc}")


# ============================================================
# 9. MODEL PERFORMANCE
# ============================================================

st.divider()
st.header("Model Performance")

if selected_model == "Random Forest":
    performance_r2 = RANDOM_FOREST_R2
    performance_rmse = RANDOM_FOREST_RMSE
else:
    performance_r2 = NEURAL_NETWORK_R2
    performance_rmse = NEURAL_NETWORK_RMSE

metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric("R² Score", f"{performance_r2:.4f}")

with metric_col2:
    st.metric("RMSE", f"${performance_rmse:,.2f}")

st.write(
    f"The displayed R² score is **{performance_r2 * 100:.1f}%** "
    "when expressed as a percentage."
)

st.caption(
    "The displayed metrics come from separate recorded training runs. "
    "Do not interpret differences between the models as a controlled "
    "comparison until both are evaluated on the exact same test rows."
)


# ============================================================
# 10. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

if random_forest is not None:
    st.divider()
    st.header("Random Forest Feature Importance")

    feature_importance = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": (
                random_forest.named_steps["regressor"]
                .feature_importances_
            ),
        }
    )

    feature_importance["Feature"] = feature_importance["Feature"].map(
        FEATURE_LABELS
    )

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=True,
    )

    st.bar_chart(feature_importance.set_index("Feature"))

    most_important_feature = feature_importance.iloc[-1]["Feature"]

    st.write(
        f"**{most_important_feature}** has the highest feature "
        "importance in the Random Forest model. Feature importance "
        "describes the model's use of a feature, not causation."
    )


# ============================================================
# 11. ABOUT THE PROJECT
# ============================================================

st.divider()
st.header("About This Project")

st.write(
    """
This project estimates residential sale prices using two models:

- **Random Forest:** an ensemble of decision trees.
- **Neural Network:** a NumPy implementation with two ReLU hidden layers.

Both models use the following six property features:

- Overall Quality
- Living Area
- Garage Capacity
- Basement Area
- Full Bathrooms
- Year Built
"""
)


# ============================================================
# 12. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "For educational and demonstration purposes only. "
    "Predictions are estimates, not professional real-estate valuations."
)
