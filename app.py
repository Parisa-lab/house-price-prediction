
"""Streamlit app for Random Forest and neural-network house-price prediction."""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from model_metrics import load_model_metrics
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
RANDOM_FOREST_METRICS_PATH = Path("models/random_forest_metrics.json")

NEURAL_NETWORK_PATH = Path("models/neural_network.npz")
NEURAL_NETWORK_METRICS_PATH = Path("models/neural_network_metrics.json")

FEATURE_LABELS = {
    "OverallQual": "Overall Quality",
    "GrLivArea": "Living Area",
    "GarageCars": "Garage Capacity",
    "TotalBsmtSF": "Basement Area",
    "FullBath": "Full Bathrooms",
    "YearBuilt": "Year Built",
}


# ============================================================
# 3. MODEL LOADING
# ============================================================

@st.cache_resource
def load_random_forest(model_path: str, modified_time: float):
    """Load the serialized Random Forest pipeline."""
    del modified_time  # Included in the cache key to detect file changes.

    with open(model_path, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_neural_network(model_path: str, modified_time: float):
    """Load and validate the saved NumPy neural-network artifact."""
    del modified_time  # Included in the cache key to detect file changes.

    return load_neural_network_model(model_path)


@st.cache_data
def load_evaluation_metrics(
    metrics_path: str,
    artifact_path: str,
    metrics_modified_time: float,
    artifact_modified_time: float,
    expected_model_name: str,
):
    """Load metrics associated with the current model artifact."""
    # Timestamps are cache keys; validation is performed by the utility.
    del metrics_modified_time, artifact_modified_time

    return load_model_metrics(
        metrics_path,
        artifact_path,
        expected_model_name=expected_model_name,
    )


def get_evaluation_metrics(
    metrics_path: Path,
    artifact_path: Path,
    model_name: str,
):
    """Load metrics when both files exist and match the current artifact."""
    if not metrics_path.is_file() or not artifact_path.is_file():
        return None

    try:
        return load_evaluation_metrics(
            str(metrics_path),
            str(artifact_path),
            metrics_path.stat().st_mtime,
            artifact_path.stat().st_mtime,
            model_name,
        )
    except (OSError, ValueError, TypeError):
        return None


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
# 6. LOAD MODEL-SPECIFIC METRICS
# ============================================================

random_forest_metrics = (
    get_evaluation_metrics(
        RANDOM_FOREST_METRICS_PATH,
        RANDOM_FOREST_PATH,
        "random_forest",
    )
    if random_forest is not None
    else None
)

neural_network_metrics = (
    get_evaluation_metrics(
        NEURAL_NETWORK_METRICS_PATH,
        NEURAL_NETWORK_PATH,
        "neural_network",
    )
    if neural_network is not None
    else None
)


# ============================================================
# 7. HEADER AND MODEL SELECTION
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
# 8. PROPERTY INPUTS
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
# 9. PREDICTION
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
# 10. MODEL PERFORMANCE
# ============================================================

st.divider()
st.header("Model Performance")

if selected_model == "Random Forest":
    performance_metrics = random_forest_metrics
else:
    performance_metrics = neural_network_metrics

if performance_metrics is None:
    st.warning(
        "Evaluation metrics are unavailable or do not match the current "
        "model artifact. Predictions can still work. Run the appropriate "
        "training script to generate fresh metrics."
    )

    if selected_model == "Random Forest":
        st.code("python train.py", language="bash")
    else:
        st.code("python train_neural_network.py", language="bash")

else:
    saved_metrics = performance_metrics["metrics"]
    evaluation = performance_metrics["evaluation"]
    test_samples = performance_metrics["test_samples"]

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("R² Score", f"{saved_metrics['r2']:.4f}")

    with metric_col2:
        st.metric("RMSE", f"${saved_metrics['rmse']:,.2f}")

    st.metric("Mean Absolute Error (MAE)", f"${saved_metrics['mae']:,.2f}")

    st.write(
        f"The R² score is **{saved_metrics['r2'] * 100:.1f}%** "
        "when expressed as a percentage."
    )

    st.caption(
        f"Evaluated on {test_samples:,} held-out samples. "
        f"Test split: {evaluation['test_size']:.0%}; "
        f"random state: {evaluation['random_state']}. "
        "Metrics are loaded from the saved evaluation file and validated "
        "against the current model artifact."
    )


# ============================================================
# 11. RANDOM FOREST FEATURE IMPORTANCE
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
# 12. ABOUT THE PROJECT
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
# 13. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "For educational and demonstration purposes only. "
    "Predictions are estimates, not professional real-estate valuations."
)
