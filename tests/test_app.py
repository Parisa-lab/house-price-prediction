
"""Automated tests for the house-price prediction models.

Run from the repository root:
    python -m pytest -v
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neural_network import (
    FEATURES,
    load_model as load_neural_network,
    predict_price,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
RANDOM_FOREST_PATH = ROOT_DIR / "models" / "model.pkl"
NEURAL_NETWORK_PATH = ROOT_DIR / "models" / "neural_network.npz"

SAMPLE_INPUT = pd.DataFrame(
    [
        {
            "OverallQual": 5,
            "GrLivArea": 1500,
            "GarageCars": 2,
            "TotalBsmtSF": 800,
            "FullBath": 2,
            "YearBuilt": 2000,
        }
    ],
    columns=FEATURES,
)


@pytest.fixture(scope="module")
def random_forest():
    """Load the existing Random Forest model."""
    if not RANDOM_FOREST_PATH.is_file():
        pytest.skip("Random Forest model artifact is not available.")

    with RANDOM_FOREST_PATH.open("rb") as file:
        model = pickle.load(file)

    return model


@pytest.fixture(scope="module")
def neural_network():
    """Load and validate the trained neural-network artifact."""
    if not NEURAL_NETWORK_PATH.is_file():
        pytest.skip("Neural-network model artifact is not available.")

    return load_neural_network(NEURAL_NETWORK_PATH)


def test_random_forest_model_has_expected_pipeline(random_forest):
    """Verify the Random Forest pipeline contains its expected regressor."""
    assert hasattr(random_forest, "named_steps")
    assert "regressor" in random_forest.named_steps

    regressor = random_forest.named_steps["regressor"]
    assert hasattr(regressor, "feature_importances_")
    assert len(regressor.feature_importances_) == len(FEATURES)


def test_random_forest_prediction_is_finite(random_forest):
    """Verify the Random Forest returns one finite prediction."""
    prediction = random_forest.predict(SAMPLE_INPUT)

    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_random_forest_feature_importances_are_valid(random_forest):
    """Verify feature importances are finite and sum to approximately one."""
    importances = random_forest.named_steps[
        "regressor"
    ].feature_importances_

    assert np.isfinite(importances).all()
    assert np.all(importances >= 0)
    assert np.isclose(importances.sum(), 1.0)


def test_neural_network_has_expected_features(neural_network):
    """Verify the neural-network artifact preserves feature order."""
    assert neural_network["features"].tolist() == FEATURES
    assert neural_network["target"].item() == "SalePrice"
    assert neural_network["hidden_units"].tolist() == [32, 16]


def test_neural_network_prediction_is_finite(neural_network):
    """Verify the neural network returns a finite dollar prediction."""
    prediction = predict_price(
        SAMPLE_INPUT.to_numpy(dtype=np.float64),
        neural_network,
    )

    assert isinstance(prediction, float)
    assert np.isfinite(prediction)
    assert prediction > 0


def test_neural_network_rejects_wrong_input_shape(neural_network):
    """Reject inputs that do not contain all six features."""
    invalid_input = np.array([[1500.0, 2.0]])

    with pytest.raises(ValueError, match="shape"):
        predict_price(invalid_input, neural_network)


def test_neural_network_rejects_non_finite_input(neural_network):
    """Reject NaN or infinite input values."""
    invalid_input = SAMPLE_INPUT.to_numpy(dtype=np.float64).copy()
    invalid_input[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        predict_price(invalid_input, neural_network)


def test_neural_network_rejects_missing_model(tmp_path):
    """Raise a clear error when the model artifact does not exist."""
    missing_path = tmp_path / "missing_model.npz"

    with pytest.raises(FileNotFoundError):
        load_neural_network(missing_path)
