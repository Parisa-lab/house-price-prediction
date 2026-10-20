"""Automated tests for the house-price prediction project.

Run from the repository root:
    python -m pytest -v
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_metrics import load_model_metrics, save_model_metrics
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

SAMPLE_METRICS = {
    "mae": 20_000.0,
    "rmse": 30_000.0,
    "r2": 0.85,
}


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


@pytest.fixture
def metrics_artifacts(tmp_path):
    """Create a temporary model artifact and its evaluation metrics."""
    artifact_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    artifact_path.write_bytes(b"test model artifact")

    save_model_metrics(
        metrics_path,
        artifact_path,
        model_name="test_model",
        target="SalePrice",
        features=list(FEATURES),
        metrics=SAMPLE_METRICS,
        test_samples=100,
        test_size=0.2,
        random_state=42,
    )

    return metrics_path, artifact_path


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


def test_model_metrics_save_and_load(metrics_artifacts):
    """Verify saved metrics can be loaded for the matching artifact."""
    metrics_path, artifact_path = metrics_artifacts

    payload = load_model_metrics(
        metrics_path,
        artifact_path,
        expected_model_name="test_model",
    )

    assert payload is not None
    assert payload["schema_version"] == 1
    assert payload["model_name"] == "test_model"
    assert payload["target"] == "SalePrice"
    assert payload["features"] == list(FEATURES)
    assert payload["test_samples"] == 100
    assert payload["evaluation"]["test_size"] == 0.2
    assert payload["evaluation"]["random_state"] == 42
    assert payload["metrics"] == SAMPLE_METRICS


def test_model_metrics_reject_changed_artifact(metrics_artifacts):
    """Reject metrics when the model artifact has changed."""
    metrics_path, artifact_path = metrics_artifacts
    artifact_path.write_bytes(b"updated model artifact")

    payload = load_model_metrics(
        metrics_path,
        artifact_path,
        expected_model_name="test_model",
    )

    assert payload is None


def test_model_metrics_reject_wrong_model_name(metrics_artifacts):
    """Reject metrics associated with a different model."""
    metrics_path, artifact_path = metrics_artifacts

    payload = load_model_metrics(
        metrics_path,
        artifact_path,
        expected_model_name="another_model",
    )

    assert payload is None


def test_model_metrics_reject_invalid_json(tmp_path):
    """Return None rather than crashing on malformed metrics JSON."""
    metrics_path = tmp_path / "metrics.json"
    artifact_path = tmp_path / "model.pkl"

    artifact_path.write_bytes(b"test model artifact")
    metrics_path.write_text("{invalid json", encoding="utf-8")

    payload = load_model_metrics(
        metrics_path,
        artifact_path,
        expected_model_name="test_model",
    )

    assert payload is None