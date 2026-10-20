
"""NumPy neural network and inference utilities for house-price prediction.

The network uses two ReLU hidden layers. Training operates on scaled
features and targets; inference applies the saved scaling parameters.

The trained model is stored separately from the Random Forest artifact.
"""

from pathlib import Path

import numpy as np


FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
]
TARGET = "SalePrice"

RANDOM_STATE = 42
HIDDEN_UNITS = (32, 16)
LEARNING_RATE = 0.01
BATCH_SIZE = 32
MAX_EPOCHS = 1000
PATIENCE = 50

REQUIRED_ARRAYS = {
    "W1",
    "b1",
    "W2",
    "b2",
    "W3",
    "b3",
    "x_mean",
    "x_scale",
    "y_mean",
    "y_scale",
    "features",
    "target",
    "hidden_units",
}

PARAMETER_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


class NeuralNetworkRegressor:
    """Fully connected neural network with two ReLU hidden layers."""

    def __init__(
        self,
        input_size: int,
        hidden_units: tuple[int, int] = HIDDEN_UNITS,
        learning_rate: float = LEARNING_RATE,
        seed: int = RANDOM_STATE,
    ) -> None:
        if input_size < 1:
            raise ValueError("input_size must be positive.")
        if len(hidden_units) != 2 or any(
            not isinstance(n, (int, np.integer)) or n < 1
            for n in hidden_units
        ):
            raise ValueError(
                "hidden_units must contain two positive integers."
            )
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")

        self.learning_rate = learning_rate
        self.hidden_units = tuple(hidden_units)

        rng = np.random.default_rng(seed)
        h1, h2 = self.hidden_units

        # He initialization is suitable for ReLU hidden layers.
        self.W1 = rng.normal(
            0, np.sqrt(2 / input_size), (input_size, h1)
        )
        self.b1 = np.zeros((1, h1))
        self.W2 = rng.normal(0, np.sqrt(2 / h1), (h1, h2))
        self.b2 = np.zeros((1, h2))
        self.W3 = rng.normal(0, np.sqrt(1 / h2), (h2, 1))
        self.b3 = np.zeros((1, 1))

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, values)

    def _forward(
        self, X: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return intermediate activations and scaled predictions."""
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self._relu(Z2)
        predictions = A2 @ self.W3 + self.b3

        return Z1, A1, Z2, A2, predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions in the scaled target space."""
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != self.W1.shape[0]:
            raise ValueError(
                f"X must have shape (n_samples, {self.W1.shape[0]})."
            )
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values.")

        predictions = self._forward(X)[-1]
        if not np.isfinite(predictions).all():
            raise ValueError("The neural network produced invalid predictions.")

        return predictions

    def _train_batch(
        self, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Perform one forward pass and gradient-descent update."""
        n = X.shape[0]
        Z1, A1, Z2, A2, predictions = self._forward(X)

        error = predictions - y
        loss = float(np.mean(error**2))

        dZ3 = (2.0 / n) * error
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * (Z2 > 0)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        for name, gradient in (
            ("W1", dW1),
            ("b1", db1),
            ("W2", dW2),
            ("b2", db2),
            ("W3", dW3),
            ("b3", db3),
        ):
            updated = getattr(self, name) - self.learning_rate * gradient
            if not np.isfinite(updated).all():
                raise FloatingPointError(
                    f"Training produced invalid values in {name}."
                )
            setattr(self, name, updated)

        return loss

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict[str, float]]:
        """Train with mini-batches and restore the best validation weights."""
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)

        if X_train.ndim != 2 or X_train.shape[1] != self.W1.shape[0]:
            raise ValueError("X_train has an invalid shape.")
        if X_val.ndim != 2 or X_val.shape[1] != self.W1.shape[0]:
            raise ValueError("X_val has an invalid shape.")
        if y_train.shape != (len(X_train), 1):
            raise ValueError("y_train must have shape (n_samples, 1).")
        if y_val.shape != (len(X_val), 1):
            raise ValueError("y_val must have shape (n_samples, 1).")
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Training and validation data cannot be empty.")
        if not all(
            np.isfinite(values).all()
            for values in (X_train, y_train, X_val, y_val)
        ):
            raise ValueError("Training data contains non-finite values.")

        rng = np.random.default_rng(RANDOM_STATE)
        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_parameters: dict[str, np.ndarray] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            indices = rng.permutation(len(X_train))

            for start in range(0, len(indices), BATCH_SIZE):
                batch_indices = indices[start : start + BATCH_SIZE]
                self._train_batch(
                    X_train[batch_indices],
                    y_train[batch_indices],
                )

            train_predictions = self.predict(X_train)
            val_predictions = self.predict(X_val)

            train_loss = float(np.mean((train_predictions - y_train) ** 2))
            val_loss = float(np.mean((val_predictions - y_val) ** 2))

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_parameters = {
                    name: getattr(self, name).copy()
                    for name in PARAMETER_NAMES
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                break

        if best_parameters is None:
            raise RuntimeError("Training did not produce valid model weights.")

        for name, values in best_parameters.items():
            setattr(self, name, values)

        print(
            f"Training stopped after {len(history)} epochs. "
            f"Best validation MSE (scaled target): {best_val_loss:.6f}"
        )
        return history


def load_model(model_path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a saved neural-network artifact."""
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Neural-network model not found: {path}. "
            "Train the neural network before using it."
        )

    try:
        with np.load(path, allow_pickle=False) as saved:
            missing = REQUIRED_ARRAYS - set(saved.files)
            if missing:
                raise ValueError(
                    f"Model file is missing arrays: {sorted(missing)}"
                )

            model = {
                name: saved[name].copy()
                for name in REQUIRED_ARRAYS
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and "missing arrays" in str(exc):
            raise
        raise ValueError(f"Could not read neural-network model: {exc}") from exc

    if model["features"].tolist() != FEATURES:
        raise ValueError("Saved model features do not match the expected order.")
    if model["target"].shape != () or model["target"].item() != TARGET:
        raise ValueError("Saved model has an unexpected target.")

    h_units = model["hidden_units"]
    if h_units.shape != (2,) or not np.issubdtype(
        h_units.dtype, np.integer
    ):
        raise ValueError("Saved hidden_units must contain two integers.")

    h1, h2 = h_units.tolist()
    if h1 < 1 or h2 < 1:
        raise ValueError("Saved hidden_units must be positive.")

    expected_shapes = {
        "W1": (len(FEATURES), h1),
        "b1": (1, h1),
        "W2": (h1, h2),
        "b2": (1, h2),
        "W3": (h2, 1),
        "b3": (1, 1),
        "x_mean": (len(FEATURES),),
        "x_scale": (len(FEATURES),),
        "y_mean": (1,),
        "y_scale": (1,),
    }

    for name, shape in expected_shapes.items():
        if model[name].shape != shape:
            raise ValueError(
                f"Invalid shape for {name}: expected {shape}, "
                f"got {model[name].shape}."
            )
        if not np.isfinite(model[name]).all():
            raise ValueError(f"Model array {name} contains invalid values.")

    if np.any(model["x_scale"] <= 0):
        raise ValueError("Input-scaling values must be positive.")
    if np.any(model["y_scale"] <= 0):
        raise ValueError("Target-scaling values must be positive.")

    return model


def predict_price(inputs: np.ndarray, model: dict) -> float:
    """Predict one house sale price in original dollar units."""
    inputs = np.asarray(inputs, dtype=np.float64)

    if inputs.shape != (1, len(FEATURES)):
        raise ValueError(
            f"inputs must have shape (1, {len(FEATURES)})."
        )
    if not np.isfinite(inputs).all():
        raise ValueError("Input features contain non-finite values.")

    scaled_inputs = (inputs - model["x_mean"]) / model["x_scale"]

    z1 = scaled_inputs @ model["W1"] + model["b1"]
    a1 = np.maximum(0.0, z1)

    z2 = a1 @ model["W2"] + model["b2"]
    a2 = np.maximum(0.0, z2)

    scaled_prediction = a2 @ model["W3"] + model["b3"]
    prediction = (
        scaled_prediction * model["y_scale"] + model["y_mean"]
    )

    price = float(prediction[0, 0])
    if not np.isfinite(price):
        raise ValueError("The model produced a non-finite prediction.")

    return price
