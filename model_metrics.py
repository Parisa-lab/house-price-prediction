
"""Save and load evaluation metrics associated with trained model artifacts."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
REQUIRED_METRICS = ("mae", "rmse", "r2")


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_model_metrics(
    metrics_path: str | Path,
    artifact_path: str | Path,
    *,
    model_name: str,
    target: str,
    features: list[str],
    metrics: dict[str, float],
    test_samples: int,
    test_size: float,
    random_state: int,
) -> None:
    """Save validated evaluation metrics linked to a specific model artifact."""
    metrics_path = Path(metrics_path)
    artifact_path = Path(artifact_path)

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact_path}"
        )

    if not model_name.strip() or not target.strip() or not features:
        raise ValueError("Model name, target, and features are required.")

    if test_samples < 1:
        raise ValueError("test_samples must be positive.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not isinstance(random_state, int):
        raise ValueError("random_state must be an integer.")

    missing_metrics = set(REQUIRED_METRICS) - set(metrics)
    if missing_metrics:
        raise ValueError(
            f"Missing required metrics: {sorted(missing_metrics)}"
        )

    clean_metrics = {}
    for name in REQUIRED_METRICS:
        value = float(metrics[name])
        if not math.isfinite(value):
            raise ValueError(f"Metric {name!r} must be finite.")
        clean_metrics[name] = value

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "target": target,
        "features": list(features),
        "test_samples": test_samples,
        "evaluation": {
            "test_size": float(test_size),
            "random_state": random_state,
        },
        "metrics": clean_metrics,
        "artifact_sha256": _sha256_file(artifact_path),
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=metrics_path.parent,
            prefix=f".{metrics_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")

        os.replace(temporary_path, metrics_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_model_metrics(
    metrics_path: str | Path,
    artifact_path: str | Path,
    *,
    expected_model_name: str,
) -> dict[str, Any] | None:
    """Load metrics only if they match the current model artifact.

    Return None when the metrics are missing, invalid, or belong to a
    different model artifact. Prediction can still work without metrics.
    """
    metrics_path = Path(metrics_path)
    artifact_path = Path(artifact_path)

    if not metrics_path.is_file() or not artifact_path.is_file():
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("schema_version") != SCHEMA_VERSION:
            return None

        if payload.get("model_name") != expected_model_name:
            return None

        if payload.get("artifact_sha256") != _sha256_file(artifact_path):
            return None

        saved_metrics = payload.get("metrics")
        if not isinstance(saved_metrics, dict):
            return None

        for name in REQUIRED_METRICS:
            value = saved_metrics.get(name)
            if not isinstance(value, (int, float)):
                return None
            if not math.isfinite(value):
                return None

        if not isinstance(payload.get("test_samples"), int):
            return None
        if payload["test_samples"] < 1:
            return None

        if not isinstance(payload.get("features"), list):
            return None
        if not isinstance(payload.get("target"), str):
            return None
        if not isinstance(payload.get("evaluation"), dict):
            return None

        return payload

    except (OSError, ValueError, TypeError, AttributeError):
        return None
