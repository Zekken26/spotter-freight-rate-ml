"""Regression metrics used consistently across all Phase 2 experiments."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error, returned as a fraction."""
    actual, predicted = _validated_arrays(y_true, y_pred)
    denominator = float(np.sum(np.abs(actual)))
    if denominator <= 0:
        raise ValueError("WMAPE is undefined when sum(abs(y_true)) is zero")
    return float(np.sum(np.abs(actual - predicted)) / denominator)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the standard Phase 2 metric set."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
    }


def _validated_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays of equal shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite actual and predicted values")
    return actual, predicted
