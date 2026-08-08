"""Fold-local statistical baselines and Ridge model construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import DateFeatureTransformer, assert_no_forbidden_features, feature_columns


MINIMUM_PREDICTION = 1e-6


class GlobalMedianRegressor(RegressorMixin, BaseEstimator):
    """Predict the training-fold target median."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GlobalMedianRegressor":
        self.median_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.median_, dtype=float)


class MedianRatePerMileRegressor(RegressorMixin, BaseEstimator):
    """Predict distance times the training-fold median rate per mile."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MedianRatePerMileRegressor":
        distance = pd.to_numeric(X["distance"], errors="coerce").to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        valid = np.isfinite(distance) & (distance > 0) & np.isfinite(target)
        if not valid.any():
            raise ValueError("No valid positive distances for rate-per-mile baseline")
        self.median_rate_per_mile_ = float(np.median(target[valid] / distance[valid]))
        self.fallback_target_ = float(np.median(target[valid]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        distance = pd.to_numeric(X["distance"], errors="coerce").to_numpy(dtype=float)
        prediction = np.full(len(X), self.fallback_target_, dtype=float)
        valid = np.isfinite(distance) & (distance > 0)
        prediction[valid] = distance[valid] * self.median_rate_per_mile_
        return np.maximum(prediction, MINIMUM_PREDICTION)


class EquipmentShrinkageRatePerMileRegressor(RegressorMixin, BaseEstimator):
    """Equipment median RPM shrunk toward global RPM by n/(n+prior_strength)."""

    def __init__(self, prior_strength: float = 50.0) -> None:
        self.prior_strength = prior_strength

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> "EquipmentShrinkageRatePerMileRegressor":
        frame = pd.DataFrame({
            "equipment": X["equipment"].astype("string").fillna("__MISSING__"),
            "distance": pd.to_numeric(X["distance"], errors="coerce"),
            "target": np.asarray(y, dtype=float),
        })
        valid = frame["distance"].gt(0) & np.isfinite(frame["distance"]) & np.isfinite(frame["target"])
        frame = frame.loc[valid].assign(rpm=lambda item: item["target"] / item["distance"])
        if frame.empty:
            raise ValueError("No valid rows for equipment rate-per-mile baseline")
        self.global_rpm_ = float(frame["rpm"].median())
        self.fallback_target_ = float(frame["target"].median())
        grouped = frame.groupby("equipment")["rpm"].agg(["median", "count"])
        weight = grouped["count"] / (grouped["count"] + self.prior_strength)
        self.equipment_rpm_ = (
            weight * grouped["median"] + (1.0 - weight) * self.global_rpm_
        ).to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        distance = pd.to_numeric(X["distance"], errors="coerce").to_numpy(dtype=float)
        equipment = X["equipment"].astype("string").fillna("__MISSING__")
        rpm = equipment.map(self.equipment_rpm_).fillna(self.global_rpm_).to_numpy(dtype=float)
        prediction = np.full(len(X), self.fallback_target_, dtype=float)
        valid = np.isfinite(distance) & (distance > 0)
        prediction[valid] = distance[valid] * rpm[valid]
        return np.maximum(prediction, MINIMUM_PREDICTION)


def build_ridge_pipeline(feature_set: str, alpha: float = 10.0) -> Pipeline:
    """Build a fold-fit-only preprocessing and fixed-alpha Ridge pipeline."""
    numeric_columns, categorical_columns = feature_columns(feature_set)
    assert_no_forbidden_features([*numeric_columns, *categorical_columns])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("date_features", DateFeatureTransformer()),
        ("preprocessing", preprocessing),
        ("model", Ridge(alpha=alpha)),
    ])


def transform_target(y: pd.Series, strategy: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if strategy == "direct":
        return values
    if strategy == "log1p":
        return np.log1p(values)
    raise KeyError(f"Unknown target transform: {strategy}")


def inverse_target(prediction: np.ndarray, strategy: str) -> np.ndarray:
    values = np.asarray(prediction, dtype=float)
    if strategy == "log1p":
        values = np.expm1(np.clip(values, -20.0, 20.0))
    elif strategy != "direct":
        raise KeyError(f"Unknown target transform: {strategy}")
    if not np.isfinite(values).all():
        raise ValueError("Model produced non-finite predictions")
    return np.maximum(values, MINIMUM_PREDICTION)
