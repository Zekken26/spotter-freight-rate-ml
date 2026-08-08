"""Deterministic, target-free feature definitions for Phase 2 models."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


TARGET_COLUMN = "posted_rate"
ID_COLUMN = "load_id"
DATE_ORIGIN = pd.Timestamp("2025-01-01")
DATE_FEATURES = [
    "month", "day_of_week", "day_of_month", "day_of_year", "week_of_year",
    "is_weekend", "days_since_start",
]

FEATURE_SETS = {
    "minimal": {
        "numeric": ["distance", "weight", *DATE_FEATURES],
        "categorical": ["equipment"],
    },
    "full": {
        "numeric": [
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "distance", "weight", "market_index", "quote_signal", *DATE_FEATURES,
        ],
        "categorical": ["pickup", "delivery", "equipment"],
    },
    "full_without_signals": {
        "numeric": [
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "distance", "weight", *DATE_FEATURES,
        ],
        "categorical": ["pickup", "delivery", "equipment"],
    },
    "full_market_only": {
        "numeric": [
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "distance", "weight", "market_index", *DATE_FEATURES,
        ],
        "categorical": ["pickup", "delivery", "equipment"],
    },
    "full_quote_only": {
        "numeric": [
            "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "distance", "weight", "quote_signal", *DATE_FEATURES,
        ],
        "categorical": ["pickup", "delivery", "equipment"],
    },
    "december_compatible": {
        "numeric": ["distance", "weight", *DATE_FEATURES],
        "categorical": ["pickup", "delivery", "equipment"],
    },
}


class DateFeatureTransformer(TransformerMixin, BaseEstimator):
    """Replace raw date with a small fixed set of calendar/elapsed features."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "DateFeatureTransformer":
        if "date" not in X.columns:
            raise ValueError("Expected date column")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        dates = pd.to_datetime(frame["date"], errors="raise")
        iso = dates.dt.isocalendar()
        frame["month"] = dates.dt.month.astype(float)
        frame["day_of_week"] = dates.dt.dayofweek.astype(float)
        frame["day_of_month"] = dates.dt.day.astype(float)
        frame["day_of_year"] = dates.dt.dayofyear.astype(float)
        frame["week_of_year"] = iso.week.astype(float)
        frame["is_weekend"] = (dates.dt.dayofweek >= 5).astype(float)
        frame["days_since_start"] = (dates - DATE_ORIGIN).dt.days.astype(float)
        return frame.drop(columns=["date"])


def feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    """Return numeric and categorical output columns for a named feature set."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set: {feature_set}")
    definition = FEATURE_SETS[feature_set]
    return list(definition["numeric"]), list(definition["categorical"])


def predictive_input_columns(feature_set: str) -> list[str]:
    """Return raw columns required before date feature generation."""
    numeric, categorical = feature_columns(feature_set)
    raw = [column for column in [*numeric, *categorical] if column not in DATE_FEATURES]
    return list(dict.fromkeys([*raw, "date"]))


def assert_no_forbidden_features(columns: Sequence[str]) -> None:
    """Fail if identifier or target columns enter a predictive feature list."""
    forbidden = {ID_COLUMN, TARGET_COLUMN}.intersection(columns)
    if forbidden:
        raise ValueError(f"Forbidden predictive columns: {sorted(forbidden)}")
