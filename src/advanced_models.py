"""Advanced feature policies and CPU-efficient boosting model factories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from features import DATE_FEATURES, DateFeatureTransformer, assert_no_forbidden_features


RANDOM_SEED = 42
MINIMUM_PREDICTION = 1e-6

BASE_NUMERIC = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance", "weight"
]
BASE_CATEGORICAL = ["pickup", "delivery", "equipment"]
GEO_FEATURES = ["abs_lat_delta", "abs_lon_delta"]
DOMAIN_FEATURES = ["weight_per_mile"]

ADVANCED_POLICIES = {
    "full": {
        "numeric": [*BASE_NUMERIC, "market_index", "quote_signal", *DATE_FEATURES, *GEO_FEATURES, *DOMAIN_FEATURES],
        "categorical": BASE_CATEGORICAL,
    },
    "full_market_only": {
        "numeric": [*BASE_NUMERIC, "market_index", *DATE_FEATURES, *GEO_FEATURES, *DOMAIN_FEATURES],
        "categorical": BASE_CATEGORICAL,
    },
    "no_signal": {
        "numeric": [*BASE_NUMERIC, *DATE_FEATURES, *GEO_FEATURES, *DOMAIN_FEATURES],
        "categorical": BASE_CATEGORICAL,
    },
    "december_compatible": {
        "numeric": ["distance", "weight", *DATE_FEATURES, *DOMAIN_FEATURES],
        "categorical": BASE_CATEGORICAL,
    },
}


@dataclass(frozen=True)
class CatBoostConfig:
    name: str
    iterations: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float = 3.0


CATBOOST_CONFIGS = [
    # Low-budget screen: both 700 and 250 trees exceeded the laptop resource gate.
    CatBoostConfig("depth6_lr012", iterations=80, depth=6, learning_rate=0.12),
    CatBoostConfig("depth8_lr010", iterations=100, depth=8, learning_rate=0.10),
]

# Focused refinement used only after the low-budget screen identifies depth 8.
CATBOOST_REFINED_CONFIG = CatBoostConfig(
    "depth8_lr005_refined", iterations=350, depth=8, learning_rate=0.05
)


def policy_columns(policy: str, include_route: bool = False) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns exposed by one policy."""
    if policy not in ADVANCED_POLICIES:
        raise KeyError(f"Unknown advanced feature policy: {policy}")
    numeric = list(ADVANCED_POLICIES[policy]["numeric"])
    categorical = list(ADVANCED_POLICIES[policy]["categorical"])
    if include_route:
        categorical.append("route")
    assert_no_forbidden_features([*numeric, *categorical])
    return numeric, categorical


def create_route(frame: pd.DataFrame) -> pd.Series:
    """Create a deterministic directed route token without target information."""
    pickup = frame["pickup"].astype("string").fillna("__MISSING__")
    delivery = frame["delivery"].astype("string").fillna("__MISSING__")
    return pickup + " -> " + delivery


def prepare_advanced_features(
    frame: pd.DataFrame, policy: str, include_route: bool = False
) -> pd.DataFrame:
    """Generate target-free date/domain features and select an exact policy."""
    transformed = DateFeatureTransformer().fit_transform(frame)
    distance = pd.to_numeric(transformed["distance"], errors="coerce")
    weight = pd.to_numeric(transformed["weight"], errors="coerce")
    transformed["weight_per_mile"] = np.where(distance > 0, weight / distance, np.nan)
    if policy != "december_compatible":
        transformed["abs_lat_delta"] = (
            transformed["delivery_lat"] - transformed["pickup_lat"]
        ).abs()
        transformed["abs_lon_delta"] = (
            transformed["delivery_lon"] - transformed["pickup_lon"]
        ).abs()
    if include_route:
        transformed["route"] = create_route(frame)
    numeric, categorical = policy_columns(policy, include_route)
    result = transformed[[*numeric, *categorical]].copy()
    for column in categorical:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result


def build_catboost(config: CatBoostConfig) -> CatBoostRegressor:
    """Build a deterministic CPU CatBoost regressor with no filesystem logging."""
    return CatBoostRegressor(
        iterations=config.iterations,
        depth=config.depth,
        learning_rate=config.learning_rate,
        l2_leaf_reg=config.l2_leaf_reg,
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )


class AdvancedFeatureTransformer(TransformerMixin, BaseEstimator):
    """Sklearn-compatible wrapper around deterministic advanced features."""

    def __init__(self, policy: str) -> None:
        self.policy = policy

    def fit(self, X: pd.DataFrame, y: object = None) -> "AdvancedFeatureTransformer":
        policy_columns(self.policy)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return prepare_advanced_features(X, self.policy, include_route=False)


def build_hist_gradient_boosting(policy: str) -> Pipeline:
    """Build fold-local ordinal encoding plus HistGradientBoosting."""
    numeric, categorical = policy_columns(policy)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])
    categorical_mask = [False] * len(numeric) + [True] * len(categorical)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        categorical_features=categorical_mask,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )
    return Pipeline([
        ("features", AdvancedFeatureTransformer(policy)),
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def constrain_predictions(prediction: np.ndarray) -> np.ndarray:
    """Require finite predictions and safely constrain them positive."""
    values = np.asarray(prediction, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Advanced model produced non-finite predictions")
    return np.maximum(values, MINIMUM_PREDICTION)
