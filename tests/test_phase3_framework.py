"""Fast correctness and robustness tests for Phase 3."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advanced_models import (  # noqa: E402
    CatBoostConfig,
    build_catboost,
    constrain_predictions,
    create_route,
    policy_columns,
    prepare_advanced_features,
)
from diagnostics import deterministic_city_holdout  # noqa: E402
from features import ID_COLUMN, TARGET_COLUMN  # noqa: E402


def tiny_frame() -> pd.DataFrame:
    count = 24
    return pd.DataFrame({
        "load_id": [f"X-{index}" for index in range(count)],
        "pickup": ["A", "B", "C"] * 8,
        "delivery": ["D", "E"] * 12,
        "pickup_lat": np.linspace(30, 40, count),
        "pickup_lon": np.linspace(-100, -80, count),
        "delivery_lat": np.linspace(32, 42, count),
        "delivery_lon": np.linspace(-98, -78, count),
        "distance": np.linspace(100, 1_000, count),
        "equipment": ["Dry Van", "Reefer", "Flatbed"] * 8,
        "weight": np.linspace(20_000, 40_000, count),
        "date": pd.date_range("2025-01-01", periods=count),
        "market_index": np.linspace(0.8, 1.2, count),
        "quote_signal": np.linspace(1.5, 2.5, count),
        "posted_rate": np.linspace(300, 2_500, count),
    })


class Phase3FrameworkTests(unittest.TestCase):
    def test_feature_policies_exclude_identifier_and_target(self) -> None:
        frame = tiny_frame()
        for policy in ["full", "full_market_only", "no_signal", "december_compatible"]:
            features = prepare_advanced_features(frame, policy)
            self.assertNotIn(ID_COLUMN, features.columns)
            self.assertNotIn(TARGET_COLUMN, features.columns)
            numeric, categorical = policy_columns(policy)
            self.assertEqual(features.columns.tolist(), [*numeric, *categorical])

    def test_signal_policies_expose_exact_signals(self) -> None:
        full_numeric, _ = policy_columns("full")
        market_numeric, _ = policy_columns("full_market_only")
        no_signal_numeric, _ = policy_columns("no_signal")
        december_numeric, _ = policy_columns("december_compatible")
        self.assertIn("market_index", full_numeric)
        self.assertIn("quote_signal", full_numeric)
        self.assertIn("market_index", market_numeric)
        self.assertNotIn("quote_signal", market_numeric)
        self.assertNotIn("market_index", no_signal_numeric)
        self.assertNotIn("quote_signal", no_signal_numeric)
        self.assertNotIn("market_index", december_numeric)
        self.assertNotIn("quote_signal", december_numeric)

    def test_route_creation_is_deterministic(self) -> None:
        frame = tiny_frame().head(3)
        first = create_route(frame)
        second = create_route(frame.copy())
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(first.iloc[0], "A -> D")

    def test_catboost_accepts_unseen_categories_and_predicts_positive(self) -> None:
        frame = tiny_frame()
        train = frame.iloc[:20]
        validation = frame.iloc[20:].copy()
        validation["pickup"] = "UNSEEN_PICKUP"
        validation["delivery"] = "UNSEEN_DELIVERY"
        train_features = prepare_advanced_features(train, "full", include_route=True)
        validation_features = prepare_advanced_features(validation, "full", include_route=True)
        _, categorical = policy_columns("full", include_route=True)
        model = build_catboost(CatBoostConfig("test", 25, 4, 0.1))
        model.fit(train_features, train[TARGET_COLUMN], cat_features=categorical)
        prediction = constrain_predictions(model.predict(validation_features))
        self.assertEqual(len(prediction), len(validation))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_city_holdout_removes_held_cities_from_training(self) -> None:
        frame = tiny_frame()
        train_index, test_index, held_out = deterministic_city_holdout(frame, city_count=1)
        train_values = set(frame.loc[train_index, "pickup"]).union(frame.loc[train_index, "delivery"])
        self.assertTrue(set(held_out).isdisjoint(train_values))
        self.assertGreater(len(test_index), 0)
        self.assertTrue(set(train_index).isdisjoint(test_index))


if __name__ == "__main__":
    unittest.main()
