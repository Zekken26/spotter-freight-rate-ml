"""Focused leakage and correctness tests for Phase 2."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baselines import (  # noqa: E402
    MedianRatePerMileRegressor,
    build_ridge_pipeline,
    inverse_target,
)
from data_audit import find_repository_root  # noqa: E402
from features import (  # noqa: E402
    ID_COLUMN,
    TARGET_COLUMN,
    predictive_input_columns,
)
from metrics import mae, regression_metrics, rmse, wmape  # noqa: E402
from validation import expanding_monthly_folds  # noqa: E402


class Phase2FrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = find_repository_root(Path(__file__))
        cls.data = pd.read_csv(root / "train-test.csv", parse_dates=["date"])

    def test_temporal_split_boundaries_and_no_date_leakage(self) -> None:
        expected = {
            "2025-07": ("2025-06-30", "2025-07-01", "2025-07-31"),
            "2025-08": ("2025-07-31", "2025-08-01", "2025-08-31"),
            "2025-09": ("2025-08-31", "2025-09-01", "2025-09-30"),
            "2025-10": ("2025-09-30", "2025-10-01", "2025-10-31"),
        }
        for fold in expanding_monthly_folds():
            train_index, validation_index = fold.split(self.data)
            train_dates = self.data.loc[train_index, "date"]
            validation_dates = self.data.loc[validation_index, "date"]
            train_end, validation_start, validation_end = expected[fold.name]
            self.assertEqual(train_dates.max(), pd.Timestamp(train_end))
            self.assertEqual(validation_dates.min(), pd.Timestamp(validation_start))
            self.assertEqual(validation_dates.max(), pd.Timestamp(validation_end))
            self.assertLess(train_dates.max(), validation_dates.min())
            self.assertTrue(set(train_index).isdisjoint(validation_index))

    def test_load_id_and_target_are_excluded_from_all_feature_sets(self) -> None:
        for feature_set in [
            "minimal", "full", "full_without_signals", "full_market_only",
            "full_quote_only", "december_compatible"
        ]:
            columns = predictive_input_columns(feature_set)
            self.assertNotIn(ID_COLUMN, columns)
            self.assertNotIn(TARGET_COLUMN, columns)

    def test_rate_per_mile_uses_training_statistics_only(self) -> None:
        train = pd.DataFrame({"distance": [10.0, 20.0]})
        model = MedianRatePerMileRegressor().fit(train, pd.Series([20.0, 60.0]))
        validation = pd.DataFrame({"distance": [100.0]})
        self.assertAlmostEqual(model.predict(validation)[0], 250.0)

    def test_unseen_category_does_not_crash_ridge_pipeline(self) -> None:
        train = pd.DataFrame({
            "distance": [100.0, 200.0, 300.0],
            "weight": [20_000.0, np.nan, 35_000.0],
            "equipment": ["Dry Van", "Reefer", "Dry Van"],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        })
        validation = pd.DataFrame({
            "distance": [150.0],
            "weight": [25_000.0],
            "equipment": ["Never Seen Equipment"],
            "date": pd.to_datetime(["2025-02-01"]),
        })
        model = build_ridge_pipeline("minimal", alpha=10.0)
        model.fit(train, np.array([300.0, 500.0, 700.0]))
        prediction = model.predict(validation)
        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_log_inverse_is_finite_and_positive(self) -> None:
        prediction = inverse_target(np.array([-1000.0, 0.0, 5.0, 1000.0]), "log1p")
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_metric_values_on_tiny_example(self) -> None:
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        self.assertAlmostEqual(mae(actual, predicted), 15.0)
        self.assertAlmostEqual(rmse(actual, predicted), np.sqrt(250.0))
        self.assertAlmostEqual(wmape(actual, predicted), 0.1)
        self.assertEqual(set(regression_metrics(actual, predicted)), {"mae", "rmse", "wmape"})


if __name__ == "__main__":
    unittest.main()
