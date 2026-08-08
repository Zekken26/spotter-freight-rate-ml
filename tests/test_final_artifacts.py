"""Contract and deterministic-inference tests for Phase 4 deliverables."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_audit import EXPECTED_SCHEMAS, load_datasets  # noqa: E402
from features import feature_columns  # noqa: E402
from finalize import predict_from_saved_models  # noqa: E402


class FinalArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frames = load_datasets(ROOT)
        cls.main = pd.read_csv(ROOT / "validation_predictions.csv")
        cls.december = pd.read_csv(ROOT / "outputs" / "december_chart_predictions.csv")

    def test_main_submission_contract(self) -> None:
        self.assertEqual(self.main.columns.tolist(), ["load_id", "predicted_rate"])
        self.assertEqual(len(self.main), 12_000)
        self.assertFalse(self.main["load_id"].isna().any())
        self.assertFalse(self.main["load_id"].duplicated().any())
        self.assertEqual(self.main["load_id"].tolist(), self.frames["template"]["load_id"].tolist())
        self.assertTrue(np.isfinite(self.main["predicted_rate"]).all())
        self.assertTrue((self.main["predicted_rate"] > 0).all())

    def test_december_contract_and_preserved_inputs(self) -> None:
        self.assertEqual(self.december.columns.tolist(), EXPECTED_SCHEMAS["december"])
        self.assertEqual(len(self.december), 31)
        pd.testing.assert_frame_equal(
            self.december.drop(columns="predicted_rate"),
            self.frames["december"].drop(columns="predicted_rate"),
            check_dtype=False,
        )
        dates = pd.to_datetime(self.december["date"], errors="raise")
        self.assertEqual(dates.nunique(), 31)
        self.assertEqual(dates.min(), pd.Timestamp("2025-12-01"))
        self.assertEqual(dates.max(), pd.Timestamp("2025-12-31"))
        self.assertTrue(np.isfinite(self.december["predicted_rate"]).all())
        self.assertTrue((self.december["predicted_rate"] > 0).all())

    def test_locked_feature_policies_exclude_id_and_target(self) -> None:
        numeric, categorical = feature_columns("full")
        self.assertTrue({"load_id", "posted_rate"}.isdisjoint([*numeric, *categorical]))
        self.assertTrue({"market_index", "quote_signal"}.isdisjoint(
            ["distance", "weight", "pickup", "delivery", "equipment"]
        ))

    def test_saved_model_inference_is_deterministic(self) -> None:
        main_rerun, december_rerun = predict_from_saved_models(ROOT)
        pd.testing.assert_frame_equal(self.main, main_rerun, check_exact=False, rtol=1e-12, atol=1e-12)
        pd.testing.assert_frame_equal(
            self.december, december_rerun, check_exact=False, rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
