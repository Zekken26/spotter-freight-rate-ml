"""Fast contract tests for the Phase 1 audit."""

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_audit import (  # noqa: E402
    EXPECTED_SCHEMAS,
    build_audit,
    find_repository_root,
    load_datasets,
)


class DataAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = find_repository_root(Path(__file__))
        cls.frames = load_datasets(cls.root)

    def test_source_schemas_and_row_counts(self) -> None:
        self.assertEqual(self.frames["train"].shape, (48_000, 14))
        self.assertEqual(self.frames["validation"].shape, (12_000, 13))
        self.assertEqual(self.frames["template"].shape, (12_000, 2))
        self.assertEqual(self.frames["december"].shape, (31, 7))

    def test_target_and_identifier_contracts(self) -> None:
        self.assertIn("posted_rate", self.frames["train"])
        self.assertNotIn("posted_rate", self.frames["validation"])
        self.assertTrue(self.frames["train"]["load_id"].is_unique)
        self.assertTrue(self.frames["validation"]["load_id"].is_unique)
        self.assertEqual(
            self.frames["template"]["load_id"].tolist(),
            self.frames["validation"]["load_id"].tolist(),
        )

    def test_december_only_date_changes(self) -> None:
        audit = build_audit(self.frames)
        self.assertTrue(audit["december_analysis"]["only_date_changes_among_inputs"])
        self.assertTrue(audit["december_analysis"]["dates_unique"])

    def test_schema_failure_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp_path = Path(temporary_directory)
            for label, filename in {
                "train": "train-test.csv",
                "validation": "validation.csv",
                "template": "validation-predictions-template.csv",
                "december": "december-chart-inputs.csv",
            }.items():
                frame = self.frames[label].head(1).copy()
                if label == "train":
                    frame = frame.rename(columns={EXPECTED_SCHEMAS[label][0]: "unexpected_id"})
                frame.to_csv(temp_path / filename, index=False)
            with self.assertRaisesRegex(ValueError, "Unexpected train schema"):
                load_datasets(temp_path)


if __name__ == "__main__":
    unittest.main()
