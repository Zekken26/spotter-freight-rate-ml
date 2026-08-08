"""Chronological split definitions for leakage-safe monthly backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    """An expanding-window training period followed by one validation month."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    def split(self, frame: pd.DataFrame, date_column: str = "date") -> tuple[pd.Index, pd.Index]:
        dates = pd.to_datetime(frame[date_column], errors="raise")
        train_mask = dates.between(self.train_start, self.train_end, inclusive="both")
        validation_mask = dates.between(
            self.validation_start, self.validation_end, inclusive="both"
        )
        train_index, validation_index = frame.index[train_mask], frame.index[validation_mask]
        if train_index.empty or validation_index.empty:
            raise ValueError(f"Fold {self.name} produced an empty partition")
        if dates.loc[train_index].max() >= dates.loc[validation_index].min():
            raise ValueError(f"Temporal leakage detected in fold {self.name}")
        return train_index, validation_index


def expanding_monthly_folds() -> list[TemporalFold]:
    """Return the four assessment-requested expanding monthly folds."""
    definitions = [
        ("2025-07", "2025-06-30", "2025-07-01", "2025-07-31"),
        ("2025-08", "2025-07-31", "2025-08-01", "2025-08-31"),
        ("2025-09", "2025-08-31", "2025-09-01", "2025-09-30"),
        ("2025-10", "2025-09-30", "2025-10-01", "2025-10-31"),
    ]
    return [
        TemporalFold(
            name=name,
            train_start=pd.Timestamp("2025-01-01"),
            train_end=pd.Timestamp(train_end),
            validation_start=pd.Timestamp(validation_start),
            validation_end=pd.Timestamp(validation_end),
        )
        for name, train_end, validation_start, validation_end in definitions
    ]


def primary_october_fold() -> TemporalFold:
    """Return the primary September-cutoff, October-holdout split."""
    return expanding_monthly_folds()[-1]
