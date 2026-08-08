"""Reusable evaluation diagnostics for Phase 3 candidates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features import TARGET_COLUMN
from metrics import regression_metrics


def prediction_bias(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residual = actual - predicted
    return {
        "mean_residual": float(residual.mean()),
        "median_residual": float(np.median(residual)),
        "overprediction_percent": float((predicted > actual).mean() * 100),
        "underprediction_percent": float((predicted < actual).mean() * 100),
    }


def calibration_by_target_decile(
    y_true: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Return calibration and bias diagnostics by actual-target decile."""
    frame = pd.DataFrame({"actual": y_true, "prediction": y_pred})
    frame["target_decile"] = pd.qcut(
        frame["actual"], q=10, labels=False, duplicates="drop"
    ) + 1
    return (
        frame.groupby("target_decile", as_index=False)
        .agg(
            count=("actual", "size"),
            mean_actual=("actual", "mean"),
            mean_prediction=("prediction", "mean"),
        )
        .assign(
            mae=lambda item: frame.assign(
                absolute_error=(frame["actual"] - frame["prediction"]).abs()
            ).groupby("target_decile")["absolute_error"].mean().to_numpy(),
            bias=lambda item: item["mean_actual"] - item["mean_prediction"],
        )
    )


def diagnostic_masks(
    train: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, dict[str, pd.Series]]:
    train_p90 = float(train[TARGET_COLUMN].quantile(0.90))
    train_p95 = float(train[TARGET_COLUMN].quantile(0.95))
    distance_band = pd.cut(
        validation["distance"],
        [-np.inf, 500, 1_000, 2_000, np.inf],
        labels=["<=500", "500-1000", "1000-2000", ">2000"],
    )
    missing_market = validation["market_index"].isna()
    missing_weight = validation["weight"].isna()
    groups: dict[str, dict[str, pd.Series]] = {
        "target_tail": {
            "top_10_percent": validation[TARGET_COLUMN] >= train_p90,
            "top_5_percent": validation[TARGET_COLUMN] >= train_p95,
        },
        "equipment": {
            str(value): validation["equipment"].eq(value)
            for value in sorted(validation["equipment"].dropna().unique())
        },
        "distance_band": {
            str(value): distance_band.eq(value) for value in distance_band.cat.categories
        },
        "missingness": {
            "market_index_missing": missing_market,
            "weight_missing": missing_weight,
            "neither_missing": ~missing_market & ~missing_weight,
        },
    }
    train_routes = set(zip(train["pickup"], train["delivery"]))
    route_seen = pd.Series(
        [pair in train_routes for pair in zip(validation["pickup"], validation["delivery"])],
        index=validation.index,
    )
    groups["route_status"] = {"seen": route_seen, "unseen": ~route_seen}
    return groups


def calculate_error_slices(
    experiment: str,
    fold: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_series = pd.Series(prediction, index=validation.index)
    for slice_type, masks in diagnostic_masks(train, validation).items():
        for slice_value, mask in masks.items():
            count = int(mask.sum())
            if count == 0:
                continue
            metrics = regression_metrics(
                validation.loc[mask, TARGET_COLUMN].to_numpy(),
                prediction_series.loc[mask].to_numpy(),
            )
            bias = prediction_bias(
                validation.loc[mask, TARGET_COLUMN].to_numpy(),
                prediction_series.loc[mask].to_numpy(),
            )
            rows.append({
                "experiment": experiment,
                "fold": fold,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "count": count,
                **metrics,
                **bias,
            })
    return rows


def deterministic_city_holdout(
    frame: pd.DataFrame, city_count: int = 5
) -> tuple[pd.Index, pd.Index, list[str]]:
    """Hold out the most frequent endpoint cities using feature counts only."""
    endpoint_counts = pd.concat([frame["pickup"], frame["delivery"]]).value_counts()
    held_out = sorted(endpoint_counts.head(city_count).index.astype(str).tolist())
    test_mask = frame["pickup"].isin(held_out) | frame["delivery"].isin(held_out)
    train_index, test_index = frame.index[~test_mask], frame.index[test_mask]
    if train_index.empty or test_index.empty:
        raise ValueError("Synthetic city holdout produced an empty partition")
    remaining = set(frame.loc[train_index, "pickup"]).union(frame.loc[train_index, "delivery"])
    if set(held_out).intersection(remaining):
        raise ValueError("Held-out cities leaked into synthetic training data")
    return train_index, test_index, held_out
