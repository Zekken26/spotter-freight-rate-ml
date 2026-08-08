"""Fit the locked Phase 4 models and create final prediction artifacts.

The script reads employer inputs without modifying them, aligns the main
submission explicitly by ``load_id``, and writes all generated outputs to new
paths. Run from any directory with ``python src/finalize.py``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import catboost
import joblib
import numpy as np
import pandas as pd
import sklearn

from advanced_models import (
    CatBoostConfig,
    build_catboost,
    constrain_predictions,
    policy_columns,
    prepare_advanced_features,
)
from baselines import MINIMUM_PREDICTION, build_ridge_pipeline
from data_audit import EXPECTED_SCHEMAS, find_repository_root, load_datasets


RIDGE_ALPHA = 10.0
DECEMBER_POLICY = "december_compatible"
DECEMBER_CONFIG = CatBoostConfig(
    name="final_december", iterations=100, depth=8, learning_rate=0.10, l2_leaf_reg=3.0
)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_prediction(values: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate finite raw predictions and apply only the documented tiny floor."""
    raw = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(raw).all():
        raise ValueError("Model produced non-finite predictions")
    floor_mask = raw <= 0
    final = np.maximum(raw, MINIMUM_PREDICTION)
    return final, {
        "raw_minimum": float(raw.min()),
        "raw_maximum": float(raw.max()),
        "nonpositive_raw_count": int(floor_mask.sum()),
        "positive_floor": MINIMUM_PREDICTION,
    }


def align_main_predictions(
    validation: pd.DataFrame, template: pd.DataFrame, predicted_rate: np.ndarray
) -> pd.DataFrame:
    """Align predictions to the employer template by ID with one-to-one checks."""
    if validation["load_id"].isna().any() or validation["load_id"].duplicated().any():
        raise ValueError("Validation load_id values must be present and unique")
    if template["load_id"].isna().any() or template["load_id"].duplicated().any():
        raise ValueError("Template load_id values must be present and unique")
    if len(predicted_rate) != len(validation):
        raise ValueError("Prediction count does not match validation rows")

    by_id = pd.DataFrame({
        "load_id": validation["load_id"].astype(str),
        "predicted_rate": predicted_rate,
    })
    aligned = template[["load_id"]].copy()
    aligned["load_id"] = aligned["load_id"].astype(str)
    aligned = aligned.merge(by_id, on="load_id", how="left", validate="one_to_one")
    missing = int(aligned["predicted_rate"].isna().sum())
    extra = set(by_id["load_id"]) - set(aligned["load_id"])
    if missing or extra or len(aligned) != len(template):
        raise ValueError(f"ID alignment failed (missing={missing}, extra={len(extra)})")
    return aligned[["load_id", "predicted_rate"]]


def prediction_statistics(values: pd.Series | np.ndarray) -> dict[str, float]:
    series = pd.Series(np.asarray(values, dtype=float))
    return {
        "count": int(series.size),
        "minimum": float(series.min()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "mean": float(series.mean()),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "maximum": float(series.max()),
        "standard_deviation": float(series.std()),
    }


def slice_summary(validation: pd.DataFrame, predicted: np.ndarray) -> dict[str, Any]:
    frame = validation.copy()
    frame["predicted_rate"] = predicted
    frame["month"] = pd.to_datetime(frame["date"], errors="raise").dt.to_period("M").astype(str)
    frame["distance_band"] = pd.cut(
        frame["distance"], [-np.inf, 500, 1000, 2000, np.inf],
        labels=["<=500", "500-1000", "1000-2000", ">2000"],
    )
    market = pd.to_numeric(frame["market_index"], errors="coerce")
    try:
        frame["market_index_bin"] = pd.qcut(market, 5, duplicates="drop").astype("string")
    except ValueError:
        frame["market_index_bin"] = "insufficient_variation"

    def grouped(column: str) -> list[dict[str, Any]]:
        result = []
        for key, group in frame.groupby(column, observed=True, dropna=False):
            result.append({
                column: str(key),
                "rows": int(len(group)),
                "mean": float(group["predicted_rate"].mean()),
                "median": float(group["predicted_rate"].median()),
                "minimum": float(group["predicted_rate"].min()),
                "maximum": float(group["predicted_rate"].max()),
            })
        return result

    return {
        "overall": prediction_statistics(predicted),
        "by_equipment": grouped("equipment"),
        "by_distance_band": grouped("distance_band"),
        "by_month": grouped("month"),
        "by_market_index_bin": grouped("market_index_bin"),
    }


def fit_and_write(root: Path) -> dict[str, Any]:
    frames = load_datasets(root)
    train, validation = frames["train"], frames["validation"]
    template, december_input = frames["template"], frames["december"]
    target = pd.to_numeric(train["posted_rate"], errors="raise")

    (root / "models").mkdir(exist_ok=True)
    (root / "outputs").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)

    ridge = build_ridge_pipeline("full", alpha=RIDGE_ALPHA)
    ridge.fit(train, target)
    main_prediction, ridge_floor = positive_prediction(ridge.predict(validation))
    submission = align_main_predictions(validation, template, main_prediction)
    main_path = root / "validation_predictions.csv"
    submission.to_csv(main_path, index=False, lineterminator="\n")
    ridge_path = root / "models" / "final_ridge.joblib"
    joblib.dump(ridge, ridge_path)

    train_features = prepare_advanced_features(train, DECEMBER_POLICY, include_route=False)
    december_features = prepare_advanced_features(december_input, DECEMBER_POLICY, include_route=False)
    _, cat_features = policy_columns(DECEMBER_POLICY, include_route=False)
    december_model = build_catboost(DECEMBER_CONFIG)
    december_model.fit(train_features, target, cat_features=cat_features)
    december_prediction, cat_floor = positive_prediction(december_model.predict(december_features))
    # Retain constrain_predictions as a second contract check for the advanced model.
    december_prediction = constrain_predictions(december_prediction)
    december_output = december_input.copy()
    december_output["predicted_rate"] = december_prediction
    december_output = december_output[EXPECTED_SCHEMAS["december"]]
    december_path = root / "outputs" / "december_chart_predictions.csv"
    december_output.to_csv(december_path, index=False, lineterminator="\n")
    cat_path = root / "models" / "final_december_catboost.cbm"
    december_model.save_model(cat_path, format="cbm")

    summary = slice_summary(validation, main_prediction)
    train_stats = prediction_statistics(target)
    dec_stats = prediction_statistics(december_prediction)
    dec_stats.update({
        "first_five": december_output[["date", "predicted_rate"]].head().to_dict("records"),
        "last_five": december_output[["date", "predicted_rate"]].tail().to_dict("records"),
        "mean_absolute_day_to_day_change": float(
            december_output["predicted_rate"].diff().abs().dropna().mean()
        ),
        "maximum_absolute_day_to_day_change": float(
            december_output["predicted_rate"].diff().abs().dropna().max()
        ),
    })
    summary.update({"training_target": train_stats, "december": dec_stats})
    summary_path = root / "reports" / "final_prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metadata = {
        "training": {
            "rows": int(len(train)),
            "date_min": str(pd.to_datetime(train["date"]).min().date()),
            "date_max": str(pd.to_datetime(train["date"]).max().date()),
            "target": "posted_rate",
        },
        "main_model": {
            "role": "assessment validation",
            "algorithm": "sklearn.linear_model.Ridge",
            "feature_policy": "full",
            "target_transform": "direct",
            "alpha": RIDGE_ALPHA,
            "model_path": "models/final_ridge.joblib",
            "prediction_path": "validation_predictions.csv",
            "prediction_rows": int(len(submission)),
            "prediction_range": [float(main_prediction.min()), float(main_prediction.max())],
            "training_rows": int(len(train)),
            "training_date_range": [
                str(pd.to_datetime(train["date"]).min().date()),
                str(pd.to_datetime(train["date"]).max().date()),
            ],
            "random_seed": None,
            "floor_diagnostics": ridge_floor,
        },
        "december_model": {
            "role": "December chart",
            "algorithm": "catboost.CatBoostRegressor",
            "feature_policy": DECEMBER_POLICY,
            "features": train_features.columns.tolist(),
            "route_feature": False,
            "iterations": DECEMBER_CONFIG.iterations,
            "depth": DECEMBER_CONFIG.depth,
            "learning_rate": DECEMBER_CONFIG.learning_rate,
            "l2_leaf_reg": DECEMBER_CONFIG.l2_leaf_reg,
            "loss_function": "RMSE",
            "random_seed": 42,
            "task_type": "CPU",
            "model_path": "models/final_december_catboost.cbm",
            "prediction_path": "outputs/december_chart_predictions.csv",
            "prediction_rows": int(len(december_output)),
            "prediction_range": [float(december_prediction.min()), float(december_prediction.max())],
            "training_rows": int(len(train)),
            "training_date_range": [
                str(pd.to_datetime(train["date"]).min().date()),
                str(pd.to_datetime(train["date"]).max().date()),
            ],
            "floor_diagnostics": cat_floor,
        },
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": catboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    metadata["sha256"] = {
        "validation_predictions.csv": sha256_file(main_path),
        "outputs/december_chart_predictions.csv": sha256_file(december_path),
        "models/final_ridge.joblib": sha256_file(ridge_path),
        "models/final_december_catboost.cbm": sha256_file(cat_path),
    }
    metadata_path = root / "reports" / "final_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def predict_from_saved_models(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rerun inference from saved artifacts without writing any files."""
    frames = load_datasets(root)
    ridge = joblib.load(root / "models" / "final_ridge.joblib")
    main, _ = positive_prediction(ridge.predict(frames["validation"]))
    aligned = align_main_predictions(frames["validation"], frames["template"], main)

    from catboost import CatBoostRegressor

    cat = CatBoostRegressor()
    cat.load_model(root / "models" / "final_december_catboost.cbm", format="cbm")
    features = prepare_advanced_features(frames["december"], DECEMBER_POLICY, include_route=False)
    december, _ = positive_prediction(cat.predict(features))
    december_frame = frames["december"].copy()
    december_frame["predicted_rate"] = december
    return aligned, december_frame[EXPECTED_SCHEMAS["december"]]


def main() -> None:
    root = find_repository_root()
    metadata = fit_and_write(root)
    print(f"Created {metadata['main_model']['prediction_rows']:,} aligned validation predictions.")
    print(f"Created {metadata['december_model']['prediction_rows']} December predictions.")
    print("Saved both locked models and reproducibility metadata.")


if __name__ == "__main__":
    main()
