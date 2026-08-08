"""Run Phase 2 chronological baseline experiments and diagnostic reporting data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import (
    EquipmentShrinkageRatePerMileRegressor,
    GlobalMedianRegressor,
    MedianRatePerMileRegressor,
    build_ridge_pipeline,
    inverse_target,
    transform_target,
)
from data_audit import find_repository_root
from features import TARGET_COLUMN
from metrics import regression_metrics
from validation import TemporalFold, expanding_monthly_folds


@dataclass(frozen=True)
class Experiment:
    model: str
    feature_set: str
    target_transform: str

    @property
    def key(self) -> str:
        if self.model == "ridge":
            return f"ridge__{self.feature_set}__{self.target_transform}"
        return self.model


def experiments() -> list[Experiment]:
    """Return the small, purposeful Phase 2 experiment matrix."""
    statistical = [
        Experiment("global_median", "statistical", "direct"),
        Experiment("global_median_rpm", "statistical", "direct"),
        Experiment("equipment_shrunk_rpm", "statistical", "direct"),
    ]
    ridge = [
        Experiment("ridge", feature_set, target_transform)
        for feature_set in [
            "minimal", "full", "full_without_signals", "december_compatible"
        ]
        for target_transform in ["direct", "log1p"]
    ]
    signal_ablations = [
        Experiment("ridge", "full_market_only", "direct"),
        Experiment("ridge", "full_quote_only", "direct"),
    ]
    return [*statistical, *ridge, *signal_ablations]


def fit_predict(
    experiment: Experiment,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> np.ndarray:
    """Fit one experiment on one fold and return finite positive predictions."""
    if experiment.model == "global_median":
        estimator: Any = GlobalMedianRegressor()
    elif experiment.model == "global_median_rpm":
        estimator = MedianRatePerMileRegressor()
    elif experiment.model == "equipment_shrunk_rpm":
        estimator = EquipmentShrinkageRatePerMileRegressor(prior_strength=50.0)
    elif experiment.model == "ridge":
        estimator = build_ridge_pipeline(experiment.feature_set, alpha=10.0)
    else:
        raise KeyError(experiment.model)

    y_train = train[TARGET_COLUMN]
    fitted_target = (
        transform_target(y_train, experiment.target_transform)
        if experiment.model == "ridge" else y_train
    )
    estimator.fit(train, fitted_target)
    raw_prediction = estimator.predict(validation)
    prediction = (
        inverse_target(raw_prediction, experiment.target_transform)
        if experiment.model == "ridge" else np.maximum(np.asarray(raw_prediction, dtype=float), 1e-6)
    )
    if len(prediction) != len(validation) or not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise ValueError(f"Invalid predictions from {experiment.key}")
    return prediction


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rolling-fold mean, variability, and worst-fold metrics."""
    return (
        results.groupby(["experiment", "model", "feature_set", "target_transform"], as_index=False)
        .agg(
            folds=("fold", "size"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            worst_mae=("mae", "max"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            worst_rmse=("rmse", "max"),
            mean_wmape=("wmape", "mean"),
            std_wmape=("wmape", "std"),
            worst_wmape=("wmape", "max"),
        )
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )


def slice_masks(
    train: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, dict[str, pd.Series]]:
    """Create evaluation-only familiarity, missingness, and business slices."""
    train_routes = set(zip(train["pickup"], train["delivery"]))
    route_seen = pd.Series(
        [pair in train_routes for pair in zip(validation["pickup"], validation["delivery"])],
        index=validation.index,
    )
    pickup_seen = validation["pickup"].isin(set(train["pickup"].dropna()))
    delivery_seen = validation["delivery"].isin(set(train["delivery"].dropna()))
    both_city_seen = pickup_seen & delivery_seen
    distance_band = pd.cut(
        validation["distance"],
        bins=[-np.inf, 500, 1_000, 2_000, np.inf],
        labels=["<=500", "500-1000", "1000-2000", ">2000"],
        right=True,
    )
    expensive_threshold = float(train[TARGET_COLUMN].quantile(0.90))
    groups: dict[str, dict[str, pd.Series]] = {
        "route_status": {"seen": route_seen, "unseen": ~route_seen},
        "city_status": {"both_seen": both_city_seen, "new_endpoint": ~both_city_seen},
        "market_index_missing": {
            "present": validation["market_index"].notna(),
            "missing": validation["market_index"].isna(),
        },
        "weight_missing": {
            "present": validation["weight"].notna(),
            "missing": validation["weight"].isna(),
        },
        "equipment": {
            str(value): validation["equipment"].eq(value)
            for value in sorted(validation["equipment"].dropna().unique())
        },
        "distance_band": {
            str(value): distance_band.eq(value) for value in distance_band.cat.categories
        },
        "actual_rate_band": {
            "below_train_p90": validation[TARGET_COLUMN] < expensive_threshold,
            "at_or_above_train_p90": validation[TARGET_COLUMN] >= expensive_threshold,
        },
    }
    return groups


def calculate_slices(
    experiment_key: str,
    fold: TemporalFold,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prediction_series = pd.Series(prediction, index=validation.index)
    for slice_type, values in slice_masks(train, validation).items():
        for slice_value, mask in values.items():
            count = int(mask.sum())
            if count == 0:
                continue
            metric = regression_metrics(
                validation.loc[mask, TARGET_COLUMN].to_numpy(),
                prediction_series.loc[mask].to_numpy(),
            )
            rows.append({
                "experiment": experiment_key,
                "fold": fold.name,
                "train_end": fold.train_end.date().isoformat(),
                "slice_type": slice_type,
                "slice_value": slice_value,
                "count": count,
                **metric,
            })
    return rows


def save_diagnostic_figures(
    figure_dir: Path,
    experiment_key: str,
    october: pd.DataFrame,
    prediction: np.ndarray,
    results: pd.DataFrame,
    slices: pd.DataFrame,
) -> None:
    """Create focused diagnostics for the strongest rolling Ridge model."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    actual = october[TARGET_COLUMN].to_numpy(dtype=float)
    residual = actual - prediction
    absolute_error = np.abs(residual)
    color = "#0B6673"
    display_name = experiment_key.replace("__", " | ").replace("_", " ")

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    ax.scatter(actual, prediction, s=8, alpha=0.22, color=color, edgecolors="none")
    limit = max(actual.max(), prediction.max())
    ax.plot([0, limit], [0, limit], linestyle="--", color="#D95F59", linewidth=1.4)
    ax.set(title=f"October actual vs predicted\n{display_name}", xlabel="Actual posted rate ($)", ylabel="Predicted rate ($)")
    fig.tight_layout()
    fig.savefig(figure_dir / "actual_vs_predicted_october.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    cap = np.quantile(np.abs(residual), 0.99)
    ax.hist(np.clip(residual, -cap, cap), bins=60, color=color, alpha=0.9)
    ax.axvline(0, color="#D95F59", linestyle="--")
    ax.set(title="October residual distribution (1st-99th percentile display)", xlabel="Actual - predicted ($)", ylabel="Loads")
    fig.tight_layout()
    fig.savefig(figure_dir / "residual_distribution_october.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5), dpi=150)
    ax.scatter(actual, absolute_error, s=8, alpha=0.20, color=color, edgecolors="none")
    ax.set(title="October absolute error vs actual rate", xlabel="Actual posted rate ($)", ylabel="Absolute error ($)")
    fig.tight_layout()
    fig.savefig(figure_dir / "absolute_error_vs_actual_october.png", bbox_inches="tight")
    plt.close(fig)

    october_slices = slices[(slices["fold"] == "2025-10") & (slices["experiment"] == experiment_key)]
    for slice_type, filename, title in [
        ("distance_band", "mae_by_distance_band_october.png", "October MAE by distance band"),
        ("equipment", "mae_by_equipment_october.png", "October MAE by equipment"),
    ]:
        subset = october_slices[october_slices["slice_type"] == slice_type]
        fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
        ax.bar(subset["slice_value"], subset["mae"], color=color)
        ax.set(title=title, xlabel=slice_type.replace("_", " ").title(), ylabel="MAE ($)")
        fig.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.89)
        fig.savefig(figure_dir / filename, bbox_inches="tight")
        plt.close(fig)

    best_results = results[results["experiment"] == experiment_key].sort_values("validation_month")
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=150)
    ax.plot(best_results["validation_month"], best_results["mae"], marker="o", color=color, label="MAE")
    ax.plot(best_results["validation_month"], best_results["rmse"], marker="o", color="#D95F59", label="RMSE")
    ax.set(title="Strongest Ridge error by validation month", xlabel="Validation month", ylabel="Error ($)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "error_by_fold.png", bbox_inches="tight")
    plt.close(fig)


def run(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(root / "train-test.csv")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    folds = expanding_monthly_folds()
    result_rows: list[dict[str, Any]] = []
    stored_predictions: dict[tuple[str, str], np.ndarray] = {}
    fold_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for fold in folds:
        train_index, validation_index = fold.split(data)
        train, validation = data.loc[train_index].copy(), data.loc[validation_index].copy()
        fold_frames[fold.name] = (train, validation)
        for experiment in experiments():
            started = perf_counter()
            prediction = fit_predict(experiment, train, validation)
            elapsed = perf_counter() - started
            metric = regression_metrics(validation[TARGET_COLUMN].to_numpy(), prediction)
            result_rows.append({
                "experiment": experiment.key,
                "model": experiment.model,
                "feature_set": experiment.feature_set,
                "target_transform": experiment.target_transform,
                "fold": fold.name,
                "train_start": fold.train_start.date().isoformat(),
                "train_end": fold.train_end.date().isoformat(),
                "validation_month": fold.name,
                "validation_start": fold.validation_start.date().isoformat(),
                "validation_end": fold.validation_end.date().isoformat(),
                "train_rows": len(train),
                "validation_rows": len(validation),
                **metric,
                "fit_predict_seconds": elapsed,
            })
            stored_predictions[(experiment.key, fold.name)] = prediction
            print(f"{fold.name} | {experiment.key} | MAE={metric['mae']:.2f} RMSE={metric['rmse']:.2f} WMAPE={metric['wmape']:.4f}")

    results = pd.DataFrame(result_rows)
    summary = summarize_results(results)
    strongest_ridge = summary[summary["model"] == "ridge"].iloc[0]["experiment"]
    slice_rows: list[dict[str, Any]] = []
    slice_experiments = [strongest_ridge]
    alternate_transform = strongest_ridge.replace("__direct", "__log1p")
    if (alternate_transform, "2025-10") in stored_predictions:
        slice_experiments.append(alternate_transform)
    for fold in folds:
        train, validation = fold_frames[fold.name]
        for experiment_key in slice_experiments:
            slice_rows.extend(calculate_slices(
                experiment_key,
                fold,
                train,
                validation,
                stored_predictions[(experiment_key, fold.name)],
            ))
    slices = pd.DataFrame(slice_rows)

    reports = root / "reports"
    results.to_csv(reports / "baseline_results.csv", index=False)
    summary.to_csv(reports / "baseline_summary.csv", index=False)
    slices.to_csv(reports / "baseline_error_slices.csv", index=False)
    _, october = fold_frames["2025-10"]
    save_diagnostic_figures(
        reports / "figures" / "phase2",
        strongest_ridge,
        october,
        stored_predictions[(strongest_ridge, "2025-10")],
        results,
        slices,
    )
    return results, summary, slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Repository root (auto-detected by default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_repository_root(args.root)
    results, summary, _ = run(root)
    print("\nRolling summary ordered by mean MAE:")
    print(summary[["experiment", "mean_mae", "mean_rmse", "mean_wmape", "worst_mae"]].to_string(index=False))
    print(f"\nSaved {len(results)} fold results under {root / 'reports'}")


if __name__ == "__main__":
    main()
