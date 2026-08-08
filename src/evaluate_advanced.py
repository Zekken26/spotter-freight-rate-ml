"""Run compact Phase 3 advanced-model temporal benchmarks and stress tests."""

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
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from advanced_models import (
    CATBOOST_CONFIGS,
    CATBOOST_REFINED_CONFIG,
    CatBoostConfig,
    build_catboost,
    build_hist_gradient_boosting,
    constrain_predictions,
    policy_columns,
    prepare_advanced_features,
)
from baselines import build_ridge_pipeline, inverse_target
from data_audit import find_repository_root
from diagnostics import (
    calculate_error_slices,
    calibration_by_target_decile,
    deterministic_city_holdout,
    prediction_bias,
)
from features import TARGET_COLUMN
from metrics import regression_metrics
from validation import TemporalFold, expanding_monthly_folds


@dataclass(frozen=True)
class AdvancedExperiment:
    model: str
    policy: str
    config: str
    include_route: bool = False

    @property
    def key(self) -> str:
        route = "route" if self.include_route else "no_route"
        return f"{self.model}__{self.config}__{self.policy}__{route}"


def fit_predict(
    experiment: AdvancedExperiment,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cat_configs: dict[str, CatBoostConfig],
) -> np.ndarray:
    """Fit one fold-local advanced experiment and return positive predictions."""
    if experiment.model == "catboost":
        config = cat_configs[experiment.config]
        train_features = prepare_advanced_features(
            train, experiment.policy, experiment.include_route
        )
        validation_features = prepare_advanced_features(
            validation, experiment.policy, experiment.include_route
        )
        _, categorical = policy_columns(experiment.policy, experiment.include_route)
        model = build_catboost(config)
        model.fit(train_features, train[TARGET_COLUMN], cat_features=categorical)
        raw = model.predict(validation_features)
    elif experiment.model == "hist_gradient_boosting":
        model = build_hist_gradient_boosting(experiment.policy)
        model.fit(train, train[TARGET_COLUMN])
        raw = model.predict(validation)
    elif experiment.model == "ridge_phase2":
        phase2_policy = {
            "full": "full",
            "full_market_only": "full_market_only",
            "no_signal": "full_without_signals",
            "december_compatible": "december_compatible",
        }[experiment.policy]
        model = build_ridge_pipeline(phase2_policy, alpha=10.0)
        model.fit(train, train[TARGET_COLUMN])
        raw = inverse_target(model.predict(validation), "direct")
    else:
        raise KeyError(experiment.model)
    prediction = constrain_predictions(raw)
    if len(prediction) != len(validation):
        raise ValueError(f"Prediction length mismatch for {experiment.key}")
    return prediction


def fold_result(
    experiment: AdvancedExperiment,
    fold: TemporalFold,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    prediction: np.ndarray,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "experiment": experiment.key,
        "model": experiment.model,
        "config": experiment.config,
        "policy": experiment.policy,
        "include_route": experiment.include_route,
        "fold": fold.name,
        "train_end": fold.train_end.date().isoformat(),
        "validation_month": fold.name,
        "train_rows": len(train),
        "validation_rows": len(validation),
        **regression_metrics(validation[TARGET_COLUMN].to_numpy(), prediction),
        **prediction_bias(validation[TARGET_COLUMN].to_numpy(), prediction),
        "fit_predict_seconds": elapsed,
    }


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby(
            ["experiment", "model", "config", "policy", "include_route"],
            as_index=False,
        )
        .agg(
            folds=("fold", "size"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            worst_mae=("mae", "max"),
            mean_rmse=("rmse", "mean"),
            worst_rmse=("rmse", "max"),
            mean_wmape=("wmape", "mean"),
            worst_wmape=("wmape", "max"),
            mean_fit_seconds=("fit_predict_seconds", "mean"),
        )
    )
    october = results[results["fold"] == "2025-10"][
        ["experiment", "mae", "rmse", "wmape"]
    ].rename(columns={"mae": "october_mae", "rmse": "october_rmse", "wmape": "october_wmape"})
    summary = grouped.merge(october, on="experiment", how="left")
    ridge = summary.loc[summary["model"] == "ridge_phase2"].iloc[0]
    summary["rolling_mae_improvement_vs_ridge"] = ridge["mean_mae"] - summary["mean_mae"]
    summary["rolling_mae_improvement_percent_vs_ridge"] = (
        summary["rolling_mae_improvement_vs_ridge"] / ridge["mean_mae"] * 100
    )
    summary["october_mae_improvement_vs_ridge"] = ridge["october_mae"] - summary["october_mae"]
    summary["worst_mae_improvement_vs_ridge"] = ridge["worst_mae"] - summary["worst_mae"]
    return summary.sort_values(
        ["mean_mae", "worst_mae", "std_mae", "october_mae"]
    ).reset_index(drop=True)


def market_distribution(train: pd.DataFrame, final_validation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dated = train.assign(period=pd.to_datetime(train["date"]).dt.to_period("M").astype(str))
    frames = [("labeled", "all", train), *[("labeled", period, group) for period, group in dated.groupby("period")], ("final_validation", "2025-11_to_12", final_validation)]
    for dataset, period, frame in frames:
        values = frame["market_index"]
        rows.append({
            "dataset": dataset,
            "period": period,
            "count": len(frame),
            "missing": int(values.isna().sum()),
            "minimum": values.min(),
            "p05": values.quantile(0.05),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "p95": values.quantile(0.95),
            "maximum": values.max(),
        })
    return pd.DataFrame(rows)


def market_sensitivity_rows(
    experiment: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    quantiles = train["market_index"].dropna().quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).to_numpy()
    edges = np.unique(quantiles)
    bins = pd.cut(validation["market_index"], bins=edges, include_lowest=True, duplicates="drop")
    prediction_series = pd.Series(prediction, index=validation.index)
    rows: list[dict[str, Any]] = []
    for interval in bins.cat.categories:
        mask = bins.eq(interval)
        if not mask.any():
            continue
        actual = validation.loc[mask, TARGET_COLUMN].to_numpy()
        predicted = prediction_series.loc[mask].to_numpy()
        rows.append({
            "experiment": experiment,
            "market_bin": str(interval),
            "count": int(mask.sum()),
            "mean_market_index": float(validation.loc[mask, "market_index"].mean()),
            **regression_metrics(actual, predicted),
            **prediction_bias(actual, predicted),
        })
    missing = validation["market_index"].isna()
    if missing.any():
        actual = validation.loc[missing, TARGET_COLUMN].to_numpy()
        predicted = prediction_series.loc[missing].to_numpy()
        rows.append({
            "experiment": experiment,
            "market_bin": "missing",
            "count": int(missing.sum()),
            "mean_market_index": np.nan,
            **regression_metrics(actual, predicted),
            **prediction_bias(actual, predicted),
        })
    return rows


def save_figures(
    output: Path,
    results: pd.DataFrame,
    slices: pd.DataFrame,
    calibration: pd.DataFrame,
    city_results: pd.DataFrame,
    october_frame: pd.DataFrame,
    october_predictions: dict[str, np.ndarray],
    leading_cat: str,
    comparison_keys: list[str],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#0B6673", "#D95F59", "#6C5B7B", "#C58B32"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    for color, key in zip(colors, comparison_keys):
        subset = results[results["experiment"] == key].sort_values("fold")
        ax.plot(subset["fold"], subset["mae"], marker="o", label=key.split("__")[0], color=color)
    ax.set(title="Temporal MAE comparison", xlabel="Validation month", ylabel="MAE ($)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "model_comparison_folds.png", bbox_inches="tight")
    plt.close(fig)

    actual = october_frame[TARGET_COLUMN].to_numpy()
    prediction = october_predictions[leading_cat]
    residual = actual - prediction
    fig, ax = plt.subplots(figsize=(6.4, 5.5), dpi=150)
    ax.scatter(actual, prediction, s=8, alpha=0.2, color=colors[0], edgecolors="none")
    limit = max(actual.max(), prediction.max())
    ax.plot([0, limit], [0, limit], "--", color=colors[1])
    ax.set(title="Leading CatBoost: October actual vs predicted", xlabel="Actual rate ($)", ylabel="Predicted rate ($)")
    fig.tight_layout()
    fig.savefig(output / "catboost_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.4), dpi=150)
    cap = np.quantile(np.abs(residual), 0.99)
    ax.hist(np.clip(residual, -cap, cap), bins=60, color=colors[0])
    ax.axvline(0, linestyle="--", color=colors[1])
    ax.set(title="Leading CatBoost October residuals (1st-99th percentile display)", xlabel="Actual - predicted ($)", ylabel="Loads")
    fig.tight_layout()
    fig.savefig(output / "catboost_residual_distribution.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5), dpi=150)
    for color, (key, group) in zip(colors, calibration.groupby("experiment")):
        ax.plot(group["mean_actual"], group["mean_prediction"], marker="o", label=key.split("__")[0], color=color)
    max_value = calibration[["mean_actual", "mean_prediction"]].to_numpy().max()
    ax.plot([0, max_value], [0, max_value], "--", color="#555555")
    ax.set(title="October target-decile calibration", xlabel="Mean actual ($)", ylabel="Mean prediction ($)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "target_decile_calibration.png", bbox_inches="tight")
    plt.close(fig)

    def comparison_bar(slice_type: str, filename: str, title: str) -> None:
        subset = slices[(slices["fold"] == "2025-10") & (slices["slice_type"] == slice_type) & slices["experiment"].isin(comparison_keys[:3])]
        pivot = subset.pivot(index="slice_value", columns="experiment", values="mae")
        pivot.columns = [column.split("__")[0] for column in pivot.columns]
        ax = pivot.plot(kind="bar", figsize=(8.2, 4.7), color=colors[: len(pivot.columns)], rot=0)
        ax.set(title=title, xlabel=slice_type.replace("_", " ").title(), ylabel="MAE ($)")
        ax.legend(frameon=False)
        ax.figure.tight_layout()
        ax.figure.savefig(output / filename, bbox_inches="tight")
        plt.close(ax.figure)

    comparison_bar("distance_band", "mae_by_distance_band.png", "October MAE by distance band")
    comparison_bar("equipment", "mae_by_equipment.png", "October MAE by equipment")
    comparison_bar("target_tail", "tail_error_comparison.png", "October expensive-load MAE")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    ax.bar(city_results["model"], city_results["mae"], color=colors[: len(city_results)])
    ax.set(title="Synthetic unseen-city stress test", xlabel="Model", ylabel="MAE ($)")
    fig.tight_layout()
    fig.savefig(output / "unseen_city_stress.png", bbox_inches="tight")
    plt.close(fig)


def run(root: Path) -> None:
    data = pd.read_csv(root / "train-test.csv", parse_dates=["date"])
    final_validation = pd.read_csv(root / "validation.csv", parse_dates=["date"])
    folds = expanding_monthly_folds()
    fold_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold in folds:
        train_index, validation_index = fold.split(data)
        fold_frames[fold.name] = (data.loc[train_index].copy(), data.loc[validation_index].copy())

    cat_configs = {
        config.name: config for config in [*CATBOOST_CONFIGS, CATBOOST_REFINED_CONFIG]
    }
    result_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], np.ndarray] = {}

    def evaluate(
        experiment: AdvancedExperiment, selected_folds: list[TemporalFold] | None = None
    ) -> None:
        for fold in selected_folds or folds:
            train, validation = fold_frames[fold.name]
            started = perf_counter()
            prediction = fit_predict(experiment, train, validation, cat_configs)
            elapsed = perf_counter() - started
            row = fold_result(experiment, fold, train, validation, prediction, elapsed)
            result_rows.append(row)
            predictions[(experiment.key, fold.name)] = prediction
            print(f"{fold.name} | {experiment.key} | MAE={row['mae']:.2f} RMSE={row['rmse']:.2f} WMAPE={row['wmape']:.4f} | {elapsed:.1f}s")

    # Tune on July-September, preserving October as the primary untouched check.
    tuning_folds = folds[:-1]
    for config in CATBOOST_CONFIGS:
        evaluate(
            AdvancedExperiment("catboost", "full", config.name, False), tuning_folds
        )
    cat_tuning_results = pd.DataFrame(result_rows)
    best_config = (
        cat_tuning_results.groupby("config", as_index=False)
        .agg(mean_mae=("mae", "mean"), worst_mae=("mae", "max"), std_mae=("mae", "std"))
        .sort_values(["mean_mae", "worst_mae", "std_mae"])
        .iloc[0]["config"]
    )
    print(f"Selected CatBoost config for policy expansion: {best_config}")
    evaluate(
        AdvancedExperiment("catboost", "full", best_config, False), [folds[-1]]
    )

    for policy in ["full_market_only", "no_signal", "december_compatible"]:
        evaluate(AdvancedExperiment("catboost", policy, best_config, False))
    evaluate(AdvancedExperiment("catboost", "full", best_config, True))
    for policy in ["full", "full_market_only"]:
        evaluate(AdvancedExperiment(
            "catboost", policy, CATBOOST_REFINED_CONFIG.name, False
        ))

    for policy in ["full", "full_market_only", "no_signal", "december_compatible"]:
        evaluate(AdvancedExperiment("hist_gradient_boosting", policy, "hist_default", False))
    evaluate(AdvancedExperiment("ridge_phase2", "full", "alpha10", False))

    results = pd.DataFrame(result_rows)
    summary = summarize(results)
    cat_full = summary[(summary["model"] == "catboost") & (summary["policy"] == "full") & (summary["config"] == best_config)]
    leading_cat_row = summary[
        (summary["model"] == "catboost") & (summary["folds"] == 4)
    ].iloc[0]
    leading_cat = leading_cat_row["experiment"]
    leading_hist = summary[(summary["model"] == "hist_gradient_boosting") & (summary["policy"] == "full")].iloc[0]["experiment"]
    ridge_key = summary[summary["model"] == "ridge_phase2"].iloc[0]["experiment"]
    cat_market = summary[(summary["model"] == "catboost") & (summary["policy"] == "full_market_only")].iloc[0]["experiment"]
    cat_no_signal = summary[(summary["model"] == "catboost") & (summary["policy"] == "no_signal")].iloc[0]["experiment"]
    cat_december = summary[(summary["model"] == "catboost") & (summary["policy"] == "december_compatible")].iloc[0]["experiment"]
    cat_route_keys = cat_full["experiment"].tolist()
    serious = list(dict.fromkeys([leading_cat, *cat_route_keys, cat_market, cat_no_signal, cat_december, leading_hist, ridge_key]))

    slice_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    for key in serious:
        for fold in folds:
            train, validation = fold_frames[fold.name]
            prediction = predictions[(key, fold.name)]
            slice_rows.extend(calculate_error_slices(key, fold.name, train, validation, prediction))
            bias_rows.append({"experiment": key, "fold": fold.name, **prediction_bias(validation[TARGET_COLUMN].to_numpy(), prediction)})
    slices = pd.DataFrame(slice_rows)
    bias = pd.DataFrame(bias_rows)

    october_train, october = fold_frames["2025-10"]
    calibration_keys = [leading_cat, leading_hist, ridge_key]
    calibration_frames = []
    for key in calibration_keys:
        item = calibration_by_target_decile(october[TARGET_COLUMN].to_numpy(), predictions[(key, "2025-10")])
        item.insert(0, "experiment", key)
        calibration_frames.append(item)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    market_sensitivity = pd.DataFrame([
        row
        for key in calibration_keys
        for row in market_sensitivity_rows(key, october_train, october, predictions[(key, "2025-10")])
    ])

    # Separate feature-only synthetic unknown-city stress test.
    city_train_index, city_test_index, held_cities = deterministic_city_holdout(data, city_count=5)
    city_train, city_test = data.loc[city_train_index].copy(), data.loc[city_test_index].copy()
    stress_experiments = [
        AdvancedExperiment(
            "catboost",
            str(leading_cat_row["policy"]),
            str(leading_cat_row["config"]),
            bool(leading_cat_row["include_route"]),
        ),
        AdvancedExperiment("hist_gradient_boosting", "full", "hist_default", False),
        AdvancedExperiment("ridge_phase2", "full", "alpha10", False),
    ]
    city_rows = []
    for experiment in stress_experiments:
        started = perf_counter()
        prediction = fit_predict(experiment, city_train, city_test, cat_configs)
        city_rows.append({
            "model": experiment.model,
            "experiment": experiment.key,
            "held_out_cities": "|".join(held_cities),
            "train_rows": len(city_train),
            "test_rows": len(city_test),
            **regression_metrics(city_test[TARGET_COLUMN].to_numpy(), prediction),
            **prediction_bias(city_test[TARGET_COLUMN].to_numpy(), prediction),
            "fit_predict_seconds": perf_counter() - started,
        })
    city_results = pd.DataFrame(city_rows)

    reports = root / "reports"
    results.to_csv(reports / "advanced_results.csv", index=False)
    summary.to_csv(reports / "advanced_summary.csv", index=False)
    slices.to_csv(reports / "advanced_error_slices.csv", index=False)
    bias.to_csv(reports / "advanced_prediction_bias.csv", index=False)
    calibration.to_csv(reports / "advanced_calibration.csv", index=False)
    city_results.to_csv(reports / "unseen_city_stress_results.csv", index=False)
    market_sensitivity.to_csv(reports / "market_index_sensitivity.csv", index=False)
    market_distribution(data, final_validation).to_csv(reports / "market_index_distribution.csv", index=False)
    (reports / "held_out_cities.txt").write_text("\n".join(held_cities) + "\n", encoding="utf-8")

    save_figures(
        reports / "figures" / "phase3",
        results,
        slices,
        calibration,
        city_results,
        october,
        {key: predictions[(key, "2025-10")] for key in calibration_keys},
        leading_cat,
        [leading_cat, leading_hist, ridge_key],
    )
    print("\nAdvanced ranking:")
    print(summary[["experiment", "mean_mae", "std_mae", "worst_mae", "october_mae", "mean_rmse", "mean_wmape"]].to_string(index=False))
    print(f"\nSelected CatBoost: {leading_cat}")
    print(f"Held-out cities: {held_cities}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Repository root; auto-detected by default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(find_repository_root(args.root))


if __name__ == "__main__":
    main()
