"""Reproducible, leakage-aware audit for the Spotter freight-rate datasets.

This module performs no model fitting and never writes to employer-provided files.
Run it from any working directory with::

    python src/data_audit.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET = "posted_rate"
DATE_COLUMN = "date"
ID_COLUMN = "load_id"
CSV_FILES = {
    "train": "train-test.csv",
    "validation": "validation.csv",
    "template": "validation-predictions-template.csv",
    "december": "december-chart-inputs.csv",
}
EXPECTED_SCHEMAS = {
    "train": [
        "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
        "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
        "date", "market_index", "quote_signal", "posted_rate",
    ],
    "validation": [
        "load_id", "pickup", "delivery", "pickup_lat", "pickup_lon",
        "delivery_lat", "delivery_lon", "distance", "equipment", "weight",
        "date", "market_index", "quote_signal",
    ],
    "template": ["load_id", "predicted_rate"],
    "december": [
        "pickup", "delivery", "distance", "equipment", "weight", "date",
        "predicted_rate",
    ],
}
NUMERIC_FEATURES = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "distance",
    "weight", "market_index", "quote_signal",
]
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]


def find_repository_root(start: Path | None = None) -> Path:
    """Find the nearest parent containing every assessment CSV."""
    candidates = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve().parent]
    for candidate in candidates:
        for directory in (candidate, *candidate.parents):
            if all((directory / filename).is_file() for filename in CSV_FILES.values()):
                return directory
    expected = ", ".join(CSV_FILES.values())
    raise FileNotFoundError(f"Could not locate repository root containing: {expected}")


def load_datasets(root: Path) -> dict[str, pd.DataFrame]:
    """Load all CSVs and fail clearly if an employer schema changed."""
    frames: dict[str, pd.DataFrame] = {}
    for label, filename in CSV_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} dataset not found: {path}")
        frame = pd.read_csv(path)
        actual = frame.columns.tolist()
        if actual != EXPECTED_SCHEMAS[label]:
            raise ValueError(
                f"Unexpected {label} schema. Expected {EXPECTED_SCHEMAS[label]}, got {actual}"
            )
        frames[label] = frame
    return frames


def json_ready(value: Any) -> Any:
    """Convert pandas/numpy values into deterministic JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def frame_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Return generic schema, quality, cardinality, and range checks."""
    missing_count = frame.isna().sum()
    numeric = frame.select_dtypes(include=np.number)
    categorical = frame.select_dtypes(exclude=np.number)
    date_min = date_max = None
    invalid_dates = None
    if DATE_COLUMN in frame:
        dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
        invalid_dates = int(dates.isna().sum())
        if dates.notna().any():
            date_min, date_max = dates.min(), dates.max()
    return {
        "shape": list(frame.shape),
        "columns": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "sample_rows": frame.head(5).to_dict(orient="records"),
        "missing": {
            column: {
                "count": int(missing_count[column]),
                "percent": float(missing_count[column] / len(frame) * 100),
            }
            for column in frame.columns
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_load_ids": (
            int(frame[ID_COLUMN].duplicated().sum()) if ID_COLUMN in frame else None
        ),
        "numeric_descriptive_statistics": (
            numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            .to_dict()
            if not numeric.empty else {}
        ),
        "numeric_parse_failures_excluding_missing": {
            column: int((numeric[column].notna() & pd.to_numeric(numeric[column], errors="coerce").isna()).sum())
            for column in numeric.columns
        },
        "numeric_infinite_values": {
            column: int(np.isinf(pd.to_numeric(numeric[column], errors="coerce")).sum())
            for column in numeric.columns
        },
        "categorical_cardinality": {
            column: int(categorical[column].nunique(dropna=True))
            for column in categorical.columns
        },
        "date_min": date_min,
        "date_max": date_max,
        "invalid_dates": invalid_dates,
    }


def top_frequency(series: pd.Series, limit: int = 20) -> list[dict[str, Any]]:
    counts = series.value_counts(dropna=False).head(limit)
    return [
        {"value": str(index), "count": int(count), "percent": float(count / len(series) * 100)}
        for index, count in counts.items()
    ]


def train_audit(train: pd.DataFrame) -> dict[str, Any]:
    """Audit target behavior and freight-specific relationships."""
    target = train[TARGET]
    dates = pd.to_datetime(train[DATE_COLUMN], errors="coerce")
    route = train["pickup"].astype(str) + " -> " + train["delivery"].astype(str)
    rate_per_mile = target / train["distance"].replace(0, np.nan)
    q1, q3 = target.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    numeric_predictors = [column for column in NUMERIC_FEATURES if column in train]
    correlations = train[numeric_predictors + [TARGET]].corr(numeric_only=True)[TARGET]
    monthly = (
        train.assign(month=dates.dt.to_period("M").astype(str))
        .groupby("month")[TARGET]
        .agg(records="size", mean="mean", median="median", std="std", minimum="min", maximum="max")
        .reset_index()
    )
    by_equipment = (
        train.groupby("equipment")[TARGET]
        .agg(records="size", mean="mean", median="median", std="std", minimum="min", maximum="max")
        .reset_index()
    )
    return {
        "target_column": TARGET,
        "target_statistics": target.describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
        ).to_dict(),
        "target_quantiles": target.quantile(
            [0, 0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1]
        ).to_dict(),
        "target_skewness": float(target.skew()),
        "target_iqr_outliers": {
            "lower_fence": float(lower),
            "upper_fence": float(upper),
            "count": int(((target < lower) | (target > upper)).sum()),
            "percent": float(((target < lower) | (target > upper)).mean() * 100),
        },
        "non_positive_target_count": int((target <= 0).sum()),
        "numeric_pearson_correlations_with_target": correlations.sort_values(ascending=False).to_dict(),
        "posted_rate_vs_distance": {
            "pearson": float(target.corr(train["distance"], method="pearson")),
            # Pearson correlation of average ranks is Spearman's rho and avoids
            # making SciPy an undeclared runtime dependency.
            "spearman": float(target.rank().corr(train["distance"].rank(), method="pearson")),
        },
        "rate_per_mile_statistics": rate_per_mile.describe(
            percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        ).to_dict(),
        "zero_or_negative_distance_count": int((train["distance"] <= 0).sum()),
        "zero_or_negative_weight_count": int((train["weight"] <= 0).sum()),
        "target_by_equipment": by_equipment.to_dict(orient="records"),
        "records_by_month": monthly.to_dict(orient="records"),
        "pickup_frequency_top20": top_frequency(train["pickup"]),
        "delivery_frequency_top20": top_frequency(train["delivery"]),
        "route_frequency_top20": top_frequency(route),
        "equipment_distribution": top_frequency(train["equipment"], limit=100),
        "unique_routes": int(route.nunique()),
    }


def validation_comparison(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    """Compare schemas, category coverage, missingness, and numeric shifts."""
    train_routes = set(zip(train["pickup"], train["delivery"]))
    validation_routes = list(zip(validation["pickup"], validation["delivery"]))

    def unseen(column: str) -> dict[str, Any]:
        known = set(train[column].dropna())
        mask = ~validation[column].isin(known)
        values = sorted(validation.loc[mask, column].dropna().astype(str).unique().tolist())
        return {
            "unique_count": len(values),
            "row_count": int(mask.sum()),
            "row_percent": float(mask.mean() * 100),
            "values": values,
        }

    route_mask = pd.Series([route not in train_routes for route in validation_routes])
    unseen_route_values = sorted({f"{a} -> {b}" for (a, b), flag in zip(validation_routes, route_mask) if flag})
    known_pickups = set(train["pickup"].dropna())
    known_deliveries = set(train["delivery"].dropna())
    unseen_endpoint_mask = pd.Series([
        pickup not in known_pickups or delivery not in known_deliveries
        for pickup, delivery in validation_routes
    ])
    novel_known_endpoint_route_mask = route_mask & ~unseen_endpoint_mask
    numeric_shift: dict[str, Any] = {}
    for column in NUMERIC_FEATURES:
        train_values, validation_values = train[column], validation[column]
        pooled_scale = train_values.std(ddof=0)
        numeric_shift[column] = {
            "train_mean": float(train_values.mean()),
            "validation_mean": float(validation_values.mean()),
            "train_median": float(train_values.median()),
            "validation_median": float(validation_values.median()),
            "train_std": float(train_values.std(ddof=0)),
            "validation_std": float(validation_values.std(ddof=0)),
            "standardized_mean_difference_vs_train_std": (
                float((validation_values.mean() - train_values.mean()) / pooled_scale)
                if pooled_scale else None
            ),
            "train_p05": float(train_values.quantile(0.05)),
            "validation_p05": float(validation_values.quantile(0.05)),
            "train_p95": float(train_values.quantile(0.95)),
            "validation_p95": float(validation_values.quantile(0.95)),
        }
    return {
        "feature_schema_matches": train.drop(columns=[TARGET]).columns.tolist() == validation.columns.tolist(),
        "target_absent": TARGET not in validation.columns,
        "unseen_pickups": unseen("pickup"),
        "unseen_deliveries": unseen("delivery"),
        "unseen_equipment": unseen("equipment"),
        "unseen_routes": {
            "unique_count": len(unseen_route_values),
            "row_count": int(route_mask.sum()),
            "row_percent": float(route_mask.mean() * 100),
            "validation_unique_routes": int(len(set(validation_routes))),
            "rows_with_unseen_endpoint": int(unseen_endpoint_mask.sum()),
            "rows_with_known_endpoints_but_novel_pair": int(novel_known_endpoint_route_mask.sum()),
            "values": unseen_route_values,
        },
        "numeric_distribution_shift": numeric_shift,
        "missing_percentage_point_difference": {
            column: float(validation[column].isna().mean() * 100 - train[column].isna().mean() * 100)
            for column in validation.columns
        },
    }


def december_audit(train: pd.DataFrame, validation: pd.DataFrame, december: pd.DataFrame) -> dict[str, Any]:
    """Describe fixed-input structure and unavailable model predictors."""
    model_features = [column for column in train.columns if column not in {ID_COLUMN, TARGET}]
    december_inputs = [column for column in december.columns if column != "predicted_rate"]
    common = [column for column in model_features if column in december_inputs]
    missing = [column for column in model_features if column not in december_inputs]
    non_date = [column for column in december_inputs if column != DATE_COLUMN]
    pickup = december["pickup"].iloc[0]
    delivery = december["delivery"].iloc[0]
    train_route = train[(train["pickup"] == pickup) & (train["delivery"] == delivery)]
    validation_route = validation[(validation["pickup"] == pickup) & (validation["delivery"] == delivery)]
    return {
        "row_count": len(december),
        "available_model_features": common,
        "missing_model_features": missing,
        "extra_columns_vs_model_features": [
            column for column in december.columns if column not in model_features
        ],
        "non_date_nunique": december[non_date].nunique(dropna=False).to_dict(),
        "all_non_date_inputs_constant": bool((december[non_date].nunique(dropna=False) == 1).all()),
        "only_date_changes_among_inputs": bool((december[non_date].nunique(dropna=False) == 1).all()),
        "dates_unique": bool(december[DATE_COLUMN].nunique() == len(december)),
        "route_seen_in_train": bool(
            ((train["pickup"] == december["pickup"].iloc[0]) &
             (train["delivery"] == december["delivery"].iloc[0])).any()
        ),
        "route_seen_in_validation": bool(
            ((validation["pickup"] == december["pickup"].iloc[0]) &
             (validation["delivery"] == december["delivery"].iloc[0])).any()
        ),
        "train_route_row_count": int(len(train_route)),
        "validation_route_row_count": int(len(validation_route)),
        "train_route_target_statistics": (
            train_route[TARGET].describe().to_dict() if not train_route.empty else {}
        ),
        "pickup_seen_in_train": bool(december["pickup"].iloc[0] in set(train["pickup"])),
        "delivery_seen_in_train": bool(december["delivery"].iloc[0] in set(train["delivery"])),
        "equipment_seen_in_train": bool(december["equipment"].iloc[0] in set(train["equipment"])),
    }


def save_figures(train: pd.DataFrame, figure_dir: Path) -> None:
    """Generate a small, decision-focused set of EDA figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    color = "#0B6673"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    axes[0].hist(train[TARGET], bins=60, color=color, alpha=0.9)
    axes[0].set(title="Posted-rate distribution", xlabel="Posted rate ($)", ylabel="Loads")
    axes[1].hist(np.log1p(train[TARGET]), bins=60, color=color, alpha=0.9)
    axes[1].set(title="Log1p posted-rate distribution", xlabel="log1p(posted rate)", ylabel="Loads")
    fig.tight_layout()
    fig.savefig(figure_dir / "target_distributions.png", bbox_inches="tight")
    plt.close(fig)

    sample = train.sample(n=min(12_000, len(train)), random_state=42)
    fig, ax = plt.subplots(figsize=(7.6, 5), dpi=150)
    ax.scatter(sample["distance"], sample[TARGET], s=7, alpha=0.18, color=color, edgecolors="none")
    ax.set(title="Posted rate vs. distance (deterministic sample)", xlabel="Distance (miles)", ylabel="Posted rate ($)")
    fig.tight_layout()
    fig.savefig(figure_dir / "posted_rate_vs_distance.png", bbox_inches="tight")
    plt.close(fig)

    rate_per_mile = train[TARGET] / train["distance"].replace(0, np.nan)
    upper = rate_per_mile.quantile(0.99)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    ax.hist(rate_per_mile.clip(upper=upper), bins=60, color=color, alpha=0.9)
    ax.set(title="Rate per mile (capped at 99th percentile for display)", xlabel="Posted rate / distance ($/mile)", ylabel="Loads")
    fig.tight_layout()
    fig.savefig(figure_dir / "rate_per_mile.png", bbox_inches="tight")
    plt.close(fig)

    dated = train.assign(month=pd.to_datetime(train[DATE_COLUMN]).dt.to_period("M").dt.to_timestamp())
    monthly = dated.groupby("month")[TARGET].agg(["mean", "median"])
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    ax.plot(monthly.index, monthly["mean"], marker="o", label="Mean", color=color)
    ax.plot(monthly.index, monthly["median"], marker="o", label="Median", color="#D95F59")
    ax.set(title="Monthly posted-rate statistics", xlabel="Month", ylabel="Posted rate ($)")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_dir / "monthly_target_statistics.png", bbox_inches="tight")
    plt.close(fig)

    equipment_order = train.groupby("equipment")[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    data = [train.loc[train["equipment"] == item, TARGET].to_numpy() for item in equipment_order]
    ax.boxplot(data, tick_labels=equipment_order, showfliers=False)
    ax.set(title="Posted rate by equipment (outliers hidden for display)", xlabel="Equipment", ylabel="Posted rate ($)")
    fig.tight_layout()
    fig.savefig(figure_dir / "target_by_equipment.png", bbox_inches="tight")
    plt.close(fig)

    missing = train.isna().mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    ax.bar(missing.index, missing.values, color=color)
    ax.set(title="Training missingness by column", xlabel="Column", ylabel="Missing (%)")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    fig.savefig(figure_dir / "training_missingness.png", bbox_inches="tight")
    plt.close(fig)


def build_audit(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Build the complete audit payload without side effects."""
    train, validation = frames["train"], frames["validation"]
    return {
        "datasets": {label: frame_audit(frame) for label, frame in frames.items()},
        "training_analysis": train_audit(train),
        "validation_comparison": validation_comparison(train, validation),
        "december_analysis": december_audit(train, validation, frames["december"]),
    }


def print_summary(audit: dict[str, Any]) -> None:
    """Print concise findings while detailed results remain in JSON."""
    print("Spotter freight-rate Phase 1 data audit")
    for label, item in audit["datasets"].items():
        print(f"- {label}: {item['shape'][0]:,} rows x {item['shape'][1]} columns; dates {item['date_min']} to {item['date_max']}")
    train = audit["training_analysis"]
    compare = audit["validation_comparison"]
    december = audit["december_analysis"]
    print(f"- target: {train['target_column']}; skew={train['target_skewness']:.3f}; IQR outliers={train['target_iqr_outliers']['count']:,}")
    print(f"- unseen validation rows: pickup={compare['unseen_pickups']['row_count']:,}, delivery={compare['unseen_deliveries']['row_count']:,}, routes={compare['unseen_routes']['row_count']:,}, equipment={compare['unseen_equipment']['row_count']:,}")
    print(f"- December missing model predictors: {', '.join(december['missing_model_features'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Repository root; auto-detected by default")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_repository_root(args.root)
    frames = load_datasets(root)
    audit = build_audit(frames)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "data_audit.json"
    output.write_text(json.dumps(json_ready(audit), indent=2, sort_keys=True), encoding="utf-8")
    if not args.no_figures:
        save_figures(frames["train"], reports / "figures")
    print_summary(audit)
    print(f"- detailed audit: {output}")


if __name__ == "__main__":
    main()
