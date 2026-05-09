"""Evaluate Stage 1.5 dummy baselines through walk-forward folds.

Usage:
    python scripts/evaluate_baselines.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.walk_forward import WalkForwardFold, WalkForwardSplitter  # noqa: E402

DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "features_and_labels.parquet"
DEFAULT_HORIZON_PATH = REPO_ROOT / "config" / "horizon.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "reports" / "baseline_metrics.json"
DEFAULT_RANDOM_SEED = 42
TARGET_COLUMN = "target"
MA50_COLUMNS = ("price_vs_ma50", "ma50_slope_5d")


class Baseline(Protocol):
    """Minimal prediction protocol for static baselines."""

    name: str

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return integer class predictions for X."""


@dataclass
class RandomPriorBaseline:
    """Randomly predict class 1 according to the observed class prior."""

    class_one_probability: float
    seed: int = DEFAULT_RANDOM_SEED
    name: str = "random_baseline"

    def __post_init__(self) -> None:
        if not 0.0 <= self.class_one_probability <= 1.0:
            raise ValueError("class_one_probability must be in [0, 1].")
        self._rng = np.random.default_rng(self.seed)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._rng.binomial(1, self.class_one_probability, size=len(X)).astype(int)


@dataclass(frozen=True)
class MA50TrendBaseline:
    """Take trades only when price is above MA50 and MA50 is rising."""

    name: str = "ma50_baseline"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        missing = set(MA50_COLUMNS).difference(X.columns)
        if missing:
            raise ValueError(f"MA50 baseline missing required columns: {sorted(missing)}")
        return ((X["price_vs_ma50"] > 0.0) & (X["ma50_slope_5d"] == 1)).astype(int).to_numpy()


def evaluate_baselines(
    df: pd.DataFrame,
    splitter: WalkForwardSplitter,
    baselines: list[Baseline],
) -> dict[str, object]:
    """Evaluate baselines on each walk-forward test fold and aggregate metrics."""
    _validate_dataset(df)
    folds = list(splitter.iter_folds(df))
    if not folds:
        raise ValueError("Walk-forward splitter produced zero folds.")

    baseline_results = {baseline.name: [] for baseline in baselines}

    for fold in folds:
        test_df = df.iloc[fold.test_indices]
        y_true = test_df[TARGET_COLUMN].astype(int).to_numpy()
        for baseline in baselines:
            y_pred = baseline.predict(test_df)
            baseline_results[baseline.name].append(_fold_metrics(fold, y_true, y_pred))

    return {
        "fold_count": len(folds),
        "folds": [_fold_metadata(fold) for fold in folds],
        "baselines": {
            name: {
                "aggregate": _aggregate_metrics(rows),
                "per_fold": rows,
            }
            for name, rows in baseline_results.items()
        },
    }


def build_report(
    data_path: Path = DEFAULT_DATA_PATH,
    horizon_path: Path = DEFAULT_HORIZON_PATH,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    """Load inputs and return the full baseline report payload."""
    df = pd.read_parquet(data_path)
    _validate_dataset(df)
    class_one_prior = float(df[TARGET_COLUMN].mean())
    splitter = WalkForwardSplitter.from_horizon_config(horizon_path)
    results = evaluate_baselines(
        df=df,
        splitter=splitter,
        baselines=[
            RandomPriorBaseline(class_one_probability=class_one_prior, seed=random_seed),
            MA50TrendBaseline(),
        ],
    )
    return {
        "stage": "1.5_establish_baselines",
        "data_path": str(data_path),
        "horizon_path": str(horizon_path),
        "row_count": int(len(df)),
        "target_class_one_prior": class_one_prior,
        "random_seed": int(random_seed),
        "horizon_days": int(splitter.horizon_days),
        **results,
    }


def save_report(report: dict[str, object], output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Persist report as deterministic, readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fold_metrics(fold: WalkForwardFold, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    if len(y_pred) != len(y_true):
        raise ValueError("Prediction length must match test set length.")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0,
    )
    return {
        "fold_id": int(fold.fold_id),
        "test_start": _date(fold.test_start),
        "test_end": _date(fold.test_end),
        "test_observations": int(len(y_true)),
        "actual_class_one_count": int(y_true.sum()),
        "predicted_class_one_count": int(y_pred.sum()),
        "precision_class_1": float(precision[0]),
        "recall_class_1": float(recall[0]),
        "f1_class_1": float(f1[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metric_names = ("precision_class_1", "recall_class_1", "f1_class_1", "accuracy")
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows])),
            "std": float(np.std([row[metric] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for metric in metric_names
    }


def _fold_metadata(fold: WalkForwardFold) -> dict[str, object]:
    return {
        "fold_id": int(fold.fold_id),
        "train_start": _date(fold.train_start),
        "train_end": _date(fold.train_end),
        "embargo_start": _date(fold.embargo_start),
        "embargo_end": _date(fold.embargo_end),
        "test_start": _date(fold.test_start),
        "test_end": _date(fold.test_end),
        "train_observations": int(len(fold.train_indices)),
        "test_observations": int(len(fold.test_indices)),
    }


def _validate_dataset(df: pd.DataFrame) -> None:
    required = {TARGET_COLUMN, *MA50_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")


def _format_metric(metric: dict[str, float]) -> str:
    return f"{metric['mean']:.4f} +/- {metric['std']:.4f}"


def _print_summary(report: dict[str, object]) -> None:
    baselines = report["baselines"]
    random_metrics = baselines["random_baseline"]["aggregate"]
    ma50_metrics = baselines["ma50_baseline"]["aggregate"]
    print("# Stage 1.5 Baseline Results")
    print()
    print(f"- Walk-forward folds: {report['fold_count']}")
    print(f"- Class-1 prior used by random baseline: {report['target_class_one_prior']:.6f}")
    print()
    print("| Baseline | Precision Class 1 | Recall Class 1 | F1 Class 1 |")
    print("|---|---:|---:|---:|")
    print(
        "| Random Baseline | "
        f"{_format_metric(random_metrics['precision_class_1'])} | "
        f"{_format_metric(random_metrics['recall_class_1'])} | "
        f"{_format_metric(random_metrics['f1_class_1'])} |"
    )
    print(
        "| MA50 Baseline | "
        f"{_format_metric(ma50_metrics['precision_class_1'])} | "
        f"{_format_metric(ma50_metrics['recall_class_1'])} | "
        f"{_format_metric(ma50_metrics['f1_class_1'])} |"
    )


def _date(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--horizon", type=Path, default=DEFAULT_HORIZON_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(
        data_path=_resolve(args.data),
        horizon_path=_resolve(args.horizon),
        random_seed=args.random_seed,
    )
    save_report(report, _resolve(args.output))
    _print_summary(report)
    print()
    print(f"Report written to: {_resolve(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
