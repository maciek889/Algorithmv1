"""Verify Stage 1.4 walk-forward folds.

Usage:
    python scripts/verify_walk_forward.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.walk_forward import WalkForwardFold, WalkForwardSplitter  # noqa: E402

DEFAULT_DATA_PATH = REPO_ROOT / "data" / "processed" / "features_and_labels.parquet"
DEFAULT_HORIZON_PATH = REPO_ROOT / "config" / "horizon.yaml"


def _date(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")


def _format_fold_row(fold: WalkForwardFold) -> str:
    embargo_range = f"{_date(fold.embargo_start)} to {_date(fold.embargo_end)}"
    train_range = f"{_date(fold.train_start)} to {_date(fold.train_end)}"
    test_range = f"{_date(fold.test_start)} to {_date(fold.test_end)}"
    return (
        f"{fold.fold_id:>4}  "
        f"{train_range:<24}  "
        f"{embargo_range:<24}  "
        f"{test_range:<24}  "
        f"{len(fold.train_indices):>6}  "
        f"{len(fold.test_indices):>5}  "
        f"{fold.train_weights.min():>8.4f}  "
        f"{fold.train_weights.max():>8.4f}"
    )


def verify_walk_forward(
    data_path: Path = DEFAULT_DATA_PATH,
    horizon_path: Path = DEFAULT_HORIZON_PATH,
) -> list[WalkForwardFold]:
    """Load the dataset, build folds, and assert the embargo invariant."""
    df = pd.read_parquet(data_path)
    splitter = WalkForwardSplitter.from_horizon_config(horizon_path)
    folds = list(splitter.iter_folds(df))
    if not folds:
        raise AssertionError("Walk-forward splitter produced zero folds.")

    for fold in folds:
        actual_gap_days = (fold.test_start - fold.train_end).days
        assert actual_gap_days >= splitter.horizon_days, (
            f"Fold {fold.fold_id} embargo violation: "
            f"{actual_gap_days} days < {splitter.horizon_days} days"
        )
        assert len(fold.train_weights) == len(fold.train_indices)
        assert fold.train_weights.min() > 0
        assert fold.train_weights.max() <= 1.0

    print(f"Loaded {len(df):,} rows from {data_path}")
    print(f"Horizon embargo: {splitter.horizon_days} calendar days")
    print()
    print(
        "Fold  Train Date Range          Embargo Date Range       "
        "Test Date Range            Train    Test     W Min     W Max"
    )
    print("-" * 112)
    for fold in folds:
        print(_format_fold_row(fold))
    print("-" * 112)
    print(f"Total folds: {len(folds)}")
    return folds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--horizon", type=Path, default=DEFAULT_HORIZON_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_path = args.data if args.data.is_absolute() else REPO_ROOT / args.data
    horizon_path = args.horizon if args.horizon.is_absolute() else REPO_ROOT / args.horizon
    verify_walk_forward(data_path=data_path, horizon_path=horizon_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
