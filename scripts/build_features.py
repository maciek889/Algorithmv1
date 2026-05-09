"""Build Stage 1.3 features and labels.

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.engineering import FEATURE_COLUMNS, build_feature_frame  # noqa: E402
from src.features.labeling import build_target_labels, load_horizon_config  # noqa: E402

MASTER_PATH = REPO_ROOT / "data" / "processed" / "master_dataset.parquet"
CALENDARS_DIR = REPO_ROOT / "data" / "calendars"
HORIZON_PATH = REPO_ROOT / "config" / "horizon.yaml"
OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "features_and_labels.parquet"


def build_features_and_labels(
    master_path: Path = MASTER_PATH,
    calendars_dir: Path = CALENDARS_DIR,
    horizon_path: Path = HORIZON_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Build, clean, and save the Stage 1.3 modeling dataset."""
    master_df = pd.read_parquet(master_path)
    horizon_config = load_horizon_config(horizon_path)

    features = build_feature_frame(master_df, calendars_dir)
    target = build_target_labels(master_df, horizon_config)
    dataset = features.join(target)

    dataset = dataset.iloc[: -horizon_config.horizon_days]
    dataset = dataset.dropna(subset=FEATURE_COLUMNS + ["target"])
    dataset["target"] = dataset["target"].astype("int64")
    dataset["day_of_week"] = dataset["day_of_week"].astype("int64")
    dataset["macro_3d_flag"] = dataset["macro_3d_flag"].astype("int64")
    dataset["ma50_slope_5d"] = dataset["ma50_slope_5d"].astype("int64")
    dataset["obv_direction_5d"] = dataset["obv_direction_5d"].astype("int64")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path)
    return dataset


def dataset_summary(dataset: pd.DataFrame) -> dict[str, object]:
    """Return headline diagnostics for the final architect summary."""
    class_counts = dataset["target"].value_counts().sort_index()
    class_balance = {
        int(label): {
            "count": int(count),
            "pct": float((count / len(dataset)) * 100.0),
        }
        for label, count in class_counts.items()
    }
    correlations = (
        dataset[FEATURE_COLUMNS]
        .corrwith(dataset["target"], method="pearson")
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(3)
    )
    return {
        "shape": dataset.shape,
        "class_balance": class_balance,
        "top_correlations": correlations,
    }


def main() -> int:
    args = _parse_args()
    dataset = build_features_and_labels(
        master_path=_resolve(args.master),
        calendars_dir=_resolve(args.calendars),
        horizon_path=_resolve(args.horizon),
        output_path=_resolve(args.output),
    )
    summary = dataset_summary(dataset)

    print(f"Final dataset: {summary['shape'][0]} rows x {summary['shape'][1]} columns")
    print("Target class balance:")
    for label in (0, 1):
        row = summary["class_balance"].get(label, {"count": 0, "pct": 0.0})
        print(f"  {label}: {row['count']} ({row['pct']:.4f}%)")
    print("Top 3 Pearson correlations with target:")
    for name, value in summary["top_correlations"].items():
        print(f"  {name}: {value:.6f}")
    print(f"Output: {_resolve(args.output)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", type=Path, default=MASTER_PATH)
    parser.add_argument("--calendars", type=Path, default=CALENDARS_DIR)
    parser.add_argument("--horizon", type=Path, default=HORIZON_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
