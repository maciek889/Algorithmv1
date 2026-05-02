"""Calibrate the triple-barrier horizon from empirical barrier-hit timing.

Usage:
    python scripts/calibrate_horizon.py
    python scripts/calibrate_horizon.py --output config/horizon.yaml

Reads ``data/processed/master_dataset.parquet``, simulates static triple
barriers on 2010-2011 entries, and writes ``config/horizon.yaml``.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.exceptions import ValidationError  # noqa: E402

LOGGER = logging.getLogger(__name__)

MASTER_DATASET_PATH = REPO_ROOT / "data" / "processed" / "master_dataset.parquet"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "config" / "horizon.yaml"

CALIBRATION_START = "2010-01-01"
CALIBRATION_END = "2011-12-31"
ATR_PERIOD = 14
TP_MULTIPLIER = 3.0
SL_MULTIPLIER = 1.5
MAX_LOOKOUT = 60
PERCENTILE = 75.0


class CalibrationError(ValidationError):
    """Raised when horizon calibration cannot produce a valid result."""


def compute_atr_wilder(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
) -> pd.Series:
    """Compute classic Wilder ATR.

    This helper is used only for Stage 1.2 horizon calibration, not as a
    production feature-engineering function.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Wilder ATR lookback period.

    Returns:
        ATR series with the first ``period`` rows as ``NaN``.

    Raises:
        CalibrationError: If inputs are malformed.
    """
    if period < 1:
        raise CalibrationError("ATR period must be positive.")
    if not (len(high) == len(low) == len(close)):
        raise CalibrationError("High, low, and close series must have equal length.")
    if len(close) <= period:
        return pd.Series(np.nan, index=close.index, name="atr")

    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = pd.Series(np.nan, index=close.index, dtype="float64", name="atr")
    atr.iloc[period] = true_range.iloc[1 : period + 1].mean()
    for pos in range(period + 1, len(true_range)):
        atr.iloc[pos] = ((atr.iloc[pos - 1] * (period - 1)) + true_range.iloc[pos]) / period

    return atr


def simulate_triple_barrier(
    df: pd.DataFrame,
    entry_idx: int,
    atr_value: float,
    tp_multiplier: float = TP_MULTIPLIER,
    sl_multiplier: float = SL_MULTIPLIER,
    max_lookout: int = MAX_LOOKOUT,
) -> dict[str, Any]:
    """Simulate one long position with static ATR-based barriers.

    Args:
        df: DataFrame with ``high``, ``low``, and ``close`` columns.
        entry_idx: Integer row index of the entry day.
        atr_value: ATR known at entry, computed without using entry-day data.
        tp_multiplier: Take-profit ATR multiplier.
        sl_multiplier: Stop-loss ATR multiplier.
        max_lookout: Maximum number of future trading days to inspect.

    Returns:
        Dictionary with entry metadata, first barrier, and time-to-hit.

    Raises:
        CalibrationError: If the input frame or parameters are invalid.
    """
    required_columns = {"high", "low", "close"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise CalibrationError(f"Missing required simulation columns: {sorted(missing)}")
    if entry_idx < 0 or entry_idx >= len(df):
        raise CalibrationError(f"entry_idx out of bounds: {entry_idx}")
    if not np.isfinite(atr_value) or atr_value <= 0:
        raise CalibrationError(f"ATR must be a positive finite value, got {atr_value!r}.")
    if max_lookout < 1:
        raise CalibrationError("max_lookout must be positive.")

    entry_row = df.iloc[entry_idx]
    entry_price = float(entry_row["close"])
    tp_level = entry_price + (tp_multiplier * float(atr_value))
    sl_level = entry_price - (sl_multiplier * float(atr_value))

    max_k = min(max_lookout, len(df) - entry_idx - 1)
    for k in range(1, max_k + 1):
        row = df.iloc[entry_idx + k]
        # TP-first is the documented same-day convention for calibration.
        if float(row["high"]) >= tp_level:
            return {
                "entry_date": pd.Timestamp(df.index[entry_idx]),
                "entry_price": entry_price,
                "atr": float(atr_value),
                "tp_level": float(tp_level),
                "sl_level": float(sl_level),
                "first_barrier": "tp",
                "time_to_first_barrier": k,
                "no_barrier_hit": False,
            }
        if float(row["low"]) <= sl_level:
            return {
                "entry_date": pd.Timestamp(df.index[entry_idx]),
                "entry_price": entry_price,
                "atr": float(atr_value),
                "tp_level": float(tp_level),
                "sl_level": float(sl_level),
                "first_barrier": "sl",
                "time_to_first_barrier": k,
                "no_barrier_hit": False,
            }

    return {
        "entry_date": pd.Timestamp(df.index[entry_idx]),
        "entry_price": entry_price,
        "atr": float(atr_value),
        "tp_level": float(tp_level),
        "sl_level": float(sl_level),
        "first_barrier": "none",
        "time_to_first_barrier": None,
        "no_barrier_hit": True,
    }


def calibrate_horizon(
    master_df: pd.DataFrame,
    calibration_start: str = CALIBRATION_START,
    calibration_end: str = CALIBRATION_END,
    percentile: float = PERCENTILE,
    max_lookout: int = MAX_LOOKOUT,
) -> dict[str, Any]:
    """Calibrate horizon from 2010-2011 time-to-first-barrier distribution.

    Args:
        master_df: Master dataset with US100 OHLC columns.
        calibration_start: First entry date included in calibration.
        calibration_end: Last entry date included in calibration.
        percentile: Percentile used as the horizon.
        max_lookout: Maximum number of future trading days per simulation.

    Returns:
        Calibration diagnostics and ``horizon_days``. Includes
        ``simulation_results`` for notebooks; YAML export omits this key.

    Raises:
        CalibrationError: If required data is missing or diagnostics fail.
    """
    required_columns = {"us100_high", "us100_low", "us100_close"}
    missing = required_columns.difference(master_df.columns)
    if missing:
        raise CalibrationError(f"Missing required master columns: {sorted(missing)}")
    if master_df.empty:
        raise CalibrationError("Master dataset is empty.")
    if not isinstance(master_df.index, pd.DatetimeIndex):
        raise CalibrationError("Master dataset index must be a DatetimeIndex.")

    df = master_df.sort_index().copy()
    atr = compute_atr_wilder(
        df["us100_high"],
        df["us100_low"],
        df["us100_close"],
        period=ATR_PERIOD,
    )
    atr_for_entry = atr.shift(1)

    calibration_mask = (df.index >= pd.Timestamp(calibration_start)) & (
        df.index <= pd.Timestamp(calibration_end)
    )
    entry_positions = np.flatnonzero(calibration_mask)
    if len(entry_positions) == 0:
        raise CalibrationError("No rows found in calibration window.")

    calibration_atr = atr_for_entry.iloc[entry_positions]
    if calibration_atr.isna().any():
        first_bad = calibration_atr[calibration_atr.isna()].index[0]
        raise CalibrationError(f"ATR warm-up produced NaN inside calibration at {first_bad.date()}.")
    if int(entry_positions[-1]) + max_lookout >= len(df):
        raise CalibrationError("Master dataset does not include the required lookout buffer.")

    simulation_df = pd.DataFrame(
        {
            "high": df["us100_high"],
            "low": df["us100_low"],
            "close": df["us100_close"],
        },
        index=df.index,
    )

    simulations = [
        simulate_triple_barrier(
            simulation_df,
            int(entry_idx),
            float(atr_for_entry.iloc[int(entry_idx)]),
            tp_multiplier=TP_MULTIPLIER,
            sl_multiplier=SL_MULTIPLIER,
            max_lookout=max_lookout,
        )
        for entry_idx in entry_positions
    ]

    hit_simulations = [row for row in simulations if not row["no_barrier_hit"]]
    no_hit_count = len(simulations) - len(hit_simulations)
    if not hit_simulations:
        raise CalibrationError("No barrier hits found in calibration window.")

    hit_times = np.array([row["time_to_first_barrier"] for row in hit_simulations], dtype=float)
    raw_percentile = float(np.percentile(hit_times, percentile))
    horizon_days = int(math.ceil(raw_percentile))

    tp_count = sum(1 for row in hit_simulations if row["first_barrier"] == "tp")
    sl_count = sum(1 for row in hit_simulations if row["first_barrier"] == "sl")
    n_hits = len(hit_simulations)

    result: dict[str, Any] = {
        "horizon_days": horizon_days,
        "percentile": float(percentile),
        "n_simulations": len(simulations),
        "n_barrier_hit": n_hits,
        "n_no_barrier_hit": no_hit_count,
        "pct_no_barrier_hit": round((no_hit_count / len(simulations)) * 100.0, 4),
        "distribution_stats": {
            "mean": round(float(np.mean(hit_times)), 4),
            "median": round(float(np.median(hit_times)), 4),
            "std": round(float(np.std(hit_times, ddof=1)), 4) if n_hits > 1 else 0.0,
            "p25": round(float(np.percentile(hit_times, 25.0)), 4),
            "p50": round(float(np.percentile(hit_times, 50.0)), 4),
            "p75": round(raw_percentile, 4),
            "p90": round(float(np.percentile(hit_times, 90.0)), 4),
            "min": int(np.min(hit_times)),
            "max": int(np.max(hit_times)),
        },
        "tp_vs_sl_split": {
            "pct_hit_tp_first": round((tp_count / n_hits) * 100.0, 4),
            "pct_hit_sl_first": round((sl_count / n_hits) * 100.0, 4),
        },
        "calibration_window": f"{calibration_start} to {calibration_end}",
        "simulation_results": simulations,
    }
    return result


def write_horizon_yaml(result: dict[str, Any], output_path: Path) -> None:
    """Write the calibrated horizon YAML file."""
    payload = {
        "horizon_days": result["horizon_days"],
        "calibration": {
            "window": result["calibration_window"],
            "percentile": result["percentile"],
            "n_simulations": result["n_simulations"],
            "n_barrier_hit": result["n_barrier_hit"],
            "pct_no_barrier_hit": result["pct_no_barrier_hit"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "distribution": {
            "mean": result["distribution_stats"]["mean"],
            "median": result["distribution_stats"]["median"],
            "p25": result["distribution_stats"]["p25"],
            "p50": result["distribution_stats"]["p50"],
            "p75": result["distribution_stats"]["p75"],
            "p90": result["distribution_stats"]["p90"],
        },
        "tp_vs_sl_split": result["tp_vs_sl_split"],
        "parameters": {
            "tp_multiplier": TP_MULTIPLIER,
            "sl_multiplier": SL_MULTIPLIER,
            "atr_period": ATR_PERIOD,
            "max_lookout": MAX_LOOKOUT,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Empirically calibrated horizon for triple barrier method.\n"
        "# Generated by scripts/calibrate_horizon.py\n"
        "# DO NOT EDIT MANUALLY.\n\n"
    )
    output_path.write_text(
        header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--master",
        type=Path,
        default=MASTER_DATASET_PATH,
        help="Path to master_dataset.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for horizon.yaml output.",
    )
    return parser.parse_args()


def main() -> int:
    """Run horizon calibration from the command line."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    master_path = args.master if args.master.is_absolute() else REPO_ROOT / args.master
    output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    LOGGER.info("Reading master dataset from %s", master_path)
    master_df = pd.read_parquet(master_path)
    result = calibrate_horizon(master_df)
    write_horizon_yaml(result, output_path)

    LOGGER.info(
        "Calibrated horizon=%s days from %s simulations (%s%% no barrier hit).",
        result["horizon_days"],
        result["n_simulations"],
        result["pct_no_barrier_hit"],
    )
    LOGGER.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
