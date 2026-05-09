"""Stage 1.3 feature engineering for the US100 ML dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "price_vs_ma50",
    "price_vs_ma200",
    "ma50_slope_5d",
    "rsi_14",
    "relative_atr_14",
    "bb_position_20_2",
    "relative_volume_20",
    "obv_direction_5d",
    "vix_level",
    "vix_change_5d",
    "dxy_change_5d",
    "rel_strength_us100_spx_5d",
    "body_ratio",
    "close_position",
    "opening_gap",
    "day_of_week",
    "macro_3d_flag",
]


REQUIRED_MASTER_COLUMNS = {
    "us100_open",
    "us100_high",
    "us100_low",
    "us100_close",
    "us100_volume",
    "vix_close",
    "spx_close",
    "dxy_close",
}


def build_feature_frame(master_df: pd.DataFrame, calendars_dir: Path) -> pd.DataFrame:
    """Compute the exact Stage 1.3 feature set from the master dataset."""
    _validate_master(master_df)
    df = master_df.sort_index().copy()

    close = df["us100_close"].astype(float)
    high = df["us100_high"].astype(float)
    low = df["us100_low"].astype(float)
    open_ = df["us100_open"].astype(float)
    volume = df["us100_volume"].astype(float)

    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    atr14 = compute_wilder_atr(high, low, close, period=14)
    bb_lower, bb_upper = compute_bollinger_bands(close, window=20, num_std=2.0)
    price_range = high - low

    features = pd.DataFrame(index=df.index)
    features["price_vs_ma50"] = _safe_divide(close - ma50, ma50)
    features["price_vs_ma200"] = _safe_divide(close - ma200, ma200)
    features["ma50_slope_5d"] = (ma50 > ma50.shift(5)).astype("float64")
    features.loc[ma50.isna() | ma50.shift(5).isna(), "ma50_slope_5d"] = np.nan
    features["rsi_14"] = compute_rsi_wilder(close, period=14)
    features["relative_atr_14"] = _safe_divide(atr14, close)
    features["bb_position_20_2"] = compute_bollinger_position(close, bb_lower, bb_upper)
    features["relative_volume_20"] = _safe_divide(volume, volume.rolling(20, min_periods=20).mean())
    features["obv_direction_5d"] = compute_obv_direction(close, volume, lookback=5)
    features["vix_level"] = df["vix_close"].astype(float)
    features["vix_change_5d"] = df["vix_close"].astype(float).pct_change(5)
    features["dxy_change_5d"] = df["dxy_close"].astype(float).pct_change(5)
    features["rel_strength_us100_spx_5d"] = close.pct_change(5) - df["spx_close"].astype(float).pct_change(5)
    features["body_ratio"] = _safe_divide((close - open_).abs(), price_range)
    features["close_position"] = _safe_divide(close - low, price_range)
    features["opening_gap"] = _safe_divide(open_ - close.shift(1), close.shift(1))
    features["day_of_week"] = features.index.dayofweek.astype("float64")
    features["macro_3d_flag"] = compute_macro_3d_flag(features.index, calendars_dir).astype("float64")

    return features[FEATURE_COLUMNS]


def compute_wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Wilder ATR using data through each row."""
    if period < 1:
        raise ValueError("period must be positive")
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, and close must have equal length")

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
    if len(true_range) <= period:
        return atr

    atr.iloc[period] = true_range.iloc[1 : period + 1].mean()
    for pos in range(period + 1, len(true_range)):
        atr.iloc[pos] = ((atr.iloc[pos - 1] * (period - 1)) + true_range.iloc[pos]) / period
    return atr


def compute_rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI with Wilder smoothing."""
    if period < 1:
        raise ValueError("period must be positive")

    close = close.astype(float)
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    rsi = pd.Series(np.nan, index=close.index, dtype="float64", name="rsi")
    if len(close) <= period:
        return rsi

    avg_gain = gains.iloc[1 : period + 1].mean()
    avg_loss = losses.iloc[1 : period + 1].mean()
    rsi.iloc[period] = _rsi_from_avgs(avg_gain, avg_loss)

    for pos in range(period + 1, len(close)):
        avg_gain = ((avg_gain * (period - 1)) + gains.iloc[pos]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses.iloc[pos]) / period
        rsi.iloc[pos] = _rsi_from_avgs(avg_gain, avg_loss)

    return rsi


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Return lower and upper Bollinger bands."""
    close = close.astype(float)
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return mean - (num_std * std), mean + (num_std * std)


def compute_bollinger_position(
    close: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> pd.Series:
    """Compute clipped Bollinger position from 0 to 1."""
    position = _safe_divide(close.astype(float) - lower, upper - lower)
    return position.clip(lower=0.0, upper=1.0)


def compute_obv_direction(close: pd.Series, volume: pd.Series, lookback: int = 5) -> pd.Series:
    """Return 1 when OBV is higher than lookback days ago, else 0."""
    if lookback < 1:
        raise ValueError("lookback must be positive")

    close = close.astype(float)
    volume = volume.astype(float)
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    out = (obv > obv.shift(lookback)).astype("float64")
    out.loc[obv.shift(lookback).isna()] = np.nan
    return out


def compute_macro_3d_flag(index: pd.DatetimeIndex, calendars_dir: Path) -> pd.Series:
    """Flag if FOMC, CPI, or NFP occurs in the next 3 calendar days."""
    event_dates = _load_macro_event_dates(calendars_dir)
    if not event_dates:
        return pd.Series(0, index=index, dtype="int64", name="macro_3d_flag")

    event_days = set(pd.Timestamp(d).normalize() for d in event_dates)
    values = [
        int(any((day + pd.Timedelta(days=offset)).normalize() in event_days for offset in range(1, 4)))
        for day in pd.DatetimeIndex(index).normalize()
    ]
    return pd.Series(values, index=index, dtype="int64", name="macro_3d_flag")


def _load_macro_event_dates(calendars_dir: Path) -> list[pd.Timestamp]:
    events: list[pd.Timestamp] = []
    specs = [
        ("fomc_dates.parquet", "decision_date"),
        ("cpi_release_dates.parquet", "release_date"),
        ("nfp_release_dates.parquet", "release_date"),
    ]
    for file_name, column in specs:
        path = calendars_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Required calendar file not found: {path}")
        df = pd.read_parquet(path)
        if column not in df.columns:
            raise ValueError(f"Calendar {path} missing required column {column!r}")
        events.extend(pd.to_datetime(df[column], errors="coerce").dropna().tolist())
    return events


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.astype(float)
    result = numerator.astype(float) / denominator.replace(0.0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _rsi_from_avgs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _validate_master(master_df: pd.DataFrame) -> None:
    missing = REQUIRED_MASTER_COLUMNS.difference(master_df.columns)
    if missing:
        raise ValueError(f"Master dataset missing required columns: {sorted(missing)}")
    if master_df.empty:
        raise ValueError("Master dataset is empty")
    if not isinstance(master_df.index, pd.DatetimeIndex):
        raise ValueError("Master dataset index must be a DatetimeIndex")
