"""Triple-barrier target labeling for Stage 1.3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.engineering import compute_wilder_atr


@dataclass(frozen=True)
class HorizonConfig:
    """Triple-barrier settings loaded from config/horizon.yaml."""

    horizon_days: int
    tp_multiplier: float
    sl_multiplier: float
    atr_period: int


def load_horizon_config(path: Path) -> HorizonConfig:
    """Load Stage 1.2 horizon settings."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = payload.get("parameters", {})
    return HorizonConfig(
        horizon_days=int(payload["horizon_days"]),
        tp_multiplier=float(params["tp_multiplier"]),
        sl_multiplier=float(params["sl_multiplier"]),
        atr_period=int(params["atr_period"]),
    )


def build_target_labels(master_df: pd.DataFrame, config: HorizonConfig) -> pd.Series:
    """Generate long-only triple-barrier labels for every resolvable row."""
    required = {"us100_high", "us100_low", "us100_close"}
    missing = required.difference(master_df.columns)
    if missing:
        raise ValueError(f"Master dataset missing required label columns: {sorted(missing)}")
    if config.horizon_days < 1:
        raise ValueError("horizon_days must be positive")

    df = master_df.sort_index()
    high = df["us100_high"].astype(float)
    low = df["us100_low"].astype(float)
    close = df["us100_close"].astype(float)
    atr = compute_wilder_atr(high, low, close, period=config.atr_period)

    labels = pd.Series(np.nan, index=df.index, dtype="float64", name="target")
    last_entry_pos = len(df) - config.horizon_days - 1
    if last_entry_pos < 0:
        return labels

    for pos in range(0, last_entry_pos + 1):
        atr_value = atr.iloc[pos]
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue
        labels.iloc[pos] = label_one_entry(
            high=high,
            low=low,
            close=close,
            entry_pos=pos,
            atr_value=float(atr_value),
            horizon_days=config.horizon_days,
            tp_multiplier=config.tp_multiplier,
            sl_multiplier=config.sl_multiplier,
        )
    return labels


def label_one_entry(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    entry_pos: int,
    atr_value: float,
    horizon_days: int,
    tp_multiplier: float,
    sl_multiplier: float,
) -> int:
    """Label one long entry; same-day TP/SL breach is pessimistic."""
    if entry_pos < 0 or entry_pos >= len(close):
        raise ValueError("entry_pos out of bounds")
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if not np.isfinite(atr_value) or atr_value <= 0:
        raise ValueError("atr_value must be positive and finite")

    entry_price = float(close.iloc[entry_pos])
    tp_level = entry_price + (tp_multiplier * atr_value)
    sl_level = entry_price - (sl_multiplier * atr_value)
    max_pos = min(entry_pos + horizon_days, len(close) - 1)

    for future_pos in range(entry_pos + 1, max_pos + 1):
        hit_tp = float(high.iloc[future_pos]) >= tp_level
        hit_sl = float(low.iloc[future_pos]) <= sl_level
        if hit_sl:
            return 0
        if hit_tp:
            return 1
    return 0
