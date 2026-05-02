"""Assemble per-source frames into a single master dataset on the US100 calendar."""

from __future__ import annotations

import logging

import pandas as pd

from src.data.config import DataConfig
from src.data.exceptions import ValidationError

logger = logging.getLogger(__name__)

_FFILL_LIMIT_CONTEXT = 1
_FFILL_LIMIT_PUTCALL = 5


def build_master_dataset(
    yfinance_data: dict[str, pd.DataFrame],
    putcall: pd.Series,
    config: DataConfig,
) -> pd.DataFrame:
    """Build the master dataset anchored on US100 trading days.

    Args:
        yfinance_data: Mapping name -> OHLCV DataFrame. Must include 'us100', 'vix',
            'spx', 'dxy' (matching config.yfinance_tickers).
        putcall: Daily put/call ratio Series.
        config: DataConfig (uses training_start to gate NaN validation).

    Returns:
        DataFrame indexed by US100 trading days with columns:
            us100_open, us100_high, us100_low, us100_close, us100_volume,
            vix_open, vix_high, vix_low, vix_close,
            spx_close,
            dxy_close,
            putcall_ratio

    Raises:
        ValidationError: required source missing, or NaN in core columns post training_start.
    """
    _require(yfinance_data, ("us100", "vix", "spx", "dxy"))

    us100 = yfinance_data["us100"]
    anchor = us100.index

    out = pd.DataFrame(index=anchor)
    out["us100_open"]   = us100["open"]
    out["us100_high"]   = us100["high"]
    out["us100_low"]    = us100["low"]
    out["us100_close"]  = us100["close"]
    out["us100_volume"] = us100["volume"]

    vix = yfinance_data["vix"].reindex(anchor).ffill(limit=_FFILL_LIMIT_CONTEXT)
    out["vix_open"]  = vix["open"]
    out["vix_high"]  = vix["high"]
    out["vix_low"]   = vix["low"]
    out["vix_close"] = vix["close"]

    spx = yfinance_data["spx"].reindex(anchor).ffill(limit=_FFILL_LIMIT_CONTEXT)
    out["spx_close"] = spx["close"]

    dxy = yfinance_data["dxy"].reindex(anchor).ffill(limit=_FFILL_LIMIT_CONTEXT)
    out["dxy_close"] = dxy["close"]

    out["putcall_ratio"] = putcall.reindex(anchor).ffill(limit=_FFILL_LIMIT_PUTCALL)

    _validate_post_training_no_nan(out, config)

    logger.info("Assembled master dataset: %d rows, %s..%s",
                len(out), out.index.min().date(), out.index.max().date())
    return out


def _require(d: dict, keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValidationError(f"yfinance_data missing required series: {missing}")


def _validate_post_training_no_nan(df: pd.DataFrame, config: DataConfig) -> None:
    cutoff = pd.Timestamp(config.training_start)
    post = df.loc[df.index >= cutoff]
    core = ["us100_open", "us100_high", "us100_low", "us100_close", "us100_volume",
            "vix_close", "spx_close", "dxy_close"]
    nan_counts = post[core].isna().sum()
    bad = nan_counts[nan_counts > 0]
    if not bad.empty:
        raise ValidationError(
            f"NaN found in core columns after training_start ({config.training_start}): "
            f"{bad.to_dict()}"
        )
