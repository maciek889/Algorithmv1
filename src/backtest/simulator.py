"""Trade simulator: Triple Barrier resolution with realistic CFD costs.

The simulator receives a sequence of ML entry signals and resolves each trade
against future OHLC data using the Triple Barrier exit logic defined in
config/horizon.yaml.

Cost Model (US100 CFD):
  - Spread: 1.5 index points on entry (half-spread applied at open)
  - Overnight swap: −0.03 index points per night the position is held
    (approximation of typical retail CFD broker overnight financing)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Cost parameters for US100 CFD ────────────────────────────────────────────
SPREAD_POINTS = 1.5        # full spread cost applied on entry
SWAP_PER_NIGHT = -0.03     # points per contract per night held


@dataclass
class TradeResult:
    """One resolved trade."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float          # Close of signal day (before spread)
    effective_entry: float      # entry_price + half-spread
    exit_price: float           # resolved exit price
    exit_reason: str            # 'tp' | 'sl' | 'time'
    holding_days: int
    pnl_points: float           # raw P&L in index points (after costs)
    spread_cost: float          # spread paid in points
    swap_cost: float            # cumulative overnight cost in points
    atr_at_entry: float
    tp_level: float
    sl_level: float


@dataclass
class PortfolioState:
    """Tracks equity, position sizing and trade accounting."""

    initial_capital: float = 10_000.0
    risk_per_trade_pct: float = 0.01     # risk 1% of equity per trade
    point_value: float = 1.0             # $ per point per contract (CFD)
    equity: float = field(init=False)
    trades: list[TradeResult] = field(default_factory=list, init=False)
    equity_curve: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.equity = self.initial_capital


def resolve_trade(
    master_df: pd.DataFrame,
    entry_date: pd.Timestamp,
    atr_value: float,
    horizon_days: int,
    tp_multiplier: float,
    sl_multiplier: float,
) -> TradeResult | None:
    """Resolve a single long trade using the Triple Barrier Method.

    Uses future High/Low prices from master_dataset.  Returns None if the
    trade cannot be resolved (e.g. not enough future data).

    Pessimistic rule: if both TP and SL are breached on the same day,
    record as SL hit.
    """
    if atr_value <= 0 or not np.isfinite(atr_value):
        return None

    idx = master_df.index
    try:
        entry_pos = idx.get_loc(entry_date)
    except KeyError:
        return None

    entry_close = float(master_df.iloc[entry_pos]["us100_close"])
    effective_entry = entry_close + (SPREAD_POINTS / 2.0)  # half-spread on entry

    tp_level = effective_entry + (tp_multiplier * atr_value)
    sl_level = effective_entry - (sl_multiplier * atr_value)

    max_pos = min(entry_pos + horizon_days, len(master_df) - 1)

    if entry_pos + 1 > max_pos:
        return None

    exit_price = None
    exit_reason = None
    exit_pos = None

    for future_pos in range(entry_pos + 1, max_pos + 1):
        day_high = float(master_df.iloc[future_pos]["us100_high"])
        day_low = float(master_df.iloc[future_pos]["us100_low"])

        hit_sl = day_low <= sl_level
        hit_tp = day_high >= tp_level

        if hit_sl:
            # Pessimistic: SL first if both hit
            exit_price = sl_level
            exit_reason = "sl"
            exit_pos = future_pos
            break
        if hit_tp:
            exit_price = tp_level
            exit_reason = "tp"
            exit_pos = future_pos
            break

    if exit_price is None:
        # Time exit at the close of horizon day
        exit_pos = max_pos
        exit_price = float(master_df.iloc[exit_pos]["us100_close"])
        exit_reason = "time"

    exit_date = master_df.index[exit_pos]
    holding_days = exit_pos - entry_pos

    spread_cost = SPREAD_POINTS / 2.0  # half-spread at exit too
    swap_cost = abs(SWAP_PER_NIGHT) * holding_days
    raw_pnl = exit_price - effective_entry
    net_pnl = raw_pnl - spread_cost - swap_cost

    return TradeResult(
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_close,
        effective_entry=effective_entry,
        exit_price=exit_price,
        exit_reason=exit_reason,
        holding_days=holding_days,
        pnl_points=net_pnl,
        spread_cost=SPREAD_POINTS / 2.0 + spread_cost,  # total spread
        swap_cost=swap_cost,
        atr_at_entry=atr_value,
        tp_level=tp_level,
        sl_level=sl_level,
    )


def run_portfolio_simulation(
    trades: list[TradeResult],
    initial_capital: float = 10_000.0,
    risk_per_trade_pct: float = 0.01,
    point_value: float = 1.0,
) -> dict[str, Any]:
    """Simulate portfolio equity progression using fixed fractional sizing.

    Position sizing: risk ``risk_per_trade_pct`` of current equity on each
    trade's SL distance.  The number of "contracts" is:

        contracts = (equity × risk_pct) / (SL_distance × point_value)

    This ensures no single trade risks more than 1% of account equity.
    """
    equity = initial_capital
    peak_equity = equity
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    equity_curve: list[dict[str, Any]] = []
    gross_profit = 0.0
    gross_loss = 0.0

    for trade in trades:
        sl_distance = trade.effective_entry - trade.sl_level
        if sl_distance <= 0:
            continue

        # Position sizing: fixed fractional risk
        risk_amount = equity * risk_per_trade_pct
        contracts = risk_amount / (sl_distance * point_value)

        # Dollar P&L for this trade
        dollar_pnl = trade.pnl_points * contracts * point_value

        equity += dollar_pnl

        if dollar_pnl > 0:
            gross_profit += dollar_pnl
        else:
            gross_loss += abs(dollar_pnl)

        # Track peak and drawdown
        if equity > peak_equity:
            peak_equity = equity
        drawdown = peak_equity - equity
        drawdown_pct = drawdown / peak_equity if peak_equity > 0 else 0.0
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct
            max_drawdown = drawdown

        equity_curve.append({
            "date": trade.exit_date.strftime("%Y-%m-%d"),
            "equity": round(equity, 2),
            "trade_pnl": round(dollar_pnl, 2),
            "contracts": round(contracts, 4),
            "exit_reason": trade.exit_reason,
        })

    return {
        "initial_capital": initial_capital,
        "final_equity": round(equity, 2),
        "net_profit": round(equity - initial_capital, 2),
        "net_profit_pct": round((equity / initial_capital - 1) * 100, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct * 100, 2),
        "peak_equity": round(peak_equity, 2),
        "equity_curve": equity_curve,
    }
