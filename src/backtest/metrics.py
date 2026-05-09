"""Professional trading metrics for the backtest report."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.backtest.simulator import TradeResult


def compute_trading_metrics(
    trades: list[TradeResult],
    portfolio: dict[str, Any],
    trading_days_per_year: int = 252,
) -> dict[str, Any]:
    """Compute the full set of professional trading metrics.

    Returns a flat dictionary suitable for JSON serialisation.
    """
    if not trades:
        return _empty_metrics(portfolio)

    n_trades = len(trades)
    wins = [t for t in trades if t.pnl_points > 0]
    losses = [t for t in trades if t.pnl_points <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0

    # Exit reason breakdown
    tp_exits = sum(1 for t in trades if t.exit_reason == "tp")
    sl_exits = sum(1 for t in trades if t.exit_reason == "sl")
    time_exits = sum(1 for t in trades if t.exit_reason == "time")

    # Profit Factor
    gross_profit = portfolio["gross_profit"]
    gross_loss = portfolio["gross_loss"]
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average trade stats
    all_pnl = [t.pnl_points for t in trades]
    avg_win = float(np.mean([t.pnl_points for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl_points for t in losses])) if losses else 0.0
    avg_holding = float(np.mean([t.holding_days for t in trades]))

    # Total costs
    total_spread = sum(t.spread_cost for t in trades)
    total_swap = sum(t.swap_cost for t in trades)

    # Sharpe Ratio (annualised from per-trade returns)
    equity_changes = [e["trade_pnl"] for e in portfolio["equity_curve"]]
    if len(equity_changes) > 1 and np.std(equity_changes) > 0:
        # Calculate per-trade return as pct of equity at that point
        # Use trade P&L directly for Sharpe
        mean_pnl = float(np.mean(equity_changes))
        std_pnl = float(np.std(equity_changes, ddof=1))

        # Approximate trades per year
        if len(trades) >= 2:
            first_date = trades[0].entry_date
            last_date = trades[-1].entry_date
            total_days = (last_date - first_date).days
            if total_days > 0:
                trades_per_year = (n_trades / total_days) * 365.25
            else:
                trades_per_year = n_trades
        else:
            trades_per_year = n_trades

        sharpe = (mean_pnl / std_pnl) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    return {
        "total_trades": n_trades,
        "winning_trades": n_wins,
        "losing_trades": n_losses,
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 4),
        "net_profit_pct": portfolio["net_profit_pct"],
        "max_drawdown_pct": portfolio["max_drawdown_pct"],
        "max_drawdown_usd": portfolio["max_drawdown"],
        "annualized_sharpe": round(float(sharpe), 4),
        "avg_win_points": round(avg_win, 2),
        "avg_loss_points": round(avg_loss, 2),
        "avg_holding_days": round(avg_holding, 1),
        "exit_reasons": {
            "tp": tp_exits,
            "sl": sl_exits,
            "time": time_exits,
        },
        "total_spread_cost_points": round(total_spread, 2),
        "total_swap_cost_points": round(total_swap, 2),
        "starting_capital": portfolio["initial_capital"],
        "final_equity": portfolio["final_equity"],
        "gross_profit": portfolio["gross_profit"],
        "gross_loss": portfolio["gross_loss"],
    }


def _empty_metrics(portfolio: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "net_profit_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "max_drawdown_usd": 0.0,
        "annualized_sharpe": 0.0,
        "avg_win_points": 0.0,
        "avg_loss_points": 0.0,
        "avg_holding_days": 0.0,
        "exit_reasons": {"tp": 0, "sl": 0, "time": 0},
        "total_spread_cost_points": 0.0,
        "total_swap_cost_points": 0.0,
        "starting_capital": portfolio.get("initial_capital", 10000),
        "final_equity": portfolio.get("final_equity", 10000),
        "gross_profit": 0.0,
        "gross_loss": 0.0,
    }
