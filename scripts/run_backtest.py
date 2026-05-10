"""Stage 1.7 – Financial backtesting with transaction costs.

Re-runs the 56-fold walk-forward XGBoost loop to generate OOS predictions,
then resolves each predicted entry through a trailing-stop exit with
realistic CFD costs (spread + overnight swaps).

Usage:
    python scripts/run_backtest.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ── project imports ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.metrics import compute_trading_metrics          # noqa: E402
from src.backtest.simulator import (                              # noqa: E402
    TSL_ACTIVATION_MULTIPLIER,
    TradeResult,
    resolve_trade,
    run_portfolio_simulation,
    SPREAD_POINTS,
    SWAP_RATE_ANNUAL,
)
from src.features.engineering import compute_wilder_atr           # noqa: E402
from src.features.labeling import load_horizon_config             # noqa: E402
from src.model.walk_forward import WalkForwardSplitter            # noqa: E402

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_FEATURES_PATH = REPO_ROOT / "data" / "processed" / "features_and_labels.parquet"
DEFAULT_MASTER_PATH = REPO_ROOT / "data" / "processed" / "master_dataset.parquet"
DEFAULT_HORIZON_PATH = REPO_ROOT / "config" / "horizon.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "reports" / "backtest_results.json"
TARGET_COLUMN = "target"
RANDOM_SEED = 42
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01  # 1% of equity per trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── XGBoost static params (identical to Stage 1.6) ──────────────────────────
XGBOOST_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "learning_rate": 0.05,
    "n_estimators": 800,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 0.5,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "max_delta_step": 1,
    "early_stopping_rounds": 50,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Generate OOS predictions
# ═════════════════════════════════════════════════════════════════════════════

def generate_oos_predictions(
    features_path: Path,
    horizon_path: Path,
) -> pd.DataFrame:
    """Run the 56-fold walk-forward loop and collect OOS predictions.

    Returns a DataFrame indexed by date with columns:
        prediction  (0 or 1)
        actual      (0 or 1)
    """
    log.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN].astype(int)

    splitter = WalkForwardSplitter.from_horizon_config(horizon_path)
    folds = list(splitter.iter_folds(df))
    log.info("Walk-forward folds: %d", len(folds))

    predictions: list[pd.DataFrame] = []

    for fold in folds:
        X_train = X.iloc[fold.train_indices]
        y_train = y.iloc[fold.train_indices].to_numpy()
        X_test = X.iloc[fold.test_indices]
        y_test = y.iloc[fold.test_indices]

        # Dynamic class imbalance handling
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}
        model = XGBClassifier(**params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train, y_train,
                sample_weight=fold.train_weights,
                eval_set=[(X_test, y_test.to_numpy())],
                verbose=False,
            )

        y_pred = model.predict(X_test)

        fold_preds = pd.DataFrame({
            "prediction": y_pred.astype(int),
            "actual": y_test.values.astype(int),
        }, index=X_test.index)
        predictions.append(fold_preds)

        n_signals = int(y_pred.sum())
        log.info(
            "Fold %2d | test %s to %s | %d signals / %d obs",
            fold.fold_id,
            fold.test_start.strftime("%Y-%m-%d"),
            fold.test_end.strftime("%Y-%m-%d"),
            n_signals,
            len(X_test),
        )

    all_preds = pd.concat(predictions).sort_index()

    # Handle any duplicate dates from overlapping fold boundaries
    if all_preds.index.has_duplicates:
        log.warning("Removing %d duplicate OOS dates", all_preds.index.duplicated().sum())
        all_preds = all_preds[~all_preds.index.duplicated(keep="first")]

    log.info(
        "Total OOS predictions: %d | Signals (pred=1): %d",
        len(all_preds), int(all_preds["prediction"].sum()),
    )
    return all_preds


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Resolve trades
# ═════════════════════════════════════════════════════════════════════════════

def resolve_all_trades(
    oos_predictions: pd.DataFrame,
    master_df: pd.DataFrame,
    horizon_path: Path,
) -> list[TradeResult]:
    """Resolve each ML signal=1 into a trade using a trailing stop."""
    hcfg = load_horizon_config(horizon_path)

    # Compute ATR from master data
    high = master_df["us100_high"].astype(float)
    low = master_df["us100_low"].astype(float)
    close = master_df["us100_close"].astype(float)
    atr = compute_wilder_atr(high, low, close, period=hcfg.atr_period)

    signal_dates = oos_predictions[oos_predictions["prediction"] == 1].index
    log.info("Resolving %d trade signals...", len(signal_dates))

    trades: list[TradeResult] = []
    skipped = 0
    in_trade_until: pd.Timestamp | None = None

    for entry_date in signal_dates:
        # Skip if we're already in a position (no overlapping trades)
        if in_trade_until is not None and entry_date <= in_trade_until:
            skipped += 1
            continue

        # Get ATR on signal date
        if entry_date not in atr.index:
            skipped += 1
            continue
        atr_value = float(atr.loc[entry_date])
        if not np.isfinite(atr_value) or atr_value <= 0:
            skipped += 1
            continue

        trade = resolve_trade(
            master_df=master_df,
            entry_date=entry_date,
            atr_value=atr_value,
            horizon_days=hcfg.horizon_days,
            sl_multiplier=hcfg.sl_multiplier,
        )

        if trade is not None:
            trades.append(trade)
            in_trade_until = trade.exit_date

    log.info("Resolved: %d trades | Skipped: %d (overlapping/invalid)", len(trades), skipped)
    return trades


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Orchestrate
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(
    features_path: Path = DEFAULT_FEATURES_PATH,
    master_path: Path = DEFAULT_MASTER_PATH,
    horizon_path: Path = DEFAULT_HORIZON_PATH,
) -> dict[str, Any]:
    """Full backtest pipeline: predictions → trade resolution → metrics."""

    # 1) Generate OOS predictions
    oos = generate_oos_predictions(features_path, horizon_path)

    # 2) Load master dataset (for OHLC data)
    log.info("Loading master dataset from %s", master_path)
    master_df = pd.read_parquet(master_path)

    # 3) Resolve trades
    trades = resolve_all_trades(oos, master_df, horizon_path)

    if not trades:
        log.error("No trades could be resolved!")
        return {"error": "No trades resolved"}

    # 4) Portfolio simulation
    log.info("Running portfolio simulation (capital=$%.0f, risk=%.1f%%)",
             INITIAL_CAPITAL, RISK_PER_TRADE * 100)
    portfolio = run_portfolio_simulation(
        trades=trades,
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade_pct=RISK_PER_TRADE,
    )

    # 5) Metrics
    metrics = compute_trading_metrics(trades, portfolio)

    # Build full report
    hcfg = load_horizon_config(horizon_path)
    report = {
        "stage": "1.7_financial_backtest",
        "configuration": {
            "initial_capital": INITIAL_CAPITAL,
            "risk_per_trade_pct": RISK_PER_TRADE * 100,
            "position_sizing": "Fixed fractional: risk 1% of equity per trade based on SL distance",
            "exit_logic": "ATR trailing stop; no fixed take profit",
            "spread_points": SPREAD_POINTS,
            "swap_rate_annual": SWAP_RATE_ANNUAL,
            "horizon_days": hcfg.horizon_days,
            "initial_sl_atr_multiplier": hcfg.sl_multiplier,
            "tsl_activation_atr_multiplier": TSL_ACTIVATION_MULTIPLIER,
            "trailing_distance_atr_multiplier": hcfg.sl_multiplier,
            "atr_period": hcfg.atr_period,
        },
        "oos_summary": {
            "total_oos_observations": int(len(oos)),
            "total_signals": int(oos["prediction"].sum()),
            "signal_rate_pct": round(float(oos["prediction"].mean()) * 100, 2),
        },
        "metrics": metrics,
        "portfolio": {
            "initial_capital": portfolio["initial_capital"],
            "final_equity": portfolio["final_equity"],
            "net_profit": portfolio["net_profit"],
            "net_profit_pct": portfolio["net_profit_pct"],
            "peak_equity": portfolio["peak_equity"],
            "max_drawdown": portfolio["max_drawdown"],
            "max_drawdown_pct": portfolio["max_drawdown_pct"],
        },
        "equity_curve": portfolio["equity_curve"],
    }
    return report


def save_report(report: dict[str, Any], output_path: Path) -> None:
    """Persist the backtest report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    log.info("Report saved to %s", output_path)


def print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable backtest summary."""
    m = report["metrics"]
    p = report["portfolio"]
    cfg = report["configuration"]

    print()
    print("=" * 70)
    print("  STAGE 1.7 — Financial Backtest Results")
    print("=" * 70)
    print()
    print(f"  Starting Capital:    ${p['initial_capital']:>12,.2f}")
    print(f"  Final Equity:        ${p['final_equity']:>12,.2f}")
    print(f"  Net Profit:          ${p['net_profit']:>12,.2f}  ({p['net_profit_pct']:+.2f}%)")
    print(f"  Peak Equity:         ${p['peak_equity']:>12,.2f}")
    print(f"  Max Drawdown:        ${p['max_drawdown']:>12,.2f}  ({p['max_drawdown_pct']:.2f}%)")
    print()
    print(f"  Total Trades:        {m['total_trades']}")
    print(f"  Win Rate:            {m['win_rate_pct']:.2f}%")
    print(f"  Profit Factor:       {m['profit_factor']:.4f}")
    print(f"  Annualized Sharpe:   {m['annualized_sharpe']:.4f}")
    print()
    print(f"  Avg Win (pts):       {m['avg_win_points']:+.2f}")
    print(f"  Avg Loss (pts):      {m['avg_loss_points']:.2f}")
    print(f"  Avg Holding (days):  {m['avg_holding_days']:.1f}")
    print()
    print(f"  Exit Reasons:  SL={m['exit_reasons']['sl']}  "
          f"TSL={m['exit_reasons']['tsl']}  "
          f"Time={m['exit_reasons']['time']}")
    print()
    print(f"  Cost Model:  Spread={cfg['spread_points']} pts  "
          f"Swap={cfg['swap_rate_annual']*100:.1f}% annualized")
    print(f"  Total Costs:  Spread={m['total_spread_cost_points']:.1f} pts  "
          f"Swap={m['total_swap_cost_points']:.1f} pts")
    print()

    profitable = p["net_profit"] > 0
    print(f"  PROFITABLE AFTER COSTS? {'YES' if profitable else 'NO'}")
    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--horizon", type=Path, default=DEFAULT_HORIZON_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_backtest(
        features_path=_resolve(args.features),
        master_path=_resolve(args.master),
        horizon_path=_resolve(args.horizon),
    )
    save_report(report, _resolve(args.output))
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
