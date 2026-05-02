"""Build the master dataset end-to-end.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --refresh-all
    python scripts/build_dataset.py --refresh-source yfinance
    python scripts/build_dataset.py --refresh-source fred
    python scripts/build_dataset.py --refresh-source cboe
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.assembler import build_master_dataset  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.data.loaders.cboe_loader import load_cboe_putcall  # noqa: E402
from src.data.loaders.fomc_loader import load_fomc_dates  # noqa: E402
from src.data.loaders.fred_loader import load_fred_release_dates  # noqa: E402
from src.data.loaders.yfinance_loader import load_yfinance_series  # noqa: E402
from src.data.validators import generate_qa_report  # noqa: E402

CONFIG_PATH = REPO_ROOT / "config" / "data_config.yaml"
ENV_PATH = REPO_ROOT / ".env"

DATA_DIR = REPO_ROOT / "data"
RAW_YFIN = DATA_DIR / "raw" / "yfinance"
RAW_FRED = DATA_DIR / "raw" / "fred"
RAW_CBOE = DATA_DIR / "raw" / "cboe"
CALENDARS_DIR = DATA_DIR / "calendars"
CALENDARS_SOURCE = DATA_DIR / "calendars_source" / "fomc_dates_source.csv"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    refresh_yf = args.refresh_all or args.refresh_source == "yfinance"
    refresh_fred = args.refresh_all or args.refresh_source == "fred"
    refresh_cboe = args.refresh_all or args.refresh_source == "cboe"

    cfg = load_config(CONFIG_PATH, ENV_PATH)

    yfinance_data: dict[str, pd.DataFrame] = {}
    for spec in cfg.yfinance_tickers:
        yfinance_data[spec.name] = load_yfinance_series(
            spec.name, spec.ticker, cfg.start, cfg.end, RAW_YFIN, refresh=refresh_yf,
        )

    calendars: dict[str, pd.DataFrame] = {}
    for spec in cfg.fred_series:
        cal = load_fred_release_dates(
            spec.series_id, cfg.start, cfg.end, RAW_FRED, cfg.fred_api_key, refresh=refresh_fred,
        )
        calendars[spec.name] = cal
        CALENDARS_DIR.mkdir(parents=True, exist_ok=True)
        cal.to_parquet(CALENDARS_DIR / f"{spec.name}_release_dates.parquet")

    fomc = load_fomc_dates(CALENDARS_SOURCE, cfg.start, cfg.end)
    calendars["fomc"] = fomc
    CALENDARS_DIR.mkdir(parents=True, exist_ok=True)
    fomc.to_parquet(CALENDARS_DIR / "fomc_dates.parquet")

    putcall = load_cboe_putcall(cfg.start, cfg.end, RAW_CBOE, source=cfg.cboe.source, refresh=refresh_cboe)
    putcall_source = _detect_putcall_source(RAW_CBOE)

    master = build_master_dataset(yfinance_data, putcall, cfg)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master_path = PROCESSED_DIR / "master_dataset.parquet"
    master.to_parquet(master_path)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "data_quality_report.md"
    generate_qa_report(master, calendars, report_path, training_start=cfg.training_start, putcall_source=putcall_source)

    print()
    print(f"Master dataset: {len(master)} trading days, {master.index.min().date()} to {master.index.max().date()}")
    print(f"  -> {master_path}")
    print(f"Quality report: {report_path}")
    print(f"Calendars:      {CALENDARS_DIR}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--refresh-all", action="store_true", help="Refresh all sources, ignoring cache")
    g.add_argument("--refresh-source", choices=("yfinance", "fred", "cboe"), help="Refresh a single source")
    return p.parse_args()


def _detect_putcall_source(cache_dir: Path) -> str | None:
    """Read _cache_meta.json to learn which source ultimately fed the put/call cache."""
    import json
    meta_path = cache_dir.parent / "_cache_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    cboe = meta.get("cboe", {})
    if not cboe:
        return None
    return next(iter(cboe.keys()))


if __name__ == "__main__":
    raise SystemExit(main())
