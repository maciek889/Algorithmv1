"""Configuration loading: YAML + .env merged into a frozen dataclass."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.data.exceptions import ConfigError


@dataclass(frozen=True)
class TickerSpec:
    name: str
    ticker: str
    role: str


@dataclass(frozen=True)
class FredSeriesSpec:
    name: str
    series_id: str
    purpose: str


@dataclass(frozen=True)
class CboeSpec:
    source: str
    fallback: str


@dataclass(frozen=True)
class DataConfig:
    start: date
    end: date
    training_start: date
    yfinance_tickers: tuple[TickerSpec, ...]
    fred_series: tuple[FredSeriesSpec, ...]
    cboe: CboeSpec
    fred_api_key: str = field(repr=False)


def load_config(yaml_path: Path, env_path: Path | None = None) -> DataConfig:
    """Load YAML config and overlay environment variables.

    Args:
        yaml_path: Path to data_config.yaml.
        env_path: Optional path to .env. If omitted, dotenv searches default locations.

    Returns:
        Frozen DataConfig instance.

    Raises:
        ConfigError: file missing, YAML malformed, or required env var absent.
    """
    if not yaml_path.exists():
        raise ConfigError(f"Config file not found: {yaml_path}")

    if env_path is not None:
        load_dotenv(env_path)
    else:
        load_dotenv()

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Malformed YAML at {yaml_path}: {e}") from e

    try:
        date_range = raw["date_range"]
        tickers = tuple(TickerSpec(**t) for t in raw["yfinance_tickers"])
        fred = tuple(FredSeriesSpec(**s) for s in raw["fred_series"])
        cboe = CboeSpec(**raw["cboe"])
    except (KeyError, TypeError) as e:
        raise ConfigError(f"Config schema mismatch: {e}") from e

    fred_api_key = os.getenv("FRED_API_KEY", "").strip()
    if not fred_api_key:
        raise ConfigError(
            "FRED_API_KEY not found in environment. "
            "Copy .env.example to .env and set it. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    return DataConfig(
        start=_parse_date(date_range["start"]),
        end=_parse_date(date_range["end"]),
        training_start=_parse_date(date_range["training_start"]),
        yfinance_tickers=tickers,
        fred_series=fred,
        cboe=cboe,
        fred_api_key=fred_api_key,
    )


def _parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)
