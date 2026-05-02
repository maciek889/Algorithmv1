"""Tests for src.data.config."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.config import load_config
from src.data.exceptions import ConfigError


VALID_YAML = """
date_range:
  start: "2007-01-01"
  end: "2025-12-31"
  training_start: "2010-01-01"

yfinance_tickers:
  - {name: "us100", ticker: "^NDX", role: "primary"}
  - {name: "vix",   ticker: "^VIX", role: "context"}

fred_series:
  - {name: "cpi",  series_id: "CPIAUCSL", purpose: "release_dates_only"}

cboe:
  source: "cboe_direct"
  fallback: "yfinance_cpc"
"""


def _write_yaml(tmp_path: Path, content: str = VALID_YAML) -> Path:
    p = tmp_path / "data_config.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _write_env(tmp_path: Path, key: str = "fake_key_for_tests") -> Path:
    p = tmp_path / ".env"
    p.write_text(f"FRED_API_KEY={key}\n", encoding="utf-8")
    return p


def test_load_config_happy_path(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path)
    env_path = _write_env(tmp_path)

    cfg = load_config(yaml_path, env_path)

    assert cfg.start == date(2007, 1, 1)
    assert cfg.training_start == date(2010, 1, 1)
    assert len(cfg.yfinance_tickers) == 2
    assert cfg.yfinance_tickers[0].name == "us100"
    assert cfg.fred_api_key == "fake_key_for_tests"
    assert cfg.cboe.source == "cboe_direct"


def test_load_config_missing_yaml_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml", tmp_path / ".env")


def test_load_config_missing_fred_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    yaml_path = _write_yaml(tmp_path)
    empty_env = tmp_path / ".env"
    empty_env.write_text("FRED_API_KEY=\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="FRED_API_KEY"):
        load_config(yaml_path, empty_env)


def test_load_config_malformed_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("date_range: [unclosed", encoding="utf-8")
    env_path = _write_env(tmp_path)

    with pytest.raises(ConfigError, match="Malformed YAML"):
        load_config(bad, env_path)


def test_load_config_schema_mismatch_raises(tmp_path: Path) -> None:
    yaml_path = tmp_path / "incomplete.yaml"
    yaml_path.write_text("date_range:\n  start: '2007-01-01'\n", encoding="utf-8")
    env_path = _write_env(tmp_path)

    with pytest.raises(ConfigError, match="schema mismatch"):
        load_config(yaml_path, env_path)
