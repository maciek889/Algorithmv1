"""Tests for src.data.cache."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.cache import is_cached, read_cache, write_cache


def _sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)


def test_read_cache_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_cache(tmp_path / "nope.parquet") is None
    assert is_cached(tmp_path / "nope.parquet") is False


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "yfinance"
    path = cache_dir / "us100_2007_2025.parquet"
    df = _sample_df()

    write_cache(df, path, source="yfinance", key="us100")

    assert is_cached(path) is True
    loaded = read_cache(path)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)


def test_write_cache_updates_meta_sidecar(tmp_path: Path) -> None:
    cache_dir = tmp_path / "fred"
    path = cache_dir / "cpi.parquet"
    write_cache(_sample_df(), path, source="fred", key="cpi")
    write_cache(_sample_df(), path, source="fred", key="nfp")

    meta_path = tmp_path / "_cache_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "fred" in meta
    assert "cpi" in meta["fred"] and "nfp" in meta["fred"]


def test_write_cache_recovers_from_corrupt_meta(tmp_path: Path) -> None:
    meta_path = tmp_path / "_cache_meta.json"
    meta_path.write_text("{ not valid json", encoding="utf-8")

    cache_dir = tmp_path / "yfinance"
    write_cache(_sample_df(), cache_dir / "x.parquet", source="yfinance", key="x")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta == {"yfinance": {"x": meta["yfinance"]["x"]}}
