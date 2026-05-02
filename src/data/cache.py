"""Parquet-based cache with a small JSON metadata sidecar."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_META_FILENAME = "_cache_meta.json"


def is_cached(path: Path) -> bool:
    """Return True if a parquet cache file exists at the given path."""
    return path.exists() and path.is_file()


def read_cache(path: Path) -> pd.DataFrame | None:
    """Read a cached parquet, or return None if missing.

    Args:
        path: Full path to the .parquet file.

    Returns:
        DataFrame if cache exists, otherwise None.
    """
    if not is_cached(path):
        return None
    logger.debug("Cache hit: %s", path)
    return pd.read_parquet(path)


def write_cache(df: pd.DataFrame, path: Path, source: str, key: str) -> None:
    """Write DataFrame to parquet and update the per-source metadata sidecar.

    Args:
        df: DataFrame to persist.
        path: Destination .parquet path.
        source: Logical source name (e.g. "yfinance", "fred", "cboe").
        key: Sub-identifier within the source (ticker, series_id, etc).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.debug("Cache write: %s", path)

    meta_path = path.parent.parent / _META_FILENAME
    _update_meta(meta_path, source, key)


def _update_meta(meta_path: Path, source: str, key: str) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Corrupt cache meta at %s; rewriting fresh", meta_path)
            meta = {}
    meta.setdefault(source, {})[key] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
