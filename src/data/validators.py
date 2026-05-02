"""Generate the human-readable data quality report (Markdown)."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OK = "✅"
WARN = "⚠️"
FAIL = "❌"

_EXPECTED_PER_YEAR = {"fomc": 8, "cpi": 12, "nfp": 12}
_TOLERANCE = 2  # ± events/year before flagging


def generate_qa_report(
    master_df: pd.DataFrame,
    calendars: dict[str, pd.DataFrame],
    output_path: Path,
    training_start: date,
    putcall_source: str | None = None,
) -> None:
    """Write data_quality_report.md based on the assembled master + calendars.

    Args:
        master_df: Output of build_master_dataset.
        calendars: Mapping kind -> events DataFrame. Keys: 'fomc', 'cpi', 'nfp'.
            FOMC frame: columns include 'meeting_date'.
            CPI/NFP frames: columns include 'release_date'.
        output_path: Destination .md path; parent dirs created if missing.
        training_start: Date below which NaN in core columns is acceptable.
        putcall_source: Logical source string for §Putcall (e.g. 'cboe_direct'
            or 'yfinance_cpc').
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        _header(),
        _master_overview(master_df),
        _nan_section(master_df, training_start),
        _ohlc_sanity(master_df),
        _volume_section(master_df),
        _putcall_section(master_df, putcall_source),
        _calendar_section(calendars),
        _distribution_section(master_df),
    ]

    output_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    logger.info("Wrote QA report to %s", output_path)


def _header() -> str:
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return f"# Data Quality Report\n\n*Generated {now}*"


def _master_overview(df: pd.DataFrame) -> str:
    if df.empty:
        return f"## 1. Master overview\n\n{FAIL} master dataset is empty"

    by_year = df.groupby(df.index.year).size()
    rows = "\n".join(f"| {y} | {n} |" for y, n in by_year.items())
    return (
        "## 1. Master overview\n\n"
        f"- First date: `{df.index.min().date()}`\n"
        f"- Last date: `{df.index.max().date()}`\n"
        f"- Trading days: **{len(df)}**\n\n"
        "| Year | Trading days |\n|---|---|\n"
        f"{rows}"
    )


def _nan_section(df: pd.DataFrame, training_start: date) -> str:
    cutoff = pd.Timestamp(training_start)
    post = df.loc[df.index >= cutoff]
    core = ["us100_open", "us100_high", "us100_low", "us100_close", "us100_volume",
            "vix_close", "spx_close", "dxy_close"]
    nan = post[core].isna().sum()
    rows = []
    overall = OK
    for col in core:
        n = int(nan.get(col, 0))
        status = OK if n == 0 else FAIL
        if n > 0:
            overall = FAIL
        rows.append(f"| `{col}` | {n} | {status} |")
    body = "\n".join(rows)
    return (
        f"## 2. NaN counts post training_start ({training_start})  {overall}\n\n"
        "| Column | NaN count | Status |\n|---|---|---|\n"
        f"{body}"
    )


def _ohlc_sanity(df: pd.DataFrame) -> str:
    needed = {"us100_open", "us100_high", "us100_low", "us100_close"}
    if not needed.issubset(df.columns):
        return f"## 3. OHLC sanity\n\n{WARN} missing one of {needed}"

    bad_hl = (df["us100_high"] < df["us100_low"]).sum()
    bad_zero = ((df["us100_close"] == 0) | (df["us100_high"] == 0)).sum()

    rets = df["us100_close"].pct_change(fill_method=None)
    sigma = rets.std()
    outliers = int((rets.abs() > 5 * sigma).sum()) if sigma > 0 else 0

    status_hl = OK if bad_hl == 0 else FAIL
    status_zero = OK if bad_zero == 0 else FAIL
    status_outliers = OK if outliers <= 50 else WARN

    return (
        "## 3. OHLC sanity (US100)\n\n"
        f"- Days with high < low: **{bad_hl}** {status_hl}\n"
        f"- Days with close==0 or high==0: **{bad_zero}** {status_zero}\n"
        f"- Days with |return| > 5σ: **{outliers}** {status_outliers}"
    )


def _volume_section(df: pd.DataFrame) -> str:
    if "us100_volume" not in df.columns:
        return f"## 4. Volume\n\n{WARN} us100_volume column missing"

    vol = df["us100_volume"]
    zeros = int((vol == 0).sum())
    status = OK if zeros == 0 else FAIL
    p10, p50, p90 = (int(np.percentile(vol[vol > 0], q)) for q in (10, 50, 90))
    return (
        "## 4. Volume (US100)\n\n"
        f"- Days with zero volume: **{zeros}** {status}\n"
        f"- Volume percentiles (non-zero): p10={p10:,}, p50={p50:,}, p90={p90:,}"
    )


def _putcall_section(df: pd.DataFrame, source: str | None) -> str:
    if "putcall_ratio" not in df.columns:
        return (
            f"## 5. Put/Call coverage  {WARN}\n\n"
            f"- Source used: `{source or 'unavailable'}`\n"
            "- **Deferred**: CBOE direct CSV is bot-blocked (403); the documented "
            "yfinance fallback `^CPCE` is delisted. The `putcall_ratio` column is "
            "omitted from the master dataset until a working source is identified "
            "(planned for a later stage)."
        )

    pc = df["putcall_ratio"]
    coverage = pc.notna().mean() * 100
    is_na = pc.isna().astype(int)
    if is_na.sum() == 0:
        max_gap = 0
    else:
        groups = (is_na != is_na.shift()).cumsum()
        gap_lengths = is_na.groupby(groups).sum()
        max_gap = int(gap_lengths.max()) if not gap_lengths.empty else 0

    status = OK if coverage >= 95 else (WARN if coverage >= 80 else FAIL)
    return (
        "## 5. Put/Call coverage\n\n"
        f"- Source used: `{source or 'unknown'}`\n"
        f"- Coverage: **{coverage:.1f}%** {status}\n"
        f"- Longest consecutive-day gap: **{max_gap}**"
    )


def _calendar_section(calendars: dict[str, pd.DataFrame]) -> str:
    lines = ["## 6. Calendars\n", "| Kind | Year | Events | Expected | Status |", "|---|---|---|---|---|"]
    overall = OK
    for kind in ("fomc", "cpi", "nfp"):
        df = calendars.get(kind)
        if df is None or df.empty:
            lines.append(f"| {kind} | — | 0 | {_EXPECTED_PER_YEAR[kind]}/yr | {FAIL} |")
            overall = FAIL
            continue
        date_col = "meeting_date" if kind == "fomc" else "release_date"
        if date_col not in df.columns:
            lines.append(f"| {kind} | — | ? | {_EXPECTED_PER_YEAR[kind]}/yr | {WARN} (missing `{date_col}`) |")
            continue
        years = pd.to_datetime(df[date_col]).dt.year
        per_year = years.value_counts().sort_index()
        expected = _EXPECTED_PER_YEAR[kind]
        for y, n in per_year.items():
            deviation = abs(n - expected)
            status = OK if deviation <= _TOLERANCE else WARN
            if status == WARN:
                overall = WARN if overall == OK else overall
            lines.append(f"| {kind} | {y} | {n} | {expected} | {status} |")
    lines.insert(1, f"\nOverall: {overall}\n")
    return "\n".join(lines)


def _distribution_section(df: pd.DataFrame) -> str:
    if "us100_close" not in df.columns or "vix_close" not in df.columns:
        return f"## 7. Distribution sanity\n\n{WARN} missing core columns"

    rows = ["## 7. Distribution sanity (rolling 3-year windows)\n",
            "| Window | us100_close mean | us100_close std | vix_close mean | vix_close std |",
            "|---|---|---|---|---|"]
    start_year = df.index.min().year
    end_year = df.index.max().year
    for y0 in range(start_year, end_year, 3):
        y1 = y0 + 2
        sub = df.loc[(df.index.year >= y0) & (df.index.year <= y1)]
        if sub.empty:
            continue
        rows.append(
            f"| {y0}-{y1} | "
            f"{sub['us100_close'].mean():.1f} | {sub['us100_close'].std():.1f} | "
            f"{sub['vix_close'].mean():.2f} | {sub['vix_close'].std():.2f} |"
        )
    return "\n".join(rows)
