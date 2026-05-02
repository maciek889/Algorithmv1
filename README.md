# US100 ML Trading System — Data Pipeline (Stage 1.1)

Foundation data pipeline for an autonomous ML-based trading system on the
NASDAQ 100 (US100). This stage produces a single master parquet of OHLCV +
market context, plus release-date calendars for macro events (FOMC, CPI, NFP).

**Out of scope here**: features (MA50, RSI, ATR, …), labels, the model, and the
backtest. Those land in Stages 1.2 → 1.7.

---

## Setup

Requires Python 3.11. Other versions are not validated.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### `.env`

Create `.env` from the template and add a free FRED API key (instant signup at
<https://fred.stlouisfed.org/docs/api/api_key.html>):

```powershell
Copy-Item .env.example .env
# then edit .env and set FRED_API_KEY=...
```

### FOMC source CSV

The pipeline reads FOMC meeting dates from
`data/calendars_source/fomc_dates_source.csv` (committed to the repo).
Schema: `meeting_date,decision_date,is_unscheduled` (UTF-8, no BOM, ISO dates).
Source: <https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm>.
A `notes` column may be present and is ignored by the loader.

---

## Usage

```powershell
# Default: use cache where possible
python scripts/build_dataset.py

# Force-refresh all sources
python scripts/build_dataset.py --refresh-all

# Refresh a single source
python scripts/build_dataset.py --refresh-source yfinance
python scripts/build_dataset.py --refresh-source fred
python scripts/build_dataset.py --refresh-source cboe
```

Outputs:

| Path | Contents |
|---|---|
| `data/processed/master_dataset.parquet` | OHLCV + context, anchored on US100 trading days |
| `data/calendars/fomc_dates.parquet` | FOMC meeting dates |
| `data/calendars/cpi_release_dates.parquet` | CPI publication dates (as-of) |
| `data/calendars/nfp_release_dates.parquet` | NFP publication dates (as-of) |
| `data/reports/data_quality_report.md` | Diagnostic report — read after every build |

A clean run completes in well under five minutes; a warm-cache rerun in under
one minute.

---

## Tests

```powershell
# Fast unit tests (no external API calls)
pytest tests/ -m "not integration"

# Integration test (real FRED call: verifies CPI Jan-2024 release date)
pytest tests/ -m integration
```

Integration tests require a valid `FRED_API_KEY` in `.env`.

---

## Project layout

```
ML/
├── config/data_config.yaml       # tickers, date range, FRED series
├── data/                         # gitignored cache + outputs
│   ├── raw/{yfinance,fred,cboe}/ # parquet caches per source
│   ├── calendars/                # FOMC/CPI/NFP release dates (parquet)
│   ├── calendars_source/         # committed FOMC source CSV
│   ├── processed/                # master_dataset.parquet
│   └── reports/                  # data_quality_report.md
├── src/data/
│   ├── config.py                 # YAML + .env -> DataConfig
│   ├── exceptions.py             # DataLoadError / ValidationError / ConfigError
│   ├── cache.py                  # parquet cache helpers
│   ├── assembler.py              # build_master_dataset
│   ├── validators.py             # generate_qa_report
│   └── loaders/
│       ├── yfinance_loader.py    # OHLCV with retry + OHLC validation
│       ├── fred_loader.py        # release dates with as-of timing
│       ├── fomc_loader.py        # CSV reader for FOMC dates
│       └── cboe_loader.py        # put/call (CBOE direct + ^CPCE fallback)
├── scripts/build_dataset.py      # CLI entry point
└── tests/                        # pytest suite
```

---

## Reading the QA report

The report uses ✅ / ⚠️ / ❌ status markers per section. Expected statuses on
a clean build:

| Section | Expected | Watch for |
|---|---|---|
| 1. Master overview | 250 ± 5 trading days/yr | sudden gaps |
| 2. NaN counts post training_start | All ✅ | any ❌ → investigate before using the data |
| 3. OHLC sanity | 0 high<low, 0 zeros | ❌ on either is a hard fail |
| 4. Volume | 0 zero-volume days for US100 | ❌ → upstream issue |
| 5. Put/Call coverage | ⚠️ "deferred" (see Known Issues) | resolved when a working source ships |
| 6. Calendars | All years ≥ 2007 ✅; current year may show ⚠️ | mid-year warnings on the current year are expected |
| 7. Distribution sanity | Means/stds rise smoothly across windows | abrupt drops/spikes hint at upstream data issues |

---

## Known issues / deferred work

- **CBOE put/call ratio**: CBOE's CDN endpoints are bot-blocked (403); the
  spec's documented yfinance fallback `^CPCE` has been delisted. The
  `putcall_ratio` column is **omitted** from the master dataset. The loader
  remains in the codebase for when a working source is identified
  (likely Stage 1.3).
- **Pinned dependency drift from spec**:
  - `yfinance==1.3.0` (spec asked 0.2.40, which doesn't exist; the entire
    0.2.x line no longer authenticates against current Yahoo Finance).
  - `pandas-ta` removed (spec asked 0.3.14b0, which doesn't exist on PyPI;
    the package isn't used until Stage 1.3 anyway).

---

## Troubleshooting

- **`ConfigError: FRED_API_KEY not found`** — create `.env` (see Setup).
- **`DataLoadError: yfinance returned no data`** — Yahoo Finance can rate-limit
  or transiently fail. Re-run the build; retries with backoff are automatic.
  If persistent, try `--refresh-source yfinance` after a few minutes.
- **`DataLoadError: FOMC source CSV not found`** — see the FOMC source CSV
  setup section above.
- **`ValidationError: NaN found in core columns after training_start`** —
  upstream data has a gap larger than the 1-day forward-fill tolerance.
  Inspect the QA report's NaN section, then the raw cache for the offending
  source.
