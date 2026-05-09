import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

from src.data.loaders.fomc_loader import load_fomc_dates
from src.data.loaders.fred_loader import load_fred_release_dates
from fredapi import Fred

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FOMCLoaderWrapper:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        # Load all dates upfront
        self.df = load_fomc_dates(self.csv_path, start=pd.Timestamp("2007-01-01"), end=pd.Timestamp("2026-12-31"))
        
    def get_fomc_in_window(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
        mask = (self.df['decision_date'] >= start_date) & (self.df['decision_date'] <= end_date)
        recent = self.df[mask]
        
        if recent.empty:
            return "No FOMC meetings in the last 72 hours."
            
        fomc_info = []
        for _, row in recent.iterrows():
            fomc_info.append(f"Meeting on {row['decision_date'].strftime('%Y-%m-%d')}. Notes: {row['notes']}")
            
        return " | ".join(fomc_info)

class FredLoaderWrapper:
    def __init__(self, api_key: str):
        self.fred = Fred(api_key=api_key)
        self.cache_dir = Path("data/raw/fred")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load release dates cache
        try:
            self.cpi_dates = load_fred_release_dates("CPIAUCSL", pd.Timestamp("2007-01-01"), pd.Timestamp("2025-12-31"), self.cache_dir, api_key)
            self.nfp_dates = load_fred_release_dates("PAYEMS", pd.Timestamp("2007-01-01"), pd.Timestamp("2025-12-31"), self.cache_dir, api_key)
            self.rate_dates = load_fred_release_dates("FEDFUNDS", pd.Timestamp("2007-01-01"), pd.Timestamp("2025-12-31"), self.cache_dir, api_key)
        except Exception as e:
            logger.warning(f"Could not load FRED release dates properly: {e}")
            self.cpi_dates = pd.DataFrame(columns=['reference_date', 'release_date', 'available_from'])
            self.nfp_dates = pd.DataFrame(columns=['reference_date', 'release_date', 'available_from'])
            self.rate_dates = pd.DataFrame(columns=['reference_date', 'release_date', 'available_from'])

        # Download full series data for values (caching locally to avoid massive API calls per date)
        self.series_data = {}
        for series_id in ["CPIAUCSL", "PAYEMS", "FEDFUNDS"]:
            cache_file = self.cache_dir / f"{series_id}_values.parquet"
            if cache_file.exists():
                self.series_data[series_id] = pd.read_parquet(cache_file)
            else:
                try:
                    s = self.fred.get_series(series_id)
                    df = s.to_frame(name="value")
                    df.index.name = "reference_date"
                    df.to_parquet(cache_file)
                    self.series_data[series_id] = df
                except Exception as e:
                    logger.warning(f"Failed to download FRED values for {series_id}: {e}")
                    self.series_data[series_id] = pd.DataFrame(columns=["value"])

    def get_as_of_data(self, series_id: str, trade_date: pd.Timestamp):
        if series_id == "CPIAUCSL":
            dates_df = self.cpi_dates
        elif series_id == "PAYEMS":
            dates_df = self.nfp_dates
        else:
            dates_df = self.rate_dates
            
        valid_releases = dates_df[dates_df['available_from'] <= trade_date]
        if valid_releases.empty:
            return "N/A", "N/A"
            
        latest_release = valid_releases.sort_values('available_from').iloc[-1]
        ref_date = latest_release['reference_date']
        avail_date = latest_release['available_from'].strftime('%Y-%m-%d')
        
        val_df = self.series_data.get(series_id)
        if val_df is not None and not val_df.empty and ref_date in val_df.index:
            val = val_df.loc[ref_date, "value"]
            # Formatting
            if series_id == "CPIAUCSL":
                val_str = f"{val:.3f} (Index)"
            elif series_id == "PAYEMS":
                val_str = f"{val:,.0f}K"
            else:
                val_str = f"{val:.2f}"
            return val_str, avail_date
            
        return "Unknown", avail_date


def main():
    dates_file = "daty_transakcji_272.txt"
    fomc_file = "data/calendars_source/fomc_dates_source.csv"
    output_file = "data/reports/llm_real_payloads.json"
    
    system_prompt = (
        "You are a Macro Risk Manager. Approve trades by default. "
        "Veto ONLY if the provided FRED/FOMC data indicates a severe hawkish shock (unexpected rate hike) "
        "or an inflation (CPI) spike that threatens the tech-heavy NASDAQ (US100). "
        "Respond in JSON: {'veto': boolean, 'reason': string}."
    )
    
    with open(dates_file, 'r') as f:
        dates = [line.strip() for line in f if line.strip()]
        
    api_key = os.environ.get('FRED_API_KEY')
    fred_loader = FredLoaderWrapper(api_key=api_key)
    fomc_loader = FOMCLoaderWrapper(csv_path=fomc_file)
    
    payloads = []
    
    for date_str in dates:
        trade_date = pd.to_datetime(date_str)
        start_date = trade_date - timedelta(days=3)
        
        # 1. FOMC Activity (Last 72h)
        fomc_info = fomc_loader.get_fomc_in_window(start_date, trade_date)
        
        # 2. FRED Data (1-day lag applied via available_from)
        cpi_val, cpi_date = fred_loader.get_as_of_data("CPIAUCSL", trade_date)
        nfp_val, nfp_date = fred_loader.get_as_of_data("PAYEMS", trade_date)
        rate_val, rate_date = fred_loader.get_as_of_data("FEDFUNDS", trade_date)
        
        user_prompt = (
            f"The ML algorithm has generated a LONG signal for the US100 for {date_str}.\n"
            f"Macroeconomic Context (FRED & FOMC Data):\n"
            f"- Latest CPI Value: {cpi_val} (Available since: {cpi_date})\n"
            f"- Latest NFP Change: {nfp_val} (Available since: {nfp_date})\n"
            f"- Current Fed Funds Rate: {rate_val}%\n"
            f"- FOMC Activity (Last 72h): {fomc_info}\n\n"
            f"Based on these specific macro indicators, should we veto this trade? Look for inflation shocks or hawkish surprises."
        )
        
        payloads.append({
            "trade_date": date_str,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        })
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(payloads, f, indent=4)
        
    print(f"Generated {len(payloads)} payloads to {output_file}")

if __name__ == "__main__":
    main()
