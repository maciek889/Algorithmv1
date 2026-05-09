import json
from datetime import datetime, timedelta
# Zakładam, że masz już dostęp do swoich loaderów w projekcie
# from src.data.loaders.fomc_loader import FOMCLoader
# from src.data.loaders.fred_loader import FredLoader

def load_dates(filepath="daty_transakcji_272.txt"):
    """Pobiera 272 daty wygenerowane przez model ML."""
    with open(filepath, 'r') as f:
        dates = [line.strip() for line in f if line.strip()]
    return dates

def get_macro_context_for_date(trade_date_str):
    """
    Symulacja pobierania danych Makro/News z przedziału (T-3 do T-1).
    W pełnej wersji użyjemy tu Twojego fomc_loader i fred_loader.
    """
    trade_date = datetime.strptime(trade_date_str, "%Y-%m-%d")
    
    # Symulacja: sprawdzamy co działo się w dniach poprzedzających sygnał
    # Docelowo: 
    # 1. Sprawdzamy czy w ciągu ostatnich 3 dni było posiedzenie FOMC.
    # 2. Sprawdzamy czy wczoraj opublikowano CPI/NFP i czy znacznie odbiegały od normy.
    # 3. Pobieramy nagłówki (lub mockujemy je dla historycznego backtestu).
    
    context_text = f"As of {trade_date_str}, here is the macroeconomic context from the last 72 hours:\n"
    
    # TUTAJ podepniemy prawdziwą logikę. Na razie budujemy szkielet:
    context_text += "- FOMC Data: No immediate unexpected rate hikes detected in the last 3 days.\n"
    context_text += "- FRED Data: Inflation (CPI) and Unemployment (NFP) remain within expected market consensus.\n"
    context_text += "- General Market News: Typical market volatility. No black swan events (e.g., pandemic crash, major war escalation) explicitly dominating the news cycle.\n"

    # Przykład ręcznego "wstrzyknięcia" szoku, żeby zobaczyć czy LLM zadziała w lutym/marcu 2020:
    if "2020-02" in trade_date_str or "2020-03" in trade_date_str:
         context_text = f"As of {trade_date_str}, WARNING: Extreme market panic. Global pandemic (COVID-19) fears are crashing the markets. Circuit breakers hit. High systemic risk."

    return context_text

def build_llm_payloads():
    dates = load_dates()
    payloads = []
    
    for date_str in dates:
        macro_context = get_macro_context_for_date(date_str)
        
        # Budujemy obiekt dla pojedynczej transakcji
        trade_payload = {
            "trade_date": date_str,
            "system_prompt": "You are a Senior Quantitative Risk Manager for a US100 Swing Trading algorithm. Your default action is to APPROVE trades. You only VETO a trade if the provided news explicitly indicates an extreme macroeconomic shock, panic, or systemic black-swan event.",
            "user_prompt": f"The ML algorithm has generated a LONG signal for the US100 for {date_str}. Based on the following context, should we veto this trade? Reply strictly in JSON format: {{\"veto\": true/false, \"reason\": \"short explanation\"}}.\n\nContext:\n{macro_context}"
        }
        payloads.append(trade_payload)
        
    # Zapis do pliku JSON
    output_file = "data/reports/llm_sandbox_payloads.json"
    with open(output_file, 'w') as f:
        json.dump(payloads, f, indent=4)
        
    print(f"Sukces! Wygenerowano {len(payloads)} paczek dla LLM-a. Zapisano w {output_file}")

if __name__ == "__main__":
    build_llm_payloads()