"""
Stage 2 - LLM Risk Manager: Evaluate 272 ML trade signals via GPT-4o-mini.

Architecture:
  - ThreadPoolExecutor with 3 workers for parallelism without overwhelming
    Tier-1 rate limits (~500 RPM for gpt-4o-mini).
  - Synchronous OpenAI SDK client (thread-safe) with exponential backoff.
  - response_format={"type": "json_object"} for strict JSON output.
  - Temperature 0.0 for deterministic, reproducible decisions.
  - tqdm progress bar for real-time monitoring.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from tqdm import tqdm

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
WORKERS = 3            # thread pool size - conservative for Tier-1
MAX_RETRIES = 8
BASE_BACKOFF_S = 2.0   # initial backoff for retries

ROOT = Path(__file__).resolve().parents[1]
PAYLOADS_PATH = ROOT / "data" / "reports" / "llm_sandbox_payloads.json"
OUTPUT_PATH = ROOT / "data" / "reports" / "llm_vetoes.json"


# ----------------------------------------------------------------
# Single payload evaluator
# ----------------------------------------------------------------
def evaluate_payload(client: OpenAI, payload: dict) -> tuple[str, dict]:
    """Send one payload to GPT-4o-mini and return (trade_date, decision)."""
    trade_date = payload["trade_date"]
    messages = [
        {"role": "system", "content": payload["system_prompt"]},
        {"role": "user", "content": payload["user_prompt"]},
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            decision = json.loads(raw)
            return trade_date, decision

        except RateLimitError as exc:
            if attempt == MAX_RETRIES:
                return trade_date, {
                    "veto": None,
                    "reason": f"FAILED after {MAX_RETRIES} retries: {exc}",
                }
            wait = BASE_BACKOFF_S * (2 ** (attempt - 1)) + random.uniform(0, 1)
            tqdm.write(
                f"  [WARN] {trade_date} attempt {attempt}/{MAX_RETRIES} "
                f"(RateLimitError), retrying in {wait:.1f}s ..."
            )
            time.sleep(wait)

        except (APIConnectionError, APIError) as exc:
            if attempt == MAX_RETRIES:
                return trade_date, {
                    "veto": None,
                    "reason": f"FAILED after {MAX_RETRIES} retries: {exc}",
                }
            wait = BASE_BACKOFF_S * (2 ** (attempt - 1)) + random.uniform(0, 1)
            tqdm.write(
                f"  [WARN] {trade_date} attempt {attempt}/{MAX_RETRIES} "
                f"({type(exc).__name__}), retrying in {wait:.1f}s ..."
            )
            time.sleep(wait)

        except json.JSONDecodeError:
            return trade_date, {
                "veto": None,
                "reason": f"JSON parse error: {raw[:200]}",
            }

    return trade_date, {"veto": None, "reason": "Unknown failure"}


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main() -> None:
    # 1) Load environment
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("[ERROR] OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)

    # 2) Load payloads
    with open(PAYLOADS_PATH, "r", encoding="utf-8") as f:
        payloads: list[dict] = json.load(f)
    print(f"[INFO] Loaded {len(payloads)} payloads from {PAYLOADS_PATH.name}")
    print(f"[INFO] Using {WORKERS} workers, model={MODEL}, temperature={TEMPERATURE}")

    # 3) Parallel evaluation with ThreadPoolExecutor
    results: dict[str, dict] = {}
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(evaluate_payload, client, p): p["trade_date"]
            for p in payloads
        }

        with tqdm(total=len(payloads), desc="LLM Risk Eval", unit="trade") as pbar:
            for future in as_completed(futures):
                trade_date, decision = future.result()
                results[trade_date] = decision
                pbar.update(1)

    elapsed = time.perf_counter() - t0

    # 4) Sort by date and save
    sorted_results = dict(sorted(results.items()))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2, ensure_ascii=False)

    # 5) Compute statistics
    n_vetoed = sum(1 for d in sorted_results.values() if d.get("veto") is True)
    n_approved = sum(1 for d in sorted_results.values() if d.get("veto") is False)
    n_errors = sum(1 for d in sorted_results.values() if d.get("veto") is None)
    total = len(payloads)

    # 6) Summary
    print(f"\n{'='*55}")
    print(f"  LLM Risk Manager - Stage 2 Evaluation Complete")
    print(f"{'='*55}")
    print(f"  Total trades evaluated : {total}")
    print(f"  [OK]  Approved         : {n_approved}")
    print(f"  [X]   Vetoed           : {n_vetoed}")
    if n_errors:
        print(f"  [!]   Errors           : {n_errors}")
    print(f"  Veto rate              : {n_vetoed / total * 100:.1f}%")
    print(f"  Wall-clock time        : {elapsed:.1f}s")
    print(f"  Output saved to        : {OUTPUT_PATH}")
    print(f"{'='*55}")

    # Print vetoed trades if any
    if n_vetoed > 0:
        print(f"\nVetoed trades detail:")
        for date, dec in sorted(sorted_results.items()):
            if dec.get("veto") is True:
                print(f"   {date}: {dec.get('reason', 'N/A')}")


if __name__ == "__main__":
    main()
