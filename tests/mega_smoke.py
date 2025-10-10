# mega_smoke.py
import logging
from pybit.unified_trading import HTTP
from config import BYBIT_API_KEY, BYBIT_API_SECRET, INTERVAL

from data_loader import set_client as set_dl_client, fetch_ohlcv
from pair_finder import set_client as set_pf_client, get_candidate_pairs
from pair_selector import evaluate_pairs
import pipeline as pl

logging.basicConfig(level=logging.INFO)

def main():
    print("\n=== MEGA SMOKE START ===")
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_dl_client(client)
    set_pf_client(client)

    # 1) Loader
    df = fetch_ohlcv("BTCUSDT", INTERVAL, limit=200)
    print(f"[Loader] BTCUSDT rows: {len(df)}   last: {df.index[-1]}")

    # 2) Finder
    pairs = get_candidate_pairs(force_refresh=True)[:15]
    print(f"[Finder] Got {len(pairs)} pairs: {pairs[:10]}")

    # 3) Selector
    top = evaluate_pairs(
        pairs, INTERVAL, top_n=5, return_scores=True, debug=True, window=200, save_results=True
    )
    print(f"[Selector] Top: {top}")

    # 4) Training (быстрое одиночное обучение; можно закомментить, если не нужно)
    print("[Training] Running pipeline for BTCUSDT (use_rolling_cv=False)...")
    pl.train_on_symbol("BTCUSDT", interval=INTERVAL, threshold=0.0015, use_rolling_cv=False)

    print("=== MEGA SMOKE DONE ===\n")

if __name__ == "__main__":
    main()
