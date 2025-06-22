import os
import time
import logging
import pandas as pd
from typing import List
from data_loader import get_processed_ohlcv
from config import (
    MIN_HISTORY_BARS,
    MIN_AVG_VOLUME,
    QUOTE_ASSETS,
    INTERVAL,
    BLACKLIST_PATH,
)

from pybit.unified_trading import HTTP

logger = logging.getLogger(__name__)

# === Кеш и параметры ===
_cached_pairs = []
_last_updated = 0
CACHE_TTL = 60 * 15  # 15 минут

_client = None


def set_client(client: HTTP):
    global _client
    _client = client


def load_blacklist() -> List[str]:
    """Чтение списка заблокированных пар"""
    if not os.path.exists(BLACKLIST_PATH):
        return []
    with open(BLACKLIST_PATH, "r") as f:
        return [line.strip().upper() for line in f.readlines() if line.strip()]


def fetch_all_symbols() -> List[str]:
    """Получает все доступные пары с нужным квотом (например, USDT)"""
    if _client is None:
        raise RuntimeError("Pybit client не установлен. Используйте set_client().")

    try:
        symbols = _client.get_tickers(category="linear")["result"]["list"]
        return [
            s["symbol"] for s in symbols
            if any(q in s["symbol"] for q in QUOTE_ASSETS)
        ]
    except Exception as e:
        logger.warning(f"[pair_finder] Ошибка запроса символов: {e}")
        return []


def filter_symbols(symbols: List[str]) -> List[str]:
    """Фильтрация по объёму, длине истории, корректности данных и blacklist"""
    passed = []
    blacklist = set(load_blacklist())

    os.makedirs("logs", exist_ok=True)
    reasons = []

    for symbol in symbols:
        if symbol in blacklist:
            reasons.append((symbol, "blacklisted"))
            continue

        try:
            df = get_processed_ohlcv(symbol, interval=INTERVAL, limit=MIN_HISTORY_BARS)
            if len(df) < MIN_HISTORY_BARS:
                reasons.append((symbol, "short history"))
                continue

            if df.isnull().any().any() or (df["close"] <= 0).any():
                reasons.append((symbol, "invalid OHLCV"))
                continue

            avg_vol = df["volume"].rolling(50).mean().iloc[-1]
            if avg_vol < MIN_AVG_VOLUME:
                reasons.append((symbol, f"low volume: {avg_vol:.0f}"))
                continue

            passed.append(symbol)

        except Exception as e:
            reasons.append((symbol, f"error: {str(e)}"))
            continue

    # Логируем исключенные пары
    pd.DataFrame(reasons, columns=["symbol", "reason"]).to_csv("logs/filter_exclusions.csv", index=False)

    return passed


def get_candidate_pairs(force_refresh=False) -> List[str]:
    """Главная точка входа: возвращает отфильтрованные пары"""
    global _cached_pairs, _last_updated

    if not force_refresh and (time.time() - _last_updated < CACHE_TTL):
        return _cached_pairs

    logger.info("[pair_finder] Обновление списка пар...")
    all_symbols = fetch_all_symbols()
    filtered = filter_symbols(all_symbols)

    _cached_pairs = filtered
    _last_updated = time.time()

    logger.info(f"[pair_finder] Найдено {len(filtered)} подходящих пар.")
    return filtered
