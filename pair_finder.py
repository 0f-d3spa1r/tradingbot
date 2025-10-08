# pair_finder.py  (замени содержимое на это, если удобно; либо примени только отмеченные блоки)

import os
import time
import logging
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pybit.unified_trading import HTTP

from config import (
    MIN_HISTORY_BARS,
    MIN_AVG_VOLUME,
    QUOTE_ASSETS,
    INTERVAL,
    BLACKLIST_PATH,
)

from data_loader import fetch_ohlcv  # ⬅️ используем сырые свечи (БЕЗ TA, БЕЗ dropna)

logger = logging.getLogger(__name__)

_cached_pairs: List[str] = []
_last_updated: float = 0.0
CACHE_TTL = 60 * 15  # 15 минут

_client: Optional[HTTP] = None


def set_client(client: HTTP):
    global _client
    _client = client


def load_blacklist() -> List[str]:
    if not os.path.exists(BLACKLIST_PATH):
        return []
    with open(BLACKLIST_PATH, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]


# --- Внутреннее: безопасно получаем tickers (spot) ---
def _parse_tickers() -> List[Dict]:
    if _client is None:
        raise RuntimeError("Pybit client не установлен. Используйте set_client().")
    try:
        res = _client.get_tickers(category="spot")
        return res.get("result", {}).get("list", []) or []
    except Exception as e:
        logger.warning(f"[pair_finder] Ошибка get_tickers: {e}")
        return []


def fetch_all_symbols() -> List[str]:
    """
    Возвращаем пары только с нужным квотом. Используем endswith для надёжности.
    """
    tickers = _parse_tickers()
    if not tickers:
        return []
    syms = []
    for t in tickers:
        sym = t.get("symbol", "")
        if any(sym.endswith(q) for q in QUOTE_ASSETS):
            syms.append(sym)
    return syms


def _prefilter_by_24h_volume(symbols: List[str], min_turnover: float) -> List[str]:
    """
    Быстрый предфильтр по суточным метрикам из get_tickers() — экономит вызовы kline.
    Сравниваем по turnover24h (в валюте котировки). Если парсинг не удался — не режем (пройдёт на свечи).
    """
    tickers = _parse_tickers()
    by_symbol = {t.get("symbol", ""): t for t in tickers}
    out = []
    for s in symbols:
        t = by_symbol.get(s)
        if not t:
            continue
        try:
            turnover24h = float(t.get("turnover24h", 0.0))
            if turnover24h >= min_turnover:
                out.append(s)
        except Exception:
            out.append(s)
    return out


def filter_symbols(symbols: List[str]) -> List[str]:
    """
    Фильтрация по истории/валидности/объёму на свечах.
    Используем ЧИСТЫЙ OHLCV (без TA), чтобы не терять бары из-за dropna().
    """
    passed: List[str] = []
    blacklist = set(load_blacklist())

    os.makedirs("logs", exist_ok=True)
    reasons: List[Tuple[str, str, str]] = []  # (symbol, reason, extra)

    # Быстрый отсев по 24h обороту (порог условный: MIN_AVG_VOLUME * 50; при необходимости калибруем)
    symbols_fast = _prefilter_by_24h_volume(symbols, min_turnover=MIN_AVG_VOLUME * 50)

    for symbol in symbols_fast:
        if symbol.upper() in blacklist:
            reasons.append((symbol, "blacklisted", ""))
            continue

        try:
            # Загружаем сырые бары. Берём минимум MIN_HISTORY_BARS, но не меньше 60 для устойчивости rolling.
            limit = max(MIN_HISTORY_BARS, 60)
            df = fetch_ohlcv(symbol, interval=INTERVAL, limit=limit)

            if len(df) < MIN_HISTORY_BARS:
                reasons.append((symbol, "short_history", f"len={len(df)}"))
                continue

            # Базовая валидность
            if df.isnull().any().any() or (df["close"] <= 0).any() or (df["volume"] < 0).any():
                reasons.append((symbol, "invalid_ohlcv", "nan/neg/zero"))
                continue

            # Средний объём по последним 50 барам (на сырых данных)
            avg_vol = df["volume"].rolling(50, min_periods=50).mean().iloc[-1]
            if pd.isna(avg_vol) or avg_vol < MIN_AVG_VOLUME:
                reasons.append((symbol, "low_volume", f"{avg_vol:.0f}" if pd.notna(avg_vol) else "nan"))
                continue

            passed.append(symbol)

        except Exception as e:
            reasons.append((symbol, "error", str(e)))
            continue

    # Лог исключений (создаём файл даже если пусто)
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    df_reasons = pd.DataFrame(reasons, columns=["symbol", "reason", "extra"])
    df_reasons.insert(0, "ts_utc", ts)
    df_reasons.insert(1, "interval", INTERVAL)
    df_reasons.to_csv("logs/filter_exclusions.csv", index=False)

    return passed


def get_candidate_pairs(force_refresh: bool = False) -> List[str]:
    global _cached_pairs, _last_updated

    if _cached_pairs and not force_refresh and (time.time() - _last_updated < CACHE_TTL):
        return _cached_pairs

    logger.info("[pair_finder] Обновление списка пар...")
    all_symbols = fetch_all_symbols()
    filtered = filter_symbols(all_symbols)

    _cached_pairs = filtered
    _last_updated = time.time()

    logger.info(f"[pair_finder] Найдено {len(filtered)} подходящих пар.")
    return filtered
