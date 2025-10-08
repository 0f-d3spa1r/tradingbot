import time
import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP

from feature_engineering import generate_ta_features

logger = logging.getLogger(__name__)

# Глобальный клиент (устанавливается через set_client)
_client: Optional[HTTP] = None

# Карта интервалов к формату Bybit Unified (spot)
_INTERVAL_MAP: Dict[str, str] = {
    # минуты
    "1": "1", "1m": "1",
    "3": "3", "3m": "3",
    "5": "5", "5m": "5",
    "15": "15", "15m": "15",
    "30": "30", "30m": "30",
    # часы
    "60": "60", "1h": "60",
    "120": "120", "2h": "120",
    "240": "240", "4h": "240",
    "360": "360", "6h": "360",
    "720": "720", "12h": "720",
    # дни/недели/месяцы
    "D": "D", "1d": "D", "1D": "D",
    "W": "W", "1w": "W", "1W": "W",
    "M": "M", "1M": "M",
}

def _normalize_interval(interval: str) -> str:
    iv = interval.strip()
    if iv not in _INTERVAL_MAP:
        # последний шанс: если чисто число минут в строке — принимаем
        if iv.isdigit():
            return iv
        raise ValueError(f"Unsupported interval: {interval!r}")
    return _INTERVAL_MAP[iv]


def set_client(client: HTTP) -> None:
    """Передача клиента Pybit из pipeline"""
    global _client
    _client = client


def get_client():
    """Возвращает текущий экземпляр Pybit клиента (или None)."""
    return _client


def fetch_ohlcv(symbol: str, interval: str, limit: int = 1500) -> pd.DataFrame:
    """
    Загружает исторические OHLCV-данные с Bybit Spot через pybit SDK.
    Возвращает DataFrame c индексом DatetimeIndex(tz=UTC), колонки: open, high, low, close, volume (float64).
    """
    if _client is None:
        raise RuntimeError("Pybit клиент не инициализирован. Вызовите set_client().")

    interval_norm = _normalize_interval(interval)
    logger.info(f"[Pybit] Загрузка данных: {symbol}, interval={interval_norm}, limit={limit}")

    last_err = None
    for attempt in range(3):
        try:
            response = _client.get_kline(
                category="spot",
                symbol=symbol,
                interval=interval_norm,
                limit=limit,
            )
            raw = response.get("result", {}).get("list", [])
            if not raw:
                raise ValueError("Pybit API вернул пустой список kline")
            # DataFrame: Bybit отдаёт [timestamp, open, high, low, close, volume, turnover]
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            # timestamp → UTC
            # иногда timestamp приходит строкой; приводим к int64 ms безопасно
            ts = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
            df = df.assign(timestamp=ts).set_index("timestamp")
            # только нужные колонки
            df = df[["open", "high", "low", "close", "volume"]].astype("float64")

            # Сортировка по времени (Bybit часто отдаёт «свежее→старое»)
            df.sort_index(inplace=True)
            # Удалим дубликаты индекса, если вдруг
            df = df[~df.index.duplicated(keep="last")]

            # Базовая валидация
            bad_mask = (
                ~np.isfinite(df[["open", "high", "low", "close", "volume"]]).all(axis=1) |
                (df[["open", "high", "low", "close"]] <= 0).any(axis=1) |
                (df["volume"] < 0)
            )
            bad_count = int(bad_mask.sum())
            if bad_count:
                logger.warning(f"[Pybit] Отфильтровано «плохих» баров: {bad_count}")
                df = df[~bad_mask]

            if df.empty:
                raise ValueError("После фильтрации данных не осталось ни одной свечи")

            logger.info(f"[Pybit] Загружено {len(df)} строк OHLCV (UTC, ascending).")
            return df
        except Exception as e:
            last_err = e
            logger.warning(f"[Pybit] Ошибка при загрузке (попытка {attempt+1}/3): {e}")
            # экспоненциальный бэкоф с небольшим джиттером
            time.sleep(3 * (attempt + 1))
    # если не вышло
    raise RuntimeError(f"Не удалось получить данные от Bybit (pybit). Последняя ошибка: {last_err}")


def get_processed_ohlcv(symbol: str, interval: str, limit: int = 1500, with_ta: bool = True) -> pd.DataFrame:
    """
    Загружает OHLCV и (опционально) добавляет технические индикаторы.
    Возвращает очищенный DataFrame без NaN.
    """
    df = fetch_ohlcv(symbol, interval, limit)
    if with_ta:
        df = generate_ta_features(df)
    # удаляем NaN, которые могли появиться после TA-индикаторов
    df = df.dropna()
    return df
