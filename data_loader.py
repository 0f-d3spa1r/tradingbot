import time
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from typing import Optional
import logging

from feature_engineering import generate_ta_features

logger = logging.getLogger(__name__)

# Глобальный клиент (устанавливается через set_client)
_client: Optional[HTTP] = None

def set_client(client: HTTP) -> None:
    """Передача клиента Pybit из pipeline"""
    global _client
    _client = client


def fetch_ohlcv(symbol: str, interval: str, limit: int = 1500) -> pd.DataFrame:
    """
    Загружает исторические OHLCV-данные с Bybit Spot через pybit SDK.
    """
    assert _client is not None, "Pybit клиент не инициализирован. Вызовите set_client()."

    logger.info(f"[Pybit] Загрузка данных: {symbol}, {interval}, лимит {limit}")

    for attempt in range(3):
        try:
            response = _client.get_kline(
                category="spot",
                symbol=symbol,
                interval=interval.replace("m", ""),
                limit=limit
            )
            raw_data = response.get("result", {}).get("list", [])
            if not raw_data:
                raise ValueError("Pybit API вернул пустой список")
            break
        except Exception as e:
            logger.warning(f"[Pybit] Ошибка при загрузке данных (попытка {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError("Не удалось получить данные от Bybit (pybit)")

    # Конвертация в DataFrame
    df = pd.DataFrame(raw_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    logger.info(f"[Pybit] Загружено {len(df)} строк OHLCV.")

    return df


def get_processed_ohlcv(symbol: str, interval: str, limit: int = 1500) -> pd.DataFrame:
    """
    Загружает данные и добавляет технические индикаторы.
    """
    df = fetch_ohlcv(symbol, interval, limit)
    df = generate_ta_features(df)
    df.dropna(inplace=True)
    return df
