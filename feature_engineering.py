import pandas as pd
import numpy as np
from typing import Tuple
import logging

import ta
import talib
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

TA_FEATURES = [
    "macd", "macd_signal", "rsi", "adx", "cci", "atr",
    "ema_fast", "ema_slow", "ema_cross",
    "bb_upper", "bb_lower",
    "stoch_rsi", "williams_r", "tsi", "cmo", "trix",
    "plus_di", "minus_di",
    "rolling_return_5", "rolling_volume_5"
]

CANDLE_PATTERNS = {
    "is_doji": talib.CDLDOJI
}


def generate_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    df["macd"] = ta.trend.macd(df["close"]).fillna(0)
    df["macd_signal"] = ta.trend.macd_signal(df["close"]).fillna(0)
    df["rsi"] = ta.momentum.rsi(df["close"]).fillna(0)
    df["cci"] = ta.trend.cci(df["high"], df["low"], df["close"]).fillna(0)
    df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"]).fillna(0)

    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=12).fillna(0)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=26).fillna(0)
    df["ema_cross"] = (df["ema_fast"] > df["ema_slow"]).astype(int)

    bb = ta.volatility.BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband().fillna(0)
    df["bb_lower"] = bb.bollinger_lband().fillna(0)

    df["stoch_rsi"] = ta.momentum.stochrsi(df["close"]).fillna(0)
    df["williams_r"] = ta.momentum.williams_r(df["high"], df["low"], df["close"]).fillna(0)
    df["tsi"] = ta.momentum.tsi(df["close"]).fillna(0)
    df["trix"] = ta.trend.trix(df["close"]).fillna(0)

    # ✅ CMO через talib (исправлено)
    df["cmo"] = talib.CMO(df["close"], timeperiod=14)
    df["cmo"] = df["cmo"].fillna(0)

    # plus_di / minus_di с устойчивым rolling
    high = df["high"]
    low = df["low"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    tr = ta.volatility.AverageTrueRange(high, low, df["close"]).average_true_range().fillna(1)

    df["plus_di"] = 100 * (plus_dm.rolling(window=14, min_periods=3).mean() / tr).fillna(0)
    df["minus_di"] = 100 * (minus_dm.rolling(window=14, min_periods=3).mean() / tr).fillna(0)

    return df


def generate_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    for name, func in CANDLE_PATTERNS.items():
        df[name] = (func(df["open"], df["high"], df["low"], df["close"]) != 0).astype(int)
    return df


def generate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["session"] = pd.cut(
        df["hour"],
        bins=[0, 6, 13, 21, 24],
        labels=["Asia_1", "Europe", "US", "Asia_2"],
        right=False
    ).astype(str)  # 👈 добавь это

    return df



def generate_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["volatility_1"] = (df["high"] - df["low"]) / df["close"]
    df["volatility_3"] = df["close"].rolling(3).std()
    df["volume_change_1"] = df["volume"].pct_change(1)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(5).mean()

    # ➕ Новые сглаженные признаки
    df["rolling_return_5"] = df["close"].pct_change().rolling(5).mean().fillna(0)
    df["rolling_volume_5"] = df["volume"].rolling(5).mean().fillna(0)

    return df


def generate_clustering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rolling = df[["open", "high", "low", "close", "volume"]].rolling(window=5).mean()
    rolling.dropna(inplace=True)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rolling)

    label_series = pd.Series(labels, index=rolling.index, name="cluster_label")
    df = df.drop(columns=["cluster_label"], errors="ignore")
    df.loc[rolling.index, "cluster_label"] = label_series
    df["cluster_label"] = df["cluster_label"].astype(str)

    return df


def generate_target(df: pd.DataFrame, threshold: float = 0.0015) -> pd.DataFrame:
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1
    df["target"] = df["future_return"].apply(
        lambda x: 1 if x > threshold else (0 if x < -threshold else 2)
    )
    return df


def select_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "target" not in df.columns:
        logger.warning("`target` не найден, вызывается generate_target() с default threshold.")
        df = generate_target(df)

    df = generate_ta_features(df)
    df = generate_candle_patterns(df)
    df = generate_time_features(df)
    df = generate_derivatives(df)
    df = generate_clustering(df)

    # Удаление NaN до фильтрации
    df.dropna(inplace=True)

    X = df.drop(columns=["target", "future_return", "return", "future_close"], errors="ignore")
    y = df["target"]

    # ➕ Обработка бесконечных значений
    X = X.replace([np.inf, -np.inf], np.nan)

    # Удалим строки с NaN (после замены inf)
    X.dropna(inplace=True)

    # Синхронизируем y с очищенными X
    y = y.loc[X.index]

    # Финальные проверки
    assert not X.isnull().any().any(), "NaN в X после очистки"
    assert not y.isnull().any(), "NaN в y после очистки"

    return X, y
