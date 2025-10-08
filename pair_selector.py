# pair_selector.py
import os
import logging
from collections import OrderedDict
from typing import Dict, Union, Tuple, List, Optional

import numpy as np
import pandas as pd
import talib
from ta.trend import EMAIndicator
from pybit.unified_trading import HTTP

from data_loader import fetch_ohlcv
from model_trainer import load_model_and_scaler, predict_on_batch
from data_loader import set_client as set_data_client

logger = logging.getLogger(__name__)


# ============================================================
# 🧮 Основная функция скоринга пары
# ============================================================
def compute_pair_score(
    df: pd.DataFrame,
    model_data: Dict[str, list],
    weights: Optional[Dict[str, float]] = None,
    return_details: bool = False,
    signal_threshold: float = 0.65,
    debug: bool = False
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Рассчитывает скоринг пары на основе тех. метрик и данных модели.
    model_data: {'confidences': List[float], 'predictions': List[int]}
    """
    default_weights = {
        "volume": 1.0,
        "atr": 1.5,
        "ema_diff": 1.0,
        "adx": 1.0,
        "signals": 1.0,
        "conf_mean": 2.0,
        "conf_std": -1.0,
    }
    w = {**default_weights, **(weights or {})}

    try:
        close = df.get("close")
        volume = df.get("volume")
        high = df.get("high")
        low = df.get("low")

        if close is None or volume is None or high is None or low is None:
            raise ValueError("DataFrame missing one of required columns: ['close', 'volume', 'high', 'low']")

        if "confidences" not in model_data or not model_data["confidences"]:
            raise ValueError("model_data['confidences'] отсутствует или пуст")

        # --- Технические метрики ---
        avg_volume = volume.rolling(50, min_periods=50).mean().iloc[-1]
        atr = talib.ATR(high, low, close, timeperiod=14).fillna(0).iloc[-1]
        ema_fast = EMAIndicator(close, window=5).ema_indicator()
        ema_slow = EMAIndicator(close, window=20).ema_indicator()
        safe_close = close.replace(0, np.nan).fillna(method="bfill")
        ema_diff = abs((ema_fast - ema_slow) / safe_close).fillna(0).iloc[-1]
        adx = talib.ADX(high, low, close, timeperiod=14).fillna(0).iloc[-1]

        # --- Модельные метрики ---
        confidences = pd.Series(model_data["confidences"]).astype(float).fillna(0.0)
        signals_count = int((confidences >= signal_threshold).sum())
        mean_conf = float(confidences.mean())
        std_conf = float(confidences.std())

        # --- Нормализация ---
        norm_volume = min(float(avg_volume) / 1_000_000.0, 2.0)
        norm_atr = min(float(atr) / float(close.iloc[-1]), 0.05) * 100.0
        norm_ema = min(float(ema_diff), 0.05) * 100.0
        norm_adx = min(float(adx) / 50.0, 1.0)
        norm_signals = min(signals_count / 10.0, 1.0)

        # --- Финальный скор ---
        score = (
            norm_volume * w["volume"]
            + norm_atr * w["atr"]
            + norm_ema * w["ema_diff"]
            + norm_adx * w["adx"]
            + norm_signals * w["signals"]
            + mean_conf * w["conf_mean"]
            + std_conf * w["conf_std"]
        )
        score = round(float(score), 4)

        if return_details:
            details = OrderedDict(
                {
                    "norm_volume": round(norm_volume, 4),
                    "norm_atr": round(norm_atr, 4),
                    "norm_ema": round(norm_ema, 4),
                    "norm_adx": round(norm_adx, 4),
                    "norm_signals": round(norm_signals, 4),
                    "mean_conf": round(mean_conf, 4),
                    "std_conf": round(std_conf, 4),
                    "score": score,
                }
            )
            if debug:
                logger.info(f"[Score details] {details}")
            return score, details

        return score

    except Exception as e:
        logger.warning(f"[Score] Ошибка при расчёте score: {e}")
        return 0.0 if not return_details else (0.0, {"error": str(e)})


# ============================================================
# 🧠 Оценка нескольких пар (основная логика)
# ============================================================
def evaluate_pairs(
    symbols: List[str],
    interval: str,
    top_n: int = 5,
    weights: Optional[Dict[str, float]] = None,
    signal_threshold: float = 0.65,
    return_scores: bool = False,
    debug: bool = False,
    window: int = 150,
    save_results: bool = False,
    model=None,
    scaler=None
) -> Union[List[str], List[Tuple[str, float]]]:
    """
    Оценивает пары и возвращает топ-N по скору.
    """
    # === Загружаем модель и скейлер ===
    if model is None or scaler is None:
        model, scaler, cat_features = load_model_and_scaler()
    else:
        cat_features = None

    pair_scores: List[Tuple[str, float]] = []

    from feature_engineering import select_features

    for symbol in symbols:
        try:
            # 1️⃣ Данные: сырые свечи
            df = fetch_ohlcv(symbol, interval, limit=max(window, 60)).tail(window)

            # 2️⃣ Генерация признаков
            X, _ = select_features(df)

            # Ранние проверки
            if X.empty:
                if debug:
                    logger.warning(f"[{symbol}] Пустой набор признаков после feature engineering — пропускаю.")
                continue

            X_cat = X.select_dtypes(include=["object", "category"])
            X_num = X.select_dtypes(include=["number"])

            if X_num.empty:
                if debug:
                    logger.warning(f"[{symbol}] Нет числовых признаков (X_num.empty) — пропускаю.")
                continue

            # 3️⃣ Выравнивание под scaler.feature_names_in_
            if hasattr(scaler, "feature_names_in_"):
                for col in scaler.feature_names_in_:
                    if col not in X_num.columns:
                        X_num[col] = 0.0
                keep_order = [c for c in scaler.feature_names_in_ if c in X_num.columns]
                X_num = X_num.loc[:, keep_order]

            # 4️⃣ Скейлинг
            try:
                X_scaled = pd.DataFrame(
                    scaler.transform(X_num),
                    columns=X_num.columns,
                    index=X_num.index
                )
            except Exception as te:
                logger.warning(f"[{symbol}] Ошибка scaler.transform: {te} — пропускаю символ.")
                continue

            # 5️⃣ Финальный вход
            X_input = pd.concat(
                [X_scaled.reset_index(drop=True), X_cat.reset_index(drop=True)],
                axis=1
            )

            # Катфичи
            if cat_features is None:
                cat_features = X_input.select_dtypes(include=["object", "category"]).columns.tolist()

            # 6️⃣ Предсказания
            preds, confidences = predict_on_batch(model, X_input, cat_features=cat_features)
            model_data = {
                "confidences": confidences[-50:],
                "predictions": preds[-50:],
            }

            # 7️⃣ Скоринг
            score = compute_pair_score(
                df=df,
                model_data=model_data,
                weights=weights,
                signal_threshold=signal_threshold,
            )

            if debug:
                logger.info(f"[✓] {symbol} — Score: {score:.4f}")

            pair_scores.append((symbol, float(score)))

        except ValueError as e:
            logger.warning(f"[{symbol}] ValueError: {e}")
        except KeyError as e:
            logger.warning(f"[{symbol}] KeyError: отсутствует колонка {e}")
        except Exception as e:
            logger.exception(f"[{symbol}] Неизвестная ошибка")

    # 8️⃣ Сортировка и сохранение
    sorted_pairs = sorted(pair_scores, key=lambda x: x[1], reverse=True)
    top_pairs = sorted_pairs[:top_n]

    if debug:
        logger.info(f"[Top {top_n}] " + ", ".join(f"{s}:{score:.2f}" for s, score in top_pairs))

    if save_results:
        try:
            os.makedirs("outputs", exist_ok=True)
            ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
            out_path = f"outputs/top_pairs_{ts}.csv"
            pd.DataFrame(sorted_pairs, columns=["symbol", "score"]).to_csv(out_path, index=False)
            logger.info(f"[✓] Результаты сохранены: {out_path}")
        except Exception as e:
            logger.warning(f"[!] Не удалось сохранить файл: {e}")

    return top_pairs if return_scores else [s for s, _ in top_pairs]


# ============================================================
# 🎯 Обёртка: просто топ-N тикеров
# ============================================================
def rank_pairs(symbols: List[str], interval: str = "15", top_n: int = 5) -> List[str]:
    return evaluate_pairs(
        symbols=symbols,
        interval=interval,
        top_n=top_n,
        return_scores=False,
        debug=True,
    )


# ============================================================
# 🔌 Проксируем установку клиента
# ============================================================
def set_client(client: HTTP):
    """Передаёт клиент в data_loader для унификации."""
    set_data_client(client)
