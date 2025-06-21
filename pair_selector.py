# pair_selector.py
import os
from data_loader import get_processed_ohlcv
from model_trainer import load_model_and_scaler, predict_on_batch
import pandas as pd
import numpy as np
from typing import Dict, Union, Tuple, List
from ta.trend import EMAIndicator
import talib
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


def compute_pair_score(
    df: pd.DataFrame,
    model_data: Dict[str, list],
    weights: Dict[str, float] = None,
    return_details: bool = False,
    signal_threshold: float = 0.65,
    debug: bool = False
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Вычисляет скоринг торговой пары на основе технических и модельных факторов.

    Параметры:
        df (pd.DataFrame): Исторические данные OHLCV. Должен содержать колонки: 'close', 'volume', 'high', 'low'.
        model_data (Dict): {'confidences': List[float], 'predictions': List[int]}
        weights (Dict): Весовые коэффициенты:
            - 'volume': вес средней ликвидности (в млн)
            - 'atr': вес волатильности (ATR как % от цены)
            - 'ema_diff': вес расхождения EMA
            - 'adx': вес направленного движения
            - 'signals': вес частоты сигналов
            - 'conf_mean': вес средней уверенности
            - 'conf_std': вес разброса уверенности (отрицательный)
        return_details (bool): Если True — возвращает score и детали по метрикам.
        signal_threshold (float): Порог уверенности для учёта сигнала.
        debug (bool): Логировать подробности при return_details=True.

    Возвращает:
        float — итоговый score (или tuple с деталями, если включено).
    """

    default_weights = {
        "volume": 1.0,
        "atr": 1.5,
        "ema_diff": 1.0,
        "adx": 1.0,
        "signals": 1.0,
        "conf_mean": 2.0,
        "conf_std": -1.0
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
        avg_volume = volume.rolling(50).mean().iloc[-1]
        atr = talib.ATR(high, low, close, timeperiod=14).fillna(0).iloc[-1]
        ema_fast = EMAIndicator(close, window=5).ema_indicator()
        ema_slow = EMAIndicator(close, window=20).ema_indicator()
        safe_close = close.replace(0, np.nan).fillna(method="bfill")
        ema_diff = abs((ema_fast - ema_slow) / safe_close).fillna(0).iloc[-1]
        adx = talib.ADX(high, low, close, timeperiod=14).fillna(0).iloc[-1]

        # --- Модельные метрики ---
        confidences = pd.Series(model_data["confidences"]).fillna(0)
        signals_count = (confidences >= signal_threshold).sum()
        mean_conf = confidences.mean()
        std_conf = confidences.std()

        # --- Нормализация ---
        norm_volume = min(avg_volume / 1_000_000, 2)
        norm_atr = min(atr / close.iloc[-1], 0.05) * 100
        norm_ema = min(ema_diff, 0.05) * 100
        norm_adx = min(adx / 50, 1)
        norm_signals = min(signals_count / 10, 1)

        # --- Финальный скор ---
        score = (
            norm_volume * w["volume"] +
            norm_atr * w["atr"] +
            norm_ema * w["ema_diff"] +
            norm_adx * w["adx"] +
            norm_signals * w["signals"] +
            mean_conf * w["conf_mean"] +
            std_conf * w["conf_std"]
        )

        score = round(score, 4)

        if return_details:
            details = OrderedDict({
                "norm_volume": round(norm_volume, 4),
                "norm_atr": round(norm_atr, 4),
                "norm_ema": round(norm_ema, 4),
                "norm_adx": round(norm_adx, 4),
                "norm_signals": round(norm_signals, 4),
                "mean_conf": round(mean_conf, 4),
                "std_conf": round(std_conf, 4),
                "score": score
            })

            if debug:
                logger.info(f"[Score details] {details}")

            return score, details

        return score

    except Exception as e:
        logger.warning(f"[Score] Ошибка при расчёте score: {e}")
        return 0.0 if not return_details else (0.0, {"error": str(e)})


def evaluate_pairs(
    symbols: List[str],
    interval: str,
    top_n: int = 5,
    weights: Dict[str, float] = None,
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

    Параметры:
        symbols: Список тикеров
        interval: Интервал OHLCV (например, '15')
        top_n: Сколько пар вернуть
        weights: Кастомные веса для compute_pair_score
        signal_threshold: Порог уверенности для сигналов
        return_scores: Если True — вернуть пары вместе со score
        debug: Если True — логировать детали
        window: Сколько последних свечей взять для оценки
        save_results: Если True — сохранить top_pairs в CSV
        model, scaler: Опциональные предзагруженные модель и скейлер

    Возвращает:
        List[str] или List[Tuple[str, float]]
    """

    if model is None or scaler is None:
        model, scaler = load_model_and_scaler()

    pair_scores = []

    for symbol in symbols:
        try:
            df = get_processed_ohlcv(symbol, interval).tail(window)

            # Обработка признаков (импорт здесь, чтобы не грузить заранее)
            from feature_engineering import select_features
            X, _ = select_features(df)

            X_cat = X.select_dtypes(include=["object"])
            X_num = X.select_dtypes(include=["number"])

            X_scaled = pd.DataFrame(scaler.transform(X_num), columns=X_num.columns)
            X_input = pd.concat([X_scaled, X_cat.reset_index(drop=True)], axis=1)

            # Прогноз
            preds, confs = predict_on_batch(model, X_input)
            model_data = {
                "confidences": confs[-50:],  # последние 50 точек
                "predictions": preds[-50:]
            }

            score = compute_pair_score(
                df=df,
                model_data=model_data,
                weights=weights,
                signal_threshold=signal_threshold
            )

            if debug:
                logger.info(f"[✓] {symbol} — Score: {score:.4f}")

            pair_scores.append((symbol, score))

        except ValueError as e:
            logger.warning(f"[{symbol}] ValueError: {e}")
        except KeyError as e:
            logger.warning(f"[{symbol}] KeyError: отсутствует колонка {e}")
        except Exception as e:
            logger.exception(f"[{symbol}] Неизвестная ошибка")

    sorted_pairs = sorted(pair_scores, key=lambda x: x[1], reverse=True)
    top_pairs = sorted_pairs[:top_n]

    if debug:
        logger.info(f"[Top {top_n}] " + ", ".join(f"{s}:{score:.2f}" for s, score in top_pairs))

    if save_results:
        try:
            os.makedirs("outputs", exist_ok=True)
            pd.DataFrame(sorted_pairs, columns=["symbol", "score"]).to_csv("outputs/top_pairs.csv", index=False)
            logger.info("[✓] Результаты сохранены в outputs/top_pairs.csv")
        except Exception as e:
            logger.warning(f"[!] Не удалось сохранить файл: {e}")

    return top_pairs if return_scores else [s for s, _ in top_pairs]