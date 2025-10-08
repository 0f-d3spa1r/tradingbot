# pipeline.py
import os
import shutil
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import catboost as cb
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    ConfusionMatrixDisplay
)

from pybit.unified_trading import HTTP
from config import BYBIT_API_KEY, BYBIT_API_SECRET, CONFIDENCE_THRESHOLDS
from data_loader import set_client

from model_trainer import (
    prepare_data,
    optimize_catboost,
    train_final_model,
    rolling_cross_validation,
)

# --- setup ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def evaluate_model(model: cb.CatBoostClassifier, X_test, y_test, symbol: str, ts: str):
    """Считаем метрики, пишем отчёты и картинки с уникальными именами."""
    proba = model.predict_proba(X_test)
    confidence = np.max(proba, axis=1)
    y_pred = np.argmax(proba, axis=1)

    logger.info("=" * 30 + f" [FINAL METRICS] {symbol} " + "=" * 30)
    logger.info("[All] Accuracy: %.4f", accuracy_score(y_test, y_pred))
    logger.info("[All] F1 macro: %.4f", f1_score(y_test, y_pred, average="macro"))

    labels_order = [0, 1, 2]
    target_names = ["Down", "Up", "Neutral"]
    report = classification_report(
        y_test, y_pred, labels=labels_order, target_names=target_names, zero_division=0
    )
    (OUTPUT_DIR / f"{symbol}_{ts}_report.txt").write_text(report)

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="viridis")
    plt.title(f"Confusion Matrix ({symbol})")
    plt.savefig(OUTPUT_DIR / f"{symbol}_{ts}_confusion_matrix.png")
    plt.close()

    # метрики по порогам уверенности
    for thr in CONFIDENCE_THRESHOLDS:
        idx = confidence >= thr
        if np.any(idx):
            acc = accuracy_score(y_test[idx], y_pred[idx])
            f1 = f1_score(y_test[idx], y_pred[idx], average="macro")
            logger.info(f"[Conf >= {thr:.2f}] Accuracy: {acc:.4f}, F1_macro: {f1:.4f}")
        else:
            logger.warning(f"Нет уверенных прогнозов при пороге {thr:.2f}")

    # важности признаков (колонки CatBoost могут отличаться по названию — подстрахуемся)
    importances = model.get_feature_importance(prettified=True)
    plt.figure(figsize=(10, 6))
    feat_col = "Feature Id" if "Feature Id" in importances.columns else (
        "Feature" if "Feature" in importances.columns else importances.columns[0]
    )
    val_col = "Importances" if "Importances" in importances.columns else importances.columns[-1]
    plt.barh(importances[feat_col], importances[val_col])
    plt.title("CatBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{symbol}_{ts}_feature_importance.png")
    plt.close()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _archive_artifacts(symbol: str, ts: str):
    """
    Копируем свежие артефакты тренера в персональную папку:
    models/{symbol}/{ts}/(model.cbm, scaler.pkl, cat_features.pkl)
    + плоские файлы model_{symbol}.cbm, scaler_{symbol}.pkl, cat_features_{symbol}.pkl
    """
    # источники — где их сохраняет model_trainer
    model_src = MODEL_DIR / "saved_model.cbm"
    scaler_src = MODEL_DIR / "scaler.pkl"          # см. обновлённый save в prepare_data()
    cat_src = MODEL_DIR / "cat_features.pkl"

    target_dir = MODEL_DIR / symbol / ts
    target_dir.mkdir(parents=True, exist_ok=True)

    def _safe_copy(src: Path, dst: Path, label: str):
        try:
            if src.exists():
                shutil.copyfile(src, dst)
                logger.info(f"{label} скопирован в {dst}")
            else:
                logger.warning(f"{label} не найден: {src}")
        except Exception as e:
            logger.warning(f"Не удалось скопировать {label}: {e}")

    # в timestamp-папку
    _safe_copy(model_src, target_dir / "model.cbm", "Модель")
    _safe_copy(scaler_src, target_dir / "scaler.pkl", "Скейлер")
    _safe_copy(cat_src,    target_dir / "cat_features.pkl", "cat_features")

    # плоские алиасы по символу (удобно искать)
    _safe_copy(model_src, MODEL_DIR / f"model_{symbol}.cbm", "Модель (алиас)")
    _safe_copy(scaler_src, MODEL_DIR / f"scaler_{symbol}.pkl", "Скейлер (алиас)")
    _safe_copy(cat_src,    MODEL_DIR / f"cat_features_{symbol}.pkl", "cat_features (алиас)")

    return target_dir


def train_on_symbol(symbol: str, interval: str = "15", threshold: float = 0.0015, use_rolling_cv: bool = True):
    logger.info(f"⚙️ Обучение модели для {symbol}…")
    ts = _timestamp()

    # 1) данные
    X_train, X_test, y_train, y_test = prepare_data(symbol, interval, threshold)
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 2) быстрая оптимизация гиперов
    best_params = optimize_catboost(X_train, y_train)

    # 3) rolling CV (опционально)
    if use_rolling_cv:
        scores = rolling_cross_validation(X_train, y_train, best_params, n_splits=5)
        logger.info("📈 Rolling CV F1_macro: %.4f", float(np.mean(scores)))

    # 4) финальное обучение (model_trainer сам сохранит артефакты в models/)
    train_final_model(X_train, y_train, best_params)

    # 5) аккуратно архивируем свежие артефакты под символ/таймстемп
    run_dir = _archive_artifacts(symbol, ts)

    # 6) загружаем свежую модель для оценки
    model = cb.CatBoostClassifier()
    model.load_model(str(MODEL_DIR / "saved_model.cbm"))
    evaluate_model(model, X_test, y_test, symbol=symbol, ts=ts)

    logger.info(f"✅ Готово для {symbol}. Артефакты: {run_dir}")


if __name__ == "__main__":
    # Инициализация Bybit клиента (для загрузки OHLCV)
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)

    # Временно статический список — позже заменим на top_pairs из pair_discovery
    symbols = ["BTCUSDT", "ETHUSDT"]

    for sym in symbols:
        train_on_symbol(sym)
