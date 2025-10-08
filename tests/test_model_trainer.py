# tests/test_model_trainer.py
import logging
import pandas as pd
import numpy as np

from config import BYBIT_API_KEY, BYBIT_API_SECRET
from pybit.unified_trading import HTTP
from data_loader import set_client, get_processed_ohlcv
from feature_engineering import select_features
from model_trainer import load_model_and_scaler, predict_on_batch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 0) Инициализируем Pybit клиент для data_loader
    assert BYBIT_API_KEY and BYBIT_API_SECRET, "Проверь .env / config: пустые BYBIT_API_KEY/BYBIT_API_SECRET"
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)

    # 1) Загружаем модель, скейлер и список категориальных фичей
    logger.info("🔹 Загружаем модель и скейлер...")
    model, scaler, cat_features = load_model_and_scaler()

    # 2) Берём свежие данные и готовим фичи
    logger.info("🔹 Загружаем тестовые данные (BTCUSDT, 15m)...")
    df = get_processed_ohlcv("BTCUSDT", "15", limit=500)
    X, _ = select_features(df)

    # 3) Собираем вход для модели как в проде: scale num + приклеить cat
    X_cat = X.select_dtypes(include=["object", "category"])
    X_num = X.select_dtypes(include=["number"])
    X_scaled = pd.DataFrame(scaler.transform(X_num), columns=X_num.columns, index=X_num.index)
    X_input = pd.concat([X_scaled, X_cat], axis=1)

    # 4) Прогноз
    preds, probs = predict_on_batch(model, X_input, cat_features=cat_features)

    logger.info("✅ Предсказания получены.")
    logger.info(f"Пример: preds[:5]={preds[:5]}")
    if isinstance(probs[0], (list, tuple, np.ndarray)):
        max_probs = [max(p) for p in probs[:5]]
    else:
        max_probs = probs[:5]
    logger.info(f"Пример: max proba[:5]= {max_probs}")


if __name__ == "__main__":
    main()
