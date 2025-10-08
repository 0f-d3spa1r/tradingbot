import os
import logging
import numpy as np
import pandas as pd
from pybit.unified_trading import HTTP
from data_loader import set_client
from model_trainer import (
    prepare_data,
    optimize_catboost,
    train_final_model,
    load_model_and_scaler,
    predict_on_batch
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_smoke_test():
    print("\n=== 🚀 TradingBot Smoke Test ===\n")

    client = HTTP()  # публичные свечи не требуют API ключей
    set_client(client)

    symbol = "BTCUSDT"
    interval = "15"
    print(f"Fetching data for {symbol} ({interval}m)...")

    # 1️⃣ Получаем данные
    X_train, X_test, y_train, y_test = prepare_data(symbol, interval, threshold=0.0015)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("Feature columns:", len(X_train.columns))

    # 2️⃣ Проверяем оптимизацию (коротко)
    print("\nRunning quick Bayesian optimization (short)...")
    best_params = optimize_catboost(X_train.head(300), y_train.head(300))
    print("Best params:", best_params)

    # 3️⃣ Финальное обучение (на ограниченном объёме для скорости)
    print("\nTraining final model...")
    train_final_model(X_train.head(500), y_train.head(500), best_params)

    # 4️⃣ Проверка загрузки модели и скейлера
    model, scaler, cat_features = load_model_and_scaler()
    print(f"Loaded model OK. Cat features: {cat_features}")

    # 5️⃣ Предсказание на батче
    preds, confs = predict_on_batch(model, X_test.head(10), cat_features)
    print("\nSample predictions:")
    for i, (p, c) in enumerate(zip(preds, confs)):
        print(f"  Row {i+1}: class={p}, conf={c:.4f}")

    print("\n✅ Smoke test complete — everything appears functional.\n")

if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)
    run_smoke_test()
