import logging
import types
import pytest
from pybit.unified_trading import HTTP

import model_trainer as mt
import pipeline as pl

from data_loader import set_client
from config import BYBIT_API_KEY, BYBIT_API_SECRET

logging.basicConfig(level=logging.INFO)

@pytest.mark.smoke
def test_training_pipeline_fast(monkeypatch, tmp_path):
    # Инициализируем клиент
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)

    # 1) Стабы: быстрые параметры для CatBoost
    # 1) Стабы: быстрые параметры для CatBoost
    def stub_optimize(X_train, y_train):
        return {"depth": 4, "learning_rate": 0.12, "l2_leaf_reg": 3.0, "bagging_temperature": 0.2}

    monkeypatch.setattr(mt, "optimize_catboost", stub_optimize)
    monkeypatch.setattr(pl, "optimize_catboost", stub_optimize)  # <- важно!

    # 2) train_final_model — можно оставить как есть, но тоже замокаем в обоих пространствах
    orig_train_final = mt.train_final_model

    def stub_train_final(X_train, y_train, best_params, save_path=None):
        return orig_train_final(X_train, y_train, best_params)

    monkeypatch.setattr(mt, "train_final_model", stub_train_final)
    monkeypatch.setattr(pl, "train_final_model", stub_train_final)  # <- важно!

    # 3) Запускаем обучение для одного символа, без rolling-CV
    pl.train_on_symbol("BTCUSDT", interval="15", threshold=0.0015, use_rolling_cv=False)

    # 4) Проверяем артефакты
    # saved_model.cbm сохраняется в models/ твоим тренером + pipeline кладёт model_{symbol}.cbm
    import os
    assert os.path.exists("models/saved_model.cbm"), "Не найден saved_model.cbm"
    assert os.path.exists("models/cat_features.pkl"), "Не найден cat_features.pkl"
    assert os.path.exists("models/scaler.pkl") or os.path.exists("models/scaler_BTCUSDT.pkl"), "Не найден scaler"
