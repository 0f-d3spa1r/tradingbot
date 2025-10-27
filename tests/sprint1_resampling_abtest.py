# tests/sprint1_resampling_abtest.py
import logging
import numpy as np
from pathlib import Path

import config as C
from pybit.unified_trading import HTTP
from data_loader import set_client
from model_trainer import prepare_data, optimize_catboost, train_final_model
from sklearn.metrics import f1_score, accuracy_score
import catboost as cb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("resampling_abtest")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

SYMBOL = "BTCUSDT"
INTERVAL = "15"

def evaluate_on_test(model, X_test, y_test, tag):
    proba = model.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)
    f1 = f1_score(y_test, pred, average="macro")
    acc = accuracy_score(y_test, pred)
    log.info(f"[{tag}] Acc={acc:.4f} F1_macro={f1:.4f}")
    return f1, acc

def run_variant(strategy: str):
    # Меняем стратегию на лету
    C.USE_RESAMPLING = (strategy != "none")
    C.RESAMPLING_STRATEGY = strategy

    X_train, X_test, y_train, y_test = prepare_data(SYMBOL, INTERVAL, threshold=0.0015)
    params = optimize_catboost(X_train, y_train)
    train_final_model(X_train, y_train, params)

    model = cb.CatBoostClassifier()
    model.load_model("models/saved_model.cbm")
    return evaluate_on_test(model, X_test, y_test, tag=f"strategy={strategy}")

def main():
    client = HTTP(api_key=C.BYBIT_API_KEY, api_secret=C.BYBIT_API_SECRET)
    set_client(client)

    results = {}
    for strat in ["smote", "undersample", "none"]:
        try:
            f1, acc = run_variant(strat)
            results[strat] = {"f1_macro": float(f1), "accuracy": float(acc)}
        except Exception as e:
            log.warning(f"Variant '{strat}' failed: {e}")

    # Быстрый вывод
    print("\n=== A/B results ===")
    for k, v in results.items():
        print(f"{k:12s} -> F1={v['f1_macro']:.4f}  Acc={v['accuracy']:.4f}")

    # Рекомендация
    if results:
        best = max(results.items(), key=lambda kv: kv[1]["f1_macro"])
        print(f"\n✅ Recommended: {best[0]} (by F1_macro)")
    else:
        print("\n⚠ No successful variants")

if __name__ == "__main__":
    main()
