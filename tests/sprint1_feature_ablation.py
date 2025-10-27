# tests/sprint1_feature_ablation.py
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from pybit.unified_trading import HTTP
from config import BYBIT_API_KEY, BYBIT_API_SECRET
from data_loader import set_client, fetch_ohlcv
from feature_engineering import select_features
from model_trainer import optimize_catboost, train_final_model
import catboost as cb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("feature_ablation")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

SYMBOL = "BTCUSDT"
INTERVAL = "15"
WINDOW = 800   # больше — лучше, но дольше
DROP_CANDIDATES = [
    "rolling_volume_5",
    "rolling_return_5",
    "cmo",         # пример: часто «шумный»
    "tsi",         # если окажется лишним — выстрелим
    # добавляй при желании
]

def train_eval(X, y, tag):
    split = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # скейл только числовые
    num_cols = X_train.select_dtypes(include=["number"]).columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

    scaler = StandardScaler()
    X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    X_test_num  = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

    X_train_final = pd.concat([X_train_num, X_train[cat_cols].reset_index(drop=True)], axis=1)
    X_test_final  = pd.concat([X_test_num,  X_test[cat_cols].reset_index(drop=True)], axis=1)

    params = optimize_catboost(X_train_final, y_train)
    train_final_model(X_train_final, y_train, params)

    model = cb.CatBoostClassifier()
    model.load_model("models/saved_model.cbm")
    proba = model.predict_proba(X_test_final)
    pred = np.argmax(proba, axis=1)
    f1 = f1_score(y_test, pred, average="macro")
    acc = accuracy_score(y_test, pred)
    log.info(f"[{tag}] Acc={acc:.4f} F1_macro={f1:.4f}")
    return f1, acc

def main():
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)

    # сырые данные → фичи
    df = fetch_ohlcv(SYMBOL, INTERVAL, limit=WINDOW)
    X_full, y_full = select_features(df)

    # Базовый результат
    base_f1, base_acc = train_eval(X_full, y_full, tag="BASE")

    # Абляция: срезаем кандидатов (если колонка существует)
    X_ab = X_full.copy()
    drop_these = [c for c in DROP_CANDIDATES if c in X_ab.columns]
    X_ab = X_ab.drop(columns=drop_these, errors="ignore")
    abl_f1, abl_acc = train_eval(X_ab, y_full, tag=f"ABLATE({','.join(drop_these) or '-'})")

    print("\n=== Ablation result ===")
    print(f"BASE         -> F1={base_f1:.4f} Acc={base_acc:.4f}")
    print(f"ABLATE({','.join(drop_these) or '-'}) -> F1={abl_f1:.4f} Acc={abl_acc:.4f}")

if __name__ == "__main__":
    main()
