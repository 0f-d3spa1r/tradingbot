# tests/sprint1_threshold_search.py
import json
import logging
import numpy as np
from pathlib import Path

from pybit.unified_trading import HTTP
from config import BYBIT_API_KEY, BYBIT_API_SECRET
from data_loader import set_client
from model_trainer import prepare_data, load_model_and_scaler
from confidence_calibrator import load_calibrator, apply_calibration
from sklearn.metrics import f1_score, accuracy_score
import os


os.chdir(os.path.dirname(os.path.dirname(__file__)))  # перейти на уровень выше /tests/
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("threshold_search")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

SYMBOL = "BTCUSDT"   # можешь сменить
INTERVAL = "15"
GRID = [round(x, 2) for x in np.linspace(0.50, 0.80, 16)]  # 0.50..0.80 шаг 0.02

def main():
    # API
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)

    # Данные (берём тест из prepare_data)
    X_train, X_test, y_train, y_test = prepare_data(SYMBOL, INTERVAL, threshold=0.0015)

    # Загружаем модель и калибратор
    model, scaler, cat_features = load_model_and_scaler()
    cal = load_calibrator("models/confidence_calibrator.pkl")

    # Предсказания на тесте
    from model_trainer import predict_on_batch
    preds, conf = predict_on_batch(model, X_test, cat_features=cat_features)

    conf = np.asarray(conf)
    if conf.ndim == 2:  # безопасность
        conf = conf.max(axis=1)
    conf = apply_calibration(cal, conf)

    preds = np.asarray(preds).astype(int).ravel()

    # Считаем метрики по сетке порогов (F1_macro и coverage)
    rows = []
    for th in GRID:
        mask = conf >= th
        if mask.sum() == 0:
            rows.append({"threshold": th, "coverage": 0.0, "f1_macro": None, "accuracy": None})
            continue
        f1 = f1_score(y_test[mask], preds[mask], average="macro")
        acc = accuracy_score(y_test[mask], preds[mask])
        cov = float(mask.mean())
        rows.append({"threshold": th, "coverage": cov, "f1_macro": float(f1), "accuracy": float(acc)})
        log.info(f"th={th:.2f}: coverage={cov:.3f}, F1_macro={f1:.4f}, Acc={acc:.4f}")

    # Лучший по F1_macro (при разумном покрытии ≥ 0.15)
    valid = [r for r in rows if (r["f1_macro"] is not None and r["coverage"] >= 0.15)]
    best = max(valid, key=lambda r: r["f1_macro"]) if valid else max(rows, key=lambda r: (r["f1_macro"] or 0))

    out_json = OUT / "threshold_search.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"symbol": SYMBOL, "interval": INTERVAL, "grid": rows, "best": best}, f, indent=2)
    print(f"\n✅ Best threshold: {best['threshold']:.2f} | F1_macro={best['f1_macro']} | coverage={best['coverage']:.3f}")
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    main()
