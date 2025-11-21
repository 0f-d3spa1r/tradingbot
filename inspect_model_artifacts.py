# inspect_model_artifacts.py
import argparse
import json
import logging
import os
from pathlib import Path

import pickle
import numpy as np

try:
    import catboost as cb
except ImportError:
    cb = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"


def _safe_load_pickle(path: Path, label: str):
    if not path.exists():
        logger.warning(f"[{label}] файл не найден: {path}")
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"[{label}] загружен: {path}")
        return obj
    except Exception as e:
        logger.warning(f"[{label}] ошибка загрузки {path}: {e}")
        return None


def _safe_load_json(path: Path, label: str):
    if not path.exists():
        logger.warning(f"[{label}] файл не найден: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        logger.info(f"[{label}] загружен: {path}")
        return obj
    except Exception as e:
        logger.warning(f"[{label}] ошибка загрузки {path}: {e}")
        return None


def _find_latest_run_dir(symbol: str) -> Path | None:
    base = MODEL_DIR / symbol
    if not base.exists() or not base.is_dir():
        logger.warning(f"Для символа {symbol} нет папки {base}")
        return None

    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        logger.warning(f"Для символа {symbol} нет ни одного run_dir в {base}")
        return None

    # сортируем по имени (ts в формате YYYYMMDD_HHMMSS → лексикографически ок)
    latest = sorted(candidates, key=lambda p: p.name)[-1]
    logger.info(f"Найден последний run_dir для {symbol}: {latest}")
    return latest


def inspect_run(symbol: str, ts: str | None):
    """
    Инспектирует артефакты уровня run_dir: models/<symbol>/<ts>/
    Если ts=None, берёт последний по имени.
    """
    if ts is None:
        run_dir = _find_latest_run_dir(symbol)
        if run_dir is None:
            logger.error("Не удалось найти run_dir — выходим.")
            return
    else:
        run_dir = MODEL_DIR / symbol / ts
        if not run_dir.exists():
            logger.error(f"Указанный run_dir не существует: {run_dir}")
            return
        logger.info(f"Использую run_dir: {run_dir}")

    model_path = run_dir / "model.cbm"
    feat_path = run_dir / "feature_columns.pkl"
    cat_path = run_dir / "cat_features.pkl"
    scaler_path = run_dir / "scaler.pkl"
    temp_path = run_dir / "temperature.json"
    calib_path = run_dir / "confidence_calibrator.pkl"

    print("\n" + "=" * 80)
    print(f"🧩 ARTEFACTS FOR SYMBOL={symbol}, RUN={run_dir.name}")
    print("=" * 80 + "\n")

    # --- feature_columns ---
    feature_columns = _safe_load_pickle(feat_path, "feature_columns")
    if feature_columns is not None:
        print(f"feature_columns.pkl → type={type(feature_columns)}, len={len(feature_columns)}")
        if isinstance(feature_columns, list):
            preview = feature_columns[:20]
            print(f"  first 20 features: {preview}")
        print()

    # --- cat_features ---
    cat_features = _safe_load_pickle(cat_path, "cat_features")
    if cat_features is not None:
        print(f"cat_features.pkl → type={type(cat_features)}, len={len(cat_features)}")
        if isinstance(cat_features, list):
            preview = cat_features[:20]
            print(f"  cat feature names: {preview}")
        print()

    # --- scaler ---
    scaler = _safe_load_pickle(scaler_path, "scaler")
    if scaler is not None:
        print(f"scaler.pkl → type={type(scaler)}")
        if hasattr(scaler, "feature_names_in_"):
            names = list(scaler.feature_names_in_)
            print(f"  feature_names_in_ (len={len(names)}): {names}")
        else:
            print("  ⚠ scaler has no feature_names_in_")
        print()

    # --- temperature ---
    temp_json = _safe_load_json(temp_path, "temperature")
    if temp_json is not None:
        T = temp_json.get("T", None)
        print(f"temperature.json → {temp_json}")
        print(f"  parsed T = {T}")
        print()

    # --- calibrator ---
    calibrator = _safe_load_pickle(calib_path, "confidence_calibrator")
    if calibrator is not None:
        print(f"confidence_calibrator.pkl → type={type(calibrator)}")
        # небольшая диагностика
        attrs = [a for a in dir(calibrator) if a.endswith("_min_") or a.endswith("_max_")]
        if attrs:
            print(f"  attrs: {attrs}")
        print()

    # --- model.cbm ---
    if not model_path.exists():
        logger.warning(f"[model] файл не найден: {model_path}")
    else:
        print(f"model.cbm → {model_path}")
        if cb is None:
            print("  ⚠ catboost не установлен в этом окружении — пропускаю загрузку модели.")
        else:
            try:
                model = cb.CatBoostClassifier()
                model.load_model(str(model_path))
                params = model.get_params()
                print("  model params (subset):")
                for k in ["iterations", "depth", "learning_rate", "loss_function"]:
                    if k in params:
                        print(f"    {k}: {params[k]}")
                # shape вероятностей, если хотим
                try:
                    # сделаем фейковый input из одного нулевого объекта
                    if feature_columns and isinstance(feature_columns, list):
                        import pandas as pd
                        X_dummy = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)
                        proba = model.predict_proba(X_dummy)
                        print(f"  predict_proba(dummy) shape: {np.asarray(proba).shape}")
                except Exception as e:
                    print(f"  (diagnostic predict_proba failed: {e})")
            except Exception as e:
                print(f"  ⚠ ошибка при загрузке модели: {e}")
        print()

    print("=" * 80)
    print("✅ Инспекция завершена.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Инспекция артефактов модели (feature_columns, cat_features, scaler, T, calibrator, model)."
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Тикер/символ, например BTCUSDT или ETHUSDT",
    )
    parser.add_argument(
        "--ts",
        type=str,
        default=None,
        help="Таймстемп прогона (подпапка в models/<symbol>/). Если не указан — берётся последний.",
    )

    args = parser.parse_args()
    inspect_run(symbol=args.symbol, ts=args.ts)


if __name__ == "__main__":
    main()

#python inspect_model_artifacts.py --symbol ETHUSDT
#python inspect_model_artifacts.py --symbol BTCUSDT