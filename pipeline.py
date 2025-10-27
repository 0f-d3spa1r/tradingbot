# pipeline.py
import os
import pandas as pd
import shutil
import pickle
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
from confidence_calibrator import fit_confidence_calibrator, save_calibrator
import json
from datetime import datetime
from config import EMBARGO_BARS, MIN_CV_TRAIN, MIN_CV_VAL



# --- setup ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# === Align helper: ensures test/holdout columns match training features ===
import pickle
import catboost as cb

def _align_for_infer(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """
    Приводит входной DF к тому же набору/порядку признаков, что и при обучении:
    - пробует загрузить models/feature_columns.pkl (если делался feature bagging)
    - фильтрует/переупорядочивает столбцы
    - санитизирует категориальные (string + fillna("__NA__"))
    - возвращает df_aligned и индексы cat_features для CatBoost Pool
    """
    keep_cols = None
    try:
        with open("models/feature_columns.pkl", "rb") as f:
            keep_cols = pickle.load(f)
    except Exception:
        keep_cols = df.columns.tolist()
        logger.warning("[InferAlign] models/feature_columns.pkl not found — using current columns order")

    # filter + reorder
    cols = [c for c in keep_cols if c in df.columns]
    if len(cols) < len(keep_cols):
        missing = [c for c in keep_cols if c not in df.columns]
        logger.warning("[InferAlign] %d feature(s) missing in input: %s", len(missing), missing[:10])

    df2 = df[cols].copy()

    # sanitize categoricals
    cat_cols = df2.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_cols:
        df2[cat_cols] = df2[cat_cols].astype("string").fillna("__NA__")

    cat_idx = [df2.columns.get_loc(c) for c in cat_cols]
    logger.info("[InferAlign] kept %d features; cat=%d", len(cols), len(cat_idx))
    return df2, cat_idx



def _apply_temperature_scaling(proba: np.ndarray, T: float) -> np.ndarray:
    """Softmax temperature scaling for multi-class, from probabilities (logit-free)."""
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    logits_T = logits / max(T, 1e-6)
    m = np.max(logits_T, axis=1, keepdims=True)
    exp = np.exp(logits_T - m)
    return exp / np.sum(exp, axis=1, keepdims=True)

def _nll_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))

def _find_best_temperature(proba_val: np.ndarray, y_val: np.ndarray,
                           t_min: float, t_max: float, t_step: float) -> float:
    best_T, best_nll = 1.0, _nll_multiclass(y_val, proba_val)
    T = t_min
    while T <= t_max + 1e-9:
        proba_T = _apply_temperature_scaling(proba_val, T)
        nll = _nll_multiclass(y_val, proba_T)
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
        T += t_step
    return best_T


def find_threshold_for_precision(y_true, proba, target_precision=0.6):
    """
    Подбирает минимальный порог уверенности, при котором precision >= target_precision.
    Возвращает (threshold, coverage).
    """
    from sklearn.metrics import precision_score

    conf = proba.max(axis=1)
    y_pred = proba.argmax(axis=1)
    for th in np.linspace(0.9, 0.3, 13):  # шаг 0.05 вниз (0.9 → 0.3)
        m = conf >= th
        if not m.any():
            continue
        prec = precision_score(y_true[m], y_pred[m], average="macro", zero_division=0)
        if prec >= target_precision:
            return th, m.mean()  # найденный порог и доля охвата
    return None, 0.0


# ===== Inference alignment helpers =====
def _load_feature_and_cat_lists(symbol: str | None = None):
    """
    Грузим список колонок и кат-фичей, сохранённых при обучении.
    Приоритет: models/<SYMBOL>/<ts>/feature_columns.pkl (если хочешь — добавь),
               models/feature_columns.pkl,
               models/feature_columns_<SYMBOL>.pkl (если ведёшь по-символьно).
    Возвращает: (feature_columns: list[str] | None, cat_features: list[str] | None)
    """
    candidates = []
    if symbol:
        candidates += [
            MODEL_DIR / f"feature_columns_{symbol}.pkl",
        ]
    candidates += [
        MODEL_DIR / "feature_columns.pkl",
    ]
    feat_cols = None
    for p in candidates:
        if p.exists():
            with open(p, "rb") as f:
                feat_cols = pickle.load(f)
            break

    # cat_features
    cat_list = None
    cat_paths = []
    if symbol:
        cat_paths += [MODEL_DIR / f"cat_features_{symbol}.pkl"]
    cat_paths += [MODEL_DIR / "cat_features.pkl"]
    for p in cat_paths:
        if p.exists():
            with open(p, "rb") as f:
                cat_list = pickle.load(f)
            break

    return feat_cols, cat_list


def _align_for_infer(X: pd.DataFrame, symbol: str | None = None):
    """
    Делает входной X совместимым с моделью:
    - выбирает ТЕ ЖЕ колонки в ТОМ ЖЕ порядке, что и на обучении;
    - добавляет недостающие колонки (0.0 для числовых, '__NA__' для категориальных);
    - приводит категориальные к pandas StringDtype;
    - возвращает (X_aligned, cat_idx).
    """
    X_in = X.copy()

    # 1) грузим эталонные списки
    feat_cols, cat_features = _load_feature_and_cat_lists(symbol)

    # если список колонок не найден — используем то, что пришло (но это риск!)
    if feat_cols is None:
        feat_cols = X_in.columns.tolist()

    # 2) гарантируем наличие всех нужных колонок
    #    отсутствующие — создаём (числовые -> 0.0; категориальные -> "__NA__")
    exist_cols = set(X_in.columns)
    need_cols = list(feat_cols)

    # если список cat_features не найден — определим по dtype входного X (best-effort)
    if cat_features is None:
        cat_features = X_in.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    cat_set = set(cat_features)

    for col in need_cols:
        if col not in exist_cols:
            if col in cat_set:
                X_in[col] = "__NA__"
            else:
                X_in[col] = 0.0

    # 3) режем лишние колонки и ставим ПРАВИЛЬНЫЙ ПОРЯДОК
    X_in = X_in[need_cols]

    # 4) санитизация категориальных
    cat_cols_present = [c for c in cat_features if c in X_in.columns]
    if cat_cols_present:
        X_in[cat_cols_present] = X_in[cat_cols_present].astype("string").fillna("__NA__")

    # 5) индексы категориальных (CatBoost ожидает индексы/позиции)
    cat_idx = [X_in.columns.get_loc(c) for c in cat_cols_present]

    return X_in, cat_idx

def evaluate_model(model, X_test, y_test, symbol="model", ts=None, calib=None):
    """
    Оценивает модель на тесте (eval) и, если передан calib=(X_hold, y_hold),
    подбирает Temperature T на holdout (по NLL), применяет его, обучает изотонический
    калибратор уверенности на holdout и сохраняет и калибратор, и T.
    """
    from config import (
        CONFIDENCE_THRESHOLDS,
        TEMPERATURE_SCALING, TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_STEP,
    )

    tag = f"{symbol}" + (f"_{ts}" if ts else "")
    run_dir = MODEL_DIR / symbol / (ts if ts else "")
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- локальный хелпер для диагностики уверенности
    def _log_conf_stats(name: str, proba: np.ndarray):
        conf = proba.max(axis=1)
        logger.info("[%s] mean_conf=%.3f | p90=%.3f | p95=%.3f | max=%.3f",
                    name,
                    float(conf.mean()),
                    float(np.quantile(conf, 0.90)),
                    float(np.quantile(conf, 0.95)),
                    float(conf.max()))

    # =========================
    # 1) Temperature: подобрать по HOLDOUT (если есть)
    # =========================
    best_T = 1.0
    proba_hold = None
    y_hold_np = None

    if calib is not None:
        X_hold, y_hold = calib

        # 1.1 Выравнивание HOLDOUT под обучающий пайплайн (порядок фич, cat idx)
        X_hold_aligned, cat_idx_hold = _align_for_infer(X_hold)
        pool_hold = cb.Pool(X_hold_aligned, cat_features=cat_idx_hold)

        # 1.2 Вероятности на holdout до T
        proba_hold_raw = np.asarray(model.predict_proba(pool_hold))
        y_hold_np = np.asarray(y_hold).astype(int)

        _log_conf_stats("HOLD raw", proba_hold_raw)

        # 1.3 Подбор T по NLL (если включено и данных достаточно)
        if TEMPERATURE_SCALING and len(y_hold_np) >= 30:
            # ограничим максимум T (если в конфиге так задано)
            T_min = float(TEMPERATURE_MIN)
            T_max = float(TEMPERATURE_MAX)
            T_step = float(TEMPERATURE_STEP)

            best_T = _find_best_temperature(proba_hold_raw, y_hold_np, T_min, T_max, T_step)
            logger.info("Temperature scaling: candidate T=%.2f (chosen on holdout by NLL)", best_T)

        # 1.4 Применяем T и делаем «safeguard»: если T ухудшил macro-F1 на holdout — откатываемся
        y_pred_hold_raw = proba_hold_raw.argmax(axis=1).astype(int)
        f1m_raw = f1_score(y_hold_np, y_pred_hold_raw, average="macro")

        proba_hold_T = _apply_temperature_scaling(proba_hold_raw, best_T)
        y_pred_hold_T = proba_hold_T.argmax(axis=1).astype(int)
        f1m_T = f1_score(y_hold_np, y_pred_hold_T, average="macro")

        if f1m_T + 1e-9 < f1m_raw - 0.01:  # допускаем незначительное колебание
            logger.info("Temperature rollback: F1 holdout decreased (raw=%.4f -> T=%.4f). Using T=1.0",
                        f1m_raw, f1m_T)
            best_T = 1.0
            proba_hold = proba_hold_raw
        else:
            proba_hold = proba_hold_T

        _log_conf_stats("HOLD T", proba_hold)

        # 1.5 Сохраняем T (и алиасы)
        try:
            (run_dir / "temperature.json").write_text(json.dumps({"T": best_T}))
            (MODEL_DIR / f"temperature_{symbol}.json").write_text(json.dumps({"T": best_T}))
            (MODEL_DIR / "temperature.json").write_text(json.dumps({"T": best_T}))
        except Exception as e:
            logger.warning("Failed to save temperature.json: %s", e)

    # =========================
    # 2) Оценка на TEST (eval), уже с применённым T
    # =========================
    # 2.1 Выравнивание TEST под обучающий пайплайн
    X_test_aligned, cat_idx_test = _align_for_infer(X_test)
    pool_test = cb.Pool(X_test_aligned, cat_features=cat_idx_test)

    # 2.2 Вероятности на eval + temperature
    proba_eval = np.asarray(model.predict_proba(pool_test))
    proba_eval = _apply_temperature_scaling(proba_eval, best_T)
    _log_conf_stats("EVAL T", proba_eval)

    y_test_np   = np.asarray(y_test).astype(int)
    y_pred_eval = proba_eval.argmax(axis=1).astype(int)
    conf_eval   = proba_eval.max(axis=1)

    logger.info("=" * 30 + f" [FINAL METRICS] {tag} " + "=" * 30)
    acc_eval = accuracy_score(y_test_np, y_pred_eval)
    f1m_eval = f1_score(y_test_np, y_pred_eval, average='macro')
    logger.info("[All] Accuracy: %.4f", acc_eval)
    logger.info("[All] F1 macro: %.4f", f1m_eval)

    # per-class F1 (диагностика)
    try:
        from sklearn.metrics import precision_recall_fscore_support
        _, _, f1_per_cls, _ = precision_recall_fscore_support(
            y_test_np, y_pred_eval, labels=[0, 1, 2], zero_division=0
        )
        logger.info("[Eval per-class] F1: Down=%.3f | Up=%.3f | Neutral=%.3f",
                    f1_per_cls[0], f1_per_cls[1], f1_per_cls[2])
    except Exception:
        pass

    # 2.3 Отчёты и графики по eval
    labels_order = [0, 1, 2]
    target_names = ["Down", "Up", "Neutral"]
    report = classification_report(
        y_test_np, y_pred_eval, labels=labels_order, target_names=target_names, zero_division=0
    )
    (OUTPUT_DIR / f"{tag}_report.txt").write_text(report)

    ConfusionMatrixDisplay.from_predictions(y_test_np, y_pred_eval, cmap='viridis')
    plt.title(f"Confusion Matrix ({tag})")
    plt.savefig(OUTPUT_DIR / f"conf_matrix_{tag}.png")
    plt.close()

    # уверенные предсказания на eval (+ coverage в лог)
    for th in CONFIDENCE_THRESHOLDS:
        mask = conf_eval >= th
        coverage = float(mask.mean())
        if mask.any():
            acc = accuracy_score(y_test_np[mask], y_pred_eval[mask])
            f1m = f1_score(y_test_np[mask], y_pred_eval[mask], average='macro')
            logger.info(f"[Conf >= {th:.2f}] Coverage: {coverage:.3f} | Acc: {acc:.4f} | F1 macro: {f1m:.4f}")
        else:
            logger.warning(f"[Conf >= {th:.2f}] Coverage: 0.000 — нет уверенных прогнозов")

    # гистограммы уверенности на eval (по классам, уже после T)
    try:
        plt.figure(figsize=(8, 5))
        for cls, name in zip([0, 1, 2], ["Down", "Up", "Neutral"]):
            if np.any(y_test_np == cls):
                plt.hist(conf_eval[y_test_np == cls], bins=30, alpha=0.5, label=name)
        plt.legend()
        plt.title(f"Confidence distribution (eval) — {symbol}")
        out_hist = OUTPUT_DIR / f"conf_dist_{symbol}_{ts or 'run'}.png"
        plt.tight_layout()
        plt.savefig(out_hist)
        plt.close()
        logger.info("Saved confidence hist: %s", out_hist)
    except Exception as e:
        logger.warning("Failed to save confidence hist: %s", e)
    # =========================
    # 3) Поиск оптимального порога по precision
    # =========================
    try:
        th_auto, cov_auto = find_threshold_for_precision(y_test_np, proba_eval, target_precision=0.6)
        if th_auto is not None:
            logger.info("[Auto-threshold] Precision≥0.6 → th=%.2f | coverage=%.3f", th_auto, cov_auto)
        else:
            logger.warning("[Auto-threshold] Не найден порог для precision≥0.6")
    except Exception as e:
        logger.warning("Auto-threshold search failed: %s", e)

    # =========================
    # 3) Калибровка на HOLDOUT (если есть) — на temperature-scaled вероятностях
    # =========================
    if calib is not None and proba_hold is not None and y_hold_np is not None:
        y_pred_hold = proba_hold.argmax(axis=1).astype(int)
        conf_hold   = proba_hold.max(axis=1)
        is_correct  = (y_pred_hold == y_hold_np).astype(int)

        ir = fit_confidence_calibrator(conf_hold, is_correct)

        save_calibrator(ir, path=str(run_dir / "confidence_calibrator.pkl"))
        save_calibrator(ir, path=str(MODEL_DIR / f"confidence_calibrator_{symbol}.pkl"))
        save_calibrator(ir, path=str(MODEL_DIR / "confidence_calibrator.pkl"))
        logger.info(
            "Saved confidence calibrator → %s, and aliases: %s, %s",
            str(run_dir / "confidence_calibrator.pkl"),
            str(MODEL_DIR / f"confidence_calibrator_{symbol}.pkl"),
            str(MODEL_DIR / "confidence_calibrator.pkl"),
        )


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


def save_metadata(symbol: str, ts: str, best_params: dict, extras: dict = None):
    """
    Сохраняет ключевую информацию о тренировочном прогоне в models/metadata.json
    """
    meta_path = MODEL_DIR / "metadata.json"
    metadata = {}

    # Загружаем предыдущие метаданные, если файл существует
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            logger.warning("⚠️ metadata.json поврежден — создаю заново.")
            metadata = {}

    run_entry = {
        "symbol": symbol,
        "timestamp": ts,
        "datetime": datetime.utcnow().isoformat(),
        "best_params": best_params,
        "purged_cv": {
            "n_splits": 3,
            "embargo": EMBARGO_BARS,
            "min_train": MIN_CV_TRAIN,
            "min_val": MIN_CV_VAL
        },
        "feature_set_version": "v1",
        "calibration_set": "holdout_30pct",
        "model_type": "symbol_specific",
    }

    if extras:
        run_entry.update(extras)

    # Добавляем запись под символ
    metadata.setdefault(symbol, []).append(run_entry)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info("🧾 Metadata saved for %s → %s", symbol, meta_path)


def train_on_symbol(symbol: str, interval: str = "15", threshold: float = 0.0015, use_rolling_cv: bool = True):
    logger.info(f"⚙️ Обучение модели для {symbol}…")
    ts = _timestamp()

    # 1) данные
    X_train, X_test, y_train, y_test = prepare_data(symbol, interval, threshold)
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))
    # --- ГАРД: выравниваем длины X/y (на всякий случай) ---
    if len(X_train) != len(y_train):
        n_safe = min(len(X_train), len(y_train))
        logger.warning("[Pipeline] Train length mismatch: X=%d, y=%d → aligning to %d",
                       len(X_train), len(y_train), n_safe)
        X_train = X_train.iloc[:n_safe].reset_index(drop=True)
        y_train = y_train.iloc[:n_safe].reset_index(drop=True)

    if len(X_test) != len(y_test):
        n_safe = min(len(X_test), len(y_test))
        logger.warning("[Pipeline] Test length mismatch: X=%d, y=%d → aligning to %d",
                       len(X_test), len(y_test), n_safe)
        X_test = X_test.iloc[:n_safe].reset_index(drop=True)
        y_test = y_test.iloc[:n_safe].reset_index(drop=True)

    # 2) быстрая оптимизация гиперов
    best_params = optimize_catboost(X_train, y_train)
    logger.info("🔧 Best hyperparams: %s", {k: (int(v) if k == "depth" else float(v)) for k, v in best_params.items()})

    # 3) rolling CV (опционально, и если хватает данных)
    MIN_TRAIN_FOR_ROLLING = 400  # мягкий порог, чтобы не тратить время на крошечных выборках
    if use_rolling_cv and len(X_train) >= MIN_TRAIN_FOR_ROLLING:
        scores = rolling_cross_validation(X_train, y_train, best_params, n_splits=5)
        logger.info("📈 Rolling CV F1_macro: %.4f", float(np.mean(scores)))
    elif use_rolling_cv:
        logger.warning("🚧 Rolling CV пропущен (train=%d < %d)", len(X_train), MIN_TRAIN_FOR_ROLLING)

    # 4) финальное обучение (model_trainer сам сохранит артефакты в models/)
    train_final_model(X_train, y_train, best_params)

    # 5) аккуратно архивируем свежие артефакты под символ/таймстемп
    run_dir = _archive_artifacts(symbol, ts)

    # 6) загружаем свежую модель для оценки (fail-safe)
    model = cb.CatBoostClassifier()
    model_path = MODEL_DIR / "saved_model.cbm"
    try:
        model.load_model(str(model_path))
    except Exception as e:
        logger.exception("❌ Не удалось загрузить модель: %s", model_path)
        raise
    # 7) Сохраняем метаданные о прогоне
    save_metadata(symbol, ts, best_params)

    # === ЧЕСТНАЯ ОЦЕНКА И КАЛИБРОВКА ===
    # делим тест во времени: первые 70% → метрики, последние 30% → калибровка
    n_test = len(X_test)
    cut = max(1, int(n_test * 0.7))  # 70/30
    X_eval, y_eval = X_test.iloc[:cut], y_test.iloc[:cut]
    X_hold, y_hold = X_test.iloc[cut:], y_test.iloc[cut:]
    pct_eval = 100.0 * len(X_eval) / max(1, n_test)
    pct_hold = 100.0 * len(X_hold) / max(1, n_test)
    logger.info("[Eval/Holdout] sizes: eval=%d (%.1f%%), holdout=%d (%.1f%%), total=%d",
                len(X_eval), pct_eval, len(X_hold), pct_hold, n_test)

    # минимальный порог на размер holdout для изотоники (иначе пропускаем калибровку)
    MIN_HOLDOUT = 30
    if len(X_hold) >= MIN_HOLDOUT:
        calib_tuple = (X_hold, y_hold)
    else:
        logger.warning("Holdout слишком мал для калибровки (len=%d < %d) — пропускаю калибратор",
                       len(X_hold), MIN_HOLDOUT)
        calib_tuple = None

    # важно: symbol «как есть», ts передаём отдельным аргументом (без дублирования)
    evaluate_model(model, X_eval, y_eval, symbol=symbol, ts=ts, calib=calib_tuple)


if __name__ == "__main__":
    # Инициализация Bybit клиента (для загрузки OHLCV)
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)

    # Временно статический список — позже заменим на top_pairs из pair_discovery
    symbols = ["BTCUSDT", "ETHUSDT"]

    for sym in symbols:
        train_on_symbol(sym)
