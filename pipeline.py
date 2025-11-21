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
    ConfusionMatrixDisplay, precision_score
)

from pybit.unified_trading import HTTP
from config import BYBIT_API_KEY, BYBIT_API_SECRET, CONFIDENCE_THRESHOLDS
from data_loader import set_client

from model_trainer import (
    prepare_data,
    optimize_catboost,
    train_final_model,
    rolling_cross_validation, _atomic_write_text,
)
from confidence_calibrator import fit_confidence_calibrator, save_calibrator
import json
from datetime import datetime
from config import EMBARGO_BARS, MIN_CV_TRAIN, MIN_CV_VAL
from model_trainer import _sanitize_categoricals



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


def _apply_temperature_scaling(proba: np.ndarray, T: float) -> np.ndarray:
    """Softmax temperature scaling for multi-class, from probabilities (logit-free)."""
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    logits_T = logits / max(T, 1e-6)
    m = np.max(logits_T, axis=1, keepdims=True)
    exp = np.exp(logits_T - m)
    return exp / np.sum(exp, axis=1, keepdims=True)

def _nll_multiclass(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    NLL для мульти-класса, устойчивый к ситуации,
    когда модель вернула меньше классов, чем есть в глобальной разметке.
    Игнорируем объекты, для которых y_true >= proba.shape[1].
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)

    n_classes = proba.shape[1]
    valid_mask = (y_true >= 0) & (y_true < n_classes)

    if not np.any(valid_mask):
        # нет ни одного объекта с меткой, попадающей в диапазон предсказанных классов
        # возвращаем большой NLL, чтобы такой T точно не стал «лучшим»
        return 1e9

    yv = y_true[valid_mask]
    pv = proba[valid_mask]

    p = np.clip(pv[np.arange(len(yv)), yv], 1e-12, 1.0)
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
        logger.warning(
            "[AlignInfer] feature_columns.pkl не найден → используем входные колонки как есть (⚠ риск рассинхрона)"
        )

    # 2) гарантируем наличие всех нужных колонок
    exist_cols = set(X_in.columns)
    need_cols = list(feat_cols)

    # если список cat_features не найден — определим по dtype входного X (best-effort)
    if cat_features is None:
        cat_features = X_in.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        logger.warning(
            "[AlignInfer] cat_features.pkl не найден → fallback по dtype=%s (⚠ int-категориальные не будут определены)",
            cat_features[:10]
        )

    cat_set = set(cat_features)

    for col in need_cols:
        if col not in exist_cols:
            if col in cat_set:
                X_in[col] = "__NA__"
            else:
                X_in[col] = 0.0

    # 3) режем лишние колонки и ставим ПРАВИЛЬНЫЙ ПОРЯДОК
    X_in = X_in[need_cols]

    # 4) санитизация категориальных (string + fillna("__NA__"))
    cat_cols_present = [c for c in cat_features if c in X_in.columns]
    if cat_cols_present:
        X_in.loc[:, cat_cols_present] = X_in.loc[:, cat_cols_present].astype("string").fillna("__NA__")

    # 5) индексы категориальных (CatBoost ожидает индексы/позиции)
    cat_idx = [X_in.columns.get_loc(c) for c in cat_cols_present]

    logger.info(
        "[AlignInfer] aligned %d features (added=%d, cats=%d)",
        len(need_cols),
        len(set(need_cols) - exist_cols),
        len(cat_idx)
    )

    return X_in, cat_idx


logger = logging.getLogger(__name__)

def find_threshold_for_precision(
    y_true,
    proba,
    target_precision: float = 0.6,
    calibrator=None,
    T: float = 1.0,
    thresholds: np.ndarray | None = None,
):
    """
    Находит МИНИМАЛЬНЫЙ порог по КАЛИБРОВАННОЙ уверенности, при котором macro-precision >= target_precision.
    Последовательность соответствует прод-логике:
        proba -> Temperature (T) -> max(prob) -> Isotonic (если есть)

    Параметры:
      y_true        : true labels (array-like)
      proba         : вероятности (n_samples x n_classes)
      target_precision : целевой macro-precision
      calibrator    : sklearn.isotonic.IsotonicRegression (или None)
      T             : температура (float)
      thresholds    : np.ndarray порогов; если None -> linspace(0.90..0.30, шаг ≈ 0.05)

    Возвращает:
      (threshold, coverage)  — если найден порог; иначе (None, 0.0)
    """
    # --- валидация входа
    if proba is None or y_true is None:
        logger.warning("[find_threshold_for_precision] y_true/proba is None — returning (None, 0.0)")
        return None, 0.0

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)

    if y_true.size == 0 or proba.size == 0:
        logger.warning("[find_threshold_for_precision] Empty input — returning (None, 0.0)")
        return None, 0.0

    if y_true.shape[0] != proba.shape[0]:
        n = min(y_true.shape[0], proba.shape[0])
        logger.warning("[find_threshold_for_precision] Length mismatch (y=%d, p=%d) → truncated to %d",
                       y_true.shape[0], proba.shape[0], n)
        y_true = y_true[:n]
        proba = proba[:n]

    # --- Temperature scaling
    proba_T = _apply_temperature_scaling(proba, T)  # используй вашу реализацию из pipeline
    conf = proba_T.max(axis=1)
    y_pred = proba_T.argmax(axis=1)

    # --- Isotonic calibration (по уверенности)
    if calibrator is not None:
        try:
            # ваша helper-функция калибровки: должна вернуть np.ndarray в [0, 1]
            from model_trainer import apply_isotonic_confidence  # если уже импортирована выше — можно убрать
            conf = apply_isotonic_confidence(calibrator, conf)
        except Exception as e:
            logger.warning("[find_threshold_for_precision] Isotonic failed (%s) — using raw confidence", e)

    # --- сетка порогов
    if thresholds is None:
        thresholds = np.linspace(0.90, 0.30, 13)

    best_prec = -1.0
    best_th = None
    best_cov = 0.0

    for th in thresholds:
        m = conf >= th
        if not m.any():
            continue
        prec = precision_score(y_true[m], y_pred[m], average="macro", zero_division=0)
        cov = float(m.mean())

        if prec >= target_precision:
            logger.info("[Auto-threshold] target_precision=%.2f → th=%.2f | precision=%.3f | coverage=%.3f",
                        target_precision, th, prec, cov)
            return float(th), cov

        if prec > best_prec:
            best_prec, best_th, best_cov = float(prec), float(th), cov

    logger.warning("[Auto-threshold] No threshold reached target_precision=%.2f (best=%.3f @ th=%.2f, cov=%.3f)",
                   target_precision, max(0.0, best_prec), best_th if best_th is not None else -1.0, best_cov)
    return None, 0.0

def _archive_artifacts(symbol: str, ts: str):
    """
    Копируем свежие артефакты тренера в персональную папку:
    models/{symbol}/{ts}/(model.cbm, scaler.pkl, cat_features.pkl, feature_columns.pkl)
    + плоские файлы model_{symbol}.cbm, scaler_{symbol}.pkl, cat_features_{symbol}.pkl, feature_columns_{symbol}.pkl
    """
    # источники — где их сохраняет model_trainer / prepare_data
    model_src = MODEL_DIR / "saved_model.cbm"
    scaler_src = MODEL_DIR / "scaler.pkl"              # см. сохранение в prepare_data()
    cat_src    = MODEL_DIR / "cat_features.pkl"
    feat_src   = MODEL_DIR / "feature_columns.pkl"     # критично для выравнивания фичей на инференсе

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

    # --- в timestamp-папку (run_dir) ---
    _safe_copy(model_src, target_dir / "model.cbm",            "Модель")
    _safe_copy(scaler_src, target_dir / "scaler.pkl",          "Скейлер")
    _safe_copy(cat_src,    target_dir / "cat_features.pkl",    "cat_features")
    _safe_copy(feat_src,   target_dir / "feature_columns.pkl", "feature_columns")

    # --- плоские алиасы по символу ---
    _safe_copy(model_src, MODEL_DIR / f"model_{symbol}.cbm",             "Модель (алиас)")
    _safe_copy(scaler_src, MODEL_DIR / f"scaler_{symbol}.pkl",           "Скейлер (алиас)")
    _safe_copy(cat_src,    MODEL_DIR / f"cat_features_{symbol}.pkl",     "cat_features (алиас)")
    _safe_copy(feat_src,   MODEL_DIR / f"feature_columns_{symbol}.pkl",  "feature_columns (алиас)")

    return target_dir


def evaluate_model(model, X_test, y_test, symbol="model", ts=None, calib=None):
    """
    Оценивает модель на TEST (eval) и, если передан calib=(X_hold, y_hold),
    подбирает Temperature T по NLL на holdout (с safeguard по F1), применяет его,
    обучает IsotonicRegression на conf(holdout) и сохраняет и T, и калибратор.

    Все метрики/пороги считаются консистентно с прод-логикой:
      proba -> Temperature -> max(proba) -> Isotonic.predict(conf).
    """
    from config import (
        CONFIDENCE_THRESHOLDS,
        TEMPERATURE_SCALING, TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_STEP,
    )
    from confidence_calibrator import fit_confidence_calibrator, save_calibrator

    # --- гарантируем каталоги
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    symbol = symbol or "model"
    tag = f"{symbol}" + (f"_{ts}" if ts else "")
    run_dir = MODEL_DIR / symbol / (ts if ts else "")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- helpers
    def _log_conf_stats(name: str, proba: np.ndarray):
        conf = proba.max(axis=1)
        logger.info("[%s] mean_conf=%.3f | p90=%.3f | p95=%.3f | max=%.3f",
                    name,
                    float(conf.mean()),
                    float(np.quantile(conf, 0.90)),
                    float(np.quantile(conf, 0.95)),
                    float(conf.max()))

    def _apply_isotonic(conf: np.ndarray, calibrator):
        if calibrator is None:
            return conf
        try:
            out = calibrator.predict(conf.reshape(-1))
            out = np.asarray(out, dtype=float)
            return np.clip(out, 0.0, 1.0)
        except Exception as e:
            logger.warning("Isotonic application failed: %s — using raw confidence", e)
            return conf

    def _atomic_write(path_obj, text_payload: str):
        # Используем твой _atomic_write_text, если он есть; иначе — обычная запись
        try:
            _atomic_write_text(path_obj, text_payload)
        except Exception:
            try:
                path_obj.write_text(text_payload)
            except Exception as e:
                logger.warning("Failed to write %s: %s", str(path_obj), e)

    # =========================
    # 1) HOLDOUT: подбор температуры и обучение калибратора
    # =========================
    best_T = 1.0
    proba_hold_T = None
    y_hold_np = None
    calibrator = None

    if calib is not None:
        X_hold, y_hold = calib

        # 1.1 Выравниваем HOLDOUT под обучающий пайплайн (строгий порядок/набор фич, cat idx)
        X_hold_aligned, cat_idx_hold = _align_for_infer(X_hold, symbol=symbol)
        pool_hold = cb.Pool(X_hold_aligned, cat_features=cat_idx_hold if cat_idx_hold else None)

        # 1.2 Сырые вероятности на holdout
        proba_hold_raw = np.asarray(model.predict_proba(pool_hold))
        y_hold_np = np.asarray(y_hold).astype(int)

        _log_conf_stats("HOLD raw", proba_hold_raw)

        # 1.3 Подбор T по NLL (если включено и данных достаточно)
        if TEMPERATURE_SCALING and len(y_hold_np) >= 30:
            T_min = float(TEMPERATURE_MIN)
            T_max = float(TEMPERATURE_MAX)
            T_step = float(TEMPERATURE_STEP)
            best_T = _find_best_temperature(proba_hold_raw, y_hold_np, T_min, T_max, T_step)
            logger.info("Temperature scaling: candidate T=%.2f (chosen on holdout by NLL)", best_T)

        # 1.4 Safeguard по macro-F1
        y_pred_raw = proba_hold_raw.argmax(axis=1).astype(int)
        f1_raw = f1_score(y_hold_np, y_pred_raw, average="macro")

        proba_hold_T_cand = _apply_temperature_scaling(proba_hold_raw, best_T)
        y_pred_T = proba_hold_T_cand.argmax(axis=1).astype(int)
        f1_T = f1_score(y_hold_np, y_pred_T, average="macro")

        if f1_T + 1e-9 < f1_raw - 0.01:
            logger.info(
                "Temperature rollback: F1 holdout decreased (raw=%.4f -> T=%.4f). Using T=1.0",
                f1_raw, f1_T
            )
            best_T = 1.0
            proba_hold_T = proba_hold_raw
        else:
            proba_hold_T = proba_hold_T_cand

        _log_conf_stats("HOLD T", proba_hold_T)

        # 1.5 Сохранить T (и алиасы) атомарно
        payload = json.dumps({"T": float(best_T)})
        _atomic_write(run_dir / "temperature.json", payload)
        _atomic_write(MODEL_DIR / f"temperature_{symbol}.json", payload)
        _atomic_write(MODEL_DIR / "temperature.json", payload)

        # 1.6 Обучить калибратор уверенности на holdout (после Temperature)
        conf_hold = proba_hold_T.max(axis=1)
        is_correct = (proba_hold_T.argmax(axis=1).astype(int) == y_hold_np).astype(int)
        ir = fit_confidence_calibrator(conf_hold, is_correct)
        save_calibrator(ir, path=str(run_dir / "confidence_calibrator.pkl"))
        save_calibrator(ir, path=str(MODEL_DIR / f"confidence_calibrator_{symbol}.pkl"))
        save_calibrator(ir, path=str(MODEL_DIR / "confidence_calibrator.pkl"))
        calibrator = ir
    else:
        logger.info("[Eval] No holdout calibration provided — using raw T=1.0 and no isotonic.")

    # =========================
    # 2) TEST/EVAL: метрики и отчёты (T и изотоник как в проде)
    # =========================
    # 2.1 Выравниваем TEST под обучающий пайплайн
    X_test_aligned, cat_idx_test = _align_for_infer(X_test, symbol=symbol)
    pool_test = cb.Pool(X_test_aligned, cat_features=cat_idx_test if cat_idx_test else None)

    # 2.2 Вероятности + Temperature
    proba_eval_raw = np.asarray(model.predict_proba(pool_test))
    proba_eval_T = _apply_temperature_scaling(proba_eval_raw, best_T)
    _log_conf_stats("EVAL T", proba_eval_T)

    # 2.3 Предсказания и (калиброванная) уверенность
    y_test_np = np.asarray(y_test).astype(int)
    y_pred_eval = proba_eval_T.argmax(axis=1).astype(int)
    conf_eval_raw = proba_eval_T.max(axis=1)
    conf_eval_cal = _apply_isotonic(conf_eval_raw, calibrator)

    logger.info("=" * 30 + f" [FINAL METRICS] {tag} " + "=" * 30)
    acc_eval = accuracy_score(y_test_np, y_pred_eval)
    f1m_eval = f1_score(y_test_np, y_pred_eval, average="macro")
    logger.info("[All] Accuracy: %.4f", acc_eval)
    logger.info("[All] F1 macro: %.4f", f1m_eval)

    # ------- динамические классы (2 или 3) -------
    uniq = np.unique(y_test_np)
    # сортируем, чтобы 0,1,(2) шли в правильном порядке
    labels_order = sorted(int(c) for c in uniq)
    name_map = {0: "Down", 1: "Up", 2: "Neutral"}
    target_names = [name_map.get(c, str(c)) for c in labels_order]

    # per-class F1 (диагностика)
    try:
        if len(labels_order) >= 2:
            from sklearn.metrics import precision_recall_fscore_support
            _, _, f1_per_cls, _ = precision_recall_fscore_support(
                y_test_np,
                y_pred_eval,
                labels=labels_order,
                zero_division=0,
            )
            log_parts = []
            for c, f1_c in zip(labels_order, f1_per_cls):
                log_parts.append(f"{name_map.get(c, c)}={f1_c:.3f}")
            logger.info("[Eval per-class] F1: " + " | ".join(log_parts))
        else:
            logger.warning("[Eval] Less than 2 unique classes in y_test — skipping per-class metrics")
    except Exception as e:
        logger.warning("Per-class metrics failed: %s", e)

    # 2.4 Отчёт и матрица ошибок
    try:
        report = classification_report(
            y_test_np,
            y_pred_eval,
            labels=labels_order,
            target_names=target_names,
            zero_division=0,
        )
        (OUTPUT_DIR / f"{tag}_report.txt").write_text(report)
    except Exception as e:
        logger.warning("Failed to write classification report: %s", e)

    try:
        ConfusionMatrixDisplay.from_predictions(
            y_test_np, y_pred_eval, labels=labels_order, cmap="viridis"
        )
        plt.title(f"Confusion Matrix ({tag})")
        plt.savefig(OUTPUT_DIR / f"conf_matrix_{tag}.png")
        plt.close()
    except Exception as e:
        logger.warning("Failed to save confusion matrix: %s", e)

    # 2.5 Метрики по порогам — на КАЛИБРОВАННОЙ уверенности
    for th in CONFIDENCE_THRESHOLDS:
        mask = conf_eval_cal >= th
        coverage = float(mask.mean())
        if mask.any():
            acc_c = accuracy_score(y_test_np[mask], y_pred_eval[mask])
            f1m_c = f1_score(y_test_np[mask], y_pred_eval[mask], average="macro")
            logger.info(
                f"[Conf(cal) >= {th:.2f}] Coverage: {coverage:.3f} | "
                f"Acc: {acc_c:.4f} | F1 macro: {f1m_c:.4f}"
            )
        else:
            logger.warning(f"[Conf(cal) >= {th:.2f}] Coverage: 0.000 — нет уверенных прогнозов")

    # 2.6 Гистограммы уверенности (калиброванной)
    try:
        plt.figure(figsize=(8, 5))
        for cls in labels_order:
            name = name_map.get(cls, str(cls))
            if np.any(y_test_np == cls):
                plt.hist(
                    conf_eval_cal[y_test_np == cls],
                    bins=30,
                    alpha=0.5,
                    label=name,
                )
        plt.legend()
        plt.title(f"Confidence distribution (eval, calibrated) — {symbol}")
        out_hist = OUTPUT_DIR / f"conf_dist_{symbol}_{ts or 'run'}.png"
        plt.tight_layout()
        plt.savefig(out_hist)
        plt.close()
        logger.info("Saved confidence hist: %s", out_hist)
    except Exception as e:
        logger.warning("Failed to save confidence hist: %s", e)

    # =========================
    # 3) Поиск авто-порога по precision — на КАЛИБРОВАННОЙ уверенности
    # =========================
    try:
        # пробуем «новую» сигнатуру (с calibrator/T); если нет — fallback на старую
        try:
            th_auto, cov_auto = find_threshold_for_precision(
                y_true=y_test_np,
                proba=proba_eval_T,           # уже после Temperature
                target_precision=0.60,
                calibrator=calibrator,        # чтобы рассчитывался calibrated confidence
                T=best_T,
            )
        except TypeError:
            # старая версия без calibrator/T
            th_auto, cov_auto = find_threshold_for_precision(
                y_test_np, proba_eval_T, target_precision=0.60
            )

        if th_auto is not None:
            logger.info(
                "[Auto-threshold] Precision≥0.6 → th=%.2f | coverage=%.3f",
                th_auto, cov_auto,
            )
        else:
            logger.warning("[Auto-threshold] Не найден порог для precision≥0.6")
    except Exception as e:
        logger.warning("Auto-threshold search failed: %s", e)



def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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
