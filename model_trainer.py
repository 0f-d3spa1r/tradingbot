import os
import pickle
import logging
from typing import Tuple, List
import  pathlib, tempfile, shutil
import numpy as np
import pandas as pd

from bayes_opt import BayesianOptimization
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.cluster import KMeans
import catboost as cb
import matplotlib.pyplot as plt
import json

# ================= Project-specific imports =================
from config import (
    USE_RESAMPLING,
    FEATURE_BAGGING_FRAC,
    TEMPERATURE_SCALING,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_STEP,
    EMBARGO_BARS,
    RESAMPLING_STRATEGY,
    USE_CLASS_WEIGHTS,
    CLASS_WEIGHT_MODE,
    CONFIDENCE_THRESHOLDS,
)
from data_loader import get_processed_ohlcv
from feature_engineering import generate_target, select_features, generate_clustering

from pathlib import Path
# ================= Optional: imblearn fallbacks =================
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    SMOTE = None
    RandomUnderSampler = None

logger = logging.getLogger(__name__)


UP_THRESHOLD = 0.002
DOWN_THRESHOLD = 0.0015

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

# ---------- atomic write helpers ----------
def _atomic_write_text(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    os.replace(tmp_path, str(path))

def _atomic_copy(src: pathlib.Path, dst: pathlib.Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dst.parent)) as tmp:
        shutil.copyfile(src, tmp.name)
        tmp_path = tmp.name
    os.replace(tmp_path, str(dst))


def _sanitize_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Приводит категориальные к строкам и заполняет NaN, чтобы CatBoost не падал.
    Возвращает копию df и список cat-колонок.
    """
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("__NA__")
    return out, cat_cols

# ---------- isotonic application ----------
def apply_isotonic_confidence(ir, conf: np.ndarray) -> np.ndarray:
    # sklearn IsotonicRegression: используем predict
    conf_cal = ir.predict(conf.astype(np.float64))
    conf_cal = np.clip(conf_cal, 0.0, 1.0).astype(np.float64)
    return conf_cal

# ---------- class weights with smoothing ----------
def _compute_class_weights(y: np.ndarray, alpha: float = 2.0, cap: float = 8.0) -> dict[int, float]:
    # w_c = 1 / (count + alpha), нормируем и ограничиваем cap
    classes, counts = np.unique(y, return_counts=True)
    inv = {int(c): 1.0 / (cnt + alpha) for c, cnt in zip(classes, counts)}
    mean_inv = np.mean(list(inv.values()))
    w = {c: min(inv[c] / mean_inv, cap) for c in inv}
    return w


def prepare_data(symbol: str, interval: str, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Загружает обработанные OHLCV, генерирует фичи/кластеризацию,
    делит на train/test без перемешивания, опционально ресемплит ТОЛЬКО train,
    масштабирует числовые фичи, сохраняет scaler в models/scaler.pkl.

    Возвращает: X_train_scaled, X_test_scaled, y_train, y_test
    """
    # 1) данные -> кластеры -> фичи/таргет
    df = get_processed_ohlcv(symbol, interval)
    df = generate_clustering(df)

    X, y = select_features(df)
    logger.info("Target distribution (raw): %s", y.value_counts(normalize=True).to_dict())

    # =========================================================
    # 🔥 ФИЛЬТР 0/1 — убираем класс 2 (NEUTRAL)
    # =========================================================
    mask_01 = y.isin([0, 1])
    removed = len(y) - mask_01.sum()
    if removed > 0:
        logger.warning(f"[prepare_data] Dropped {removed} NEUTRAL samples (class=2). Using 0/1 only.")

    X = X.loc[mask_01].reset_index(drop=True)
    y = y.loc[mask_01].reset_index(drop=True).astype(int)

    logger.info("Target distribution (filtered 0/1): %s",
                y.value_counts(normalize=True).round(3).to_dict())
    # =========================================================

    # 2) сплит по времени (без shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    logger.info("Train target dist: %s", y_train.value_counts(normalize=True).round(3).to_dict())
    logger.info("Test  target dist: %s", y_test.value_counts(normalize=True).round(3).to_dict())

    # --- ГАРД: выравниваем длины ---
    if len(X_train) != len(y_train):
        n_safe = min(len(X_train), len(y_train))
        logger.warning("[prepare_data] Train mismatch: X=%d, y=%d → aligning to %d",
                       len(X_train), len(y_train), n_safe)
        X_train = X_train.iloc[:n_safe].copy().reset_index(drop=True)
        y_train = y_train.iloc[:n_safe].copy().reset_index(drop=True)

    if len(X_test) != len(y_test):
        n_safe = min(len(X_test), len(y_test))
        logger.warning("[prepare_data] Test mismatch: X=%d, y=%d → aligning to %d",
                       len(X_test), len(y_test), n_safe)
        X_test = X_test.iloc[:n_safe].copy().reset_index(drop=True)
        y_test = y_test.iloc[:n_safe].copy().reset_index(drop=True)

    # 3) разбиение по типам
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("Нет числовых признаков для скейлинга — проверь select_features().")

    # --- Санитизация NaN ---
    mask_num_train = X_train[num_cols].notna().all(axis=1)
    mask_num_test  = X_test[num_cols].notna().all(axis=1)
    mask_y_train = y_train.notna()
    mask_y_test  = y_test.notna()

    if cat_cols:
        X_train.loc[:, cat_cols] = X_train[cat_cols].astype("string").fillna("__NA__")
        X_test.loc[:, cat_cols]  = X_test[cat_cols].astype("string").fillna("__NA__")
        mask_cat_train = pd.Series(True, index=X_train.index)
        mask_cat_test  = pd.Series(True, index=X_test.index)
    else:
        mask_cat_train = pd.Series(True, index=X_train.index)
        mask_cat_test  = pd.Series(True, index=X_test.index)

    keep_train = (mask_num_train & mask_cat_train & mask_y_train)
    keep_test  = (mask_num_test  & mask_cat_test  & mask_y_test)

    if keep_train.sum() < len(keep_train):
        logger.warning("[prepare_data] Dropping %d rows with NaN (train)", int((~keep_train).sum()))
    if keep_test.sum() < len(keep_test):
        logger.warning("[prepare_data] Dropping %d rows with NaN (test)", int((~keep_test).sum()))

    X_train = X_train.loc[keep_train].reset_index(drop=True)
    y_train = y_train.loc[keep_train].reset_index(drop=True).astype(int)
    X_test = X_test.loc[keep_test].reset_index(drop=True)
    y_test = y_test.loc[keep_test].reset_index(drop=True).astype(int)

    # 4) ресемплинг — если включено
    if USE_RESAMPLING and RESAMPLING_STRATEGY != "none":
        if RESAMPLING_STRATEGY == "smote":
            if SMOTE is None:
                logger.warning("[Resampling] SMOTE unavailable")
            else:
                logger.info("[Resampling] Applying SMOTE")
                X_train_num = X_train[num_cols].reset_index(drop=True)
                X_train_cat = X_train[cat_cols].reset_index(drop=True) if cat_cols else pd.DataFrame(index=X_train_num.index)

                smote = SMOTE(random_state=42)
                X_num_res, y_train_res = smote.fit_resample(X_train_num, y_train.reset_index(drop=True))

                if not X_train_cat.empty:
                    rep = max(1, int(np.ceil(len(y_train_res) / max(1, len(X_train_cat)))))
                    X_cat_rep = pd.concat([X_train_cat] * rep, ignore_index=True).iloc[:len(y_train_res)]
                else:
                    X_cat_rep = pd.DataFrame(index=np.arange(len(y_train_res)))

                X_train = pd.concat(
                    [pd.DataFrame(X_num_res, columns=num_cols), X_cat_rep.reset_index(drop=True)],
                    axis=1
                )
                y_train = y_train_res.reset_index(drop=True).astype(int)

                if cat_cols:
                    for c in cat_cols:
                        if c in X_train.columns:
                            X_train[c] = X_train[c].astype("string")

        elif RESAMPLING_STRATEGY == "undersample":
            if RandomUnderSampler is None:
                logger.warning("[Resampling] RUS unavailable")
            else:
                logger.info("[Resampling] Applying undersampling")
                X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(
                    X_train.reset_index(drop=True), y_train.reset_index(drop=True)
                )
                y_train = y_train.astype(int)
    else:
        logger.info("[Resampling] disabled — using class weights.")

    # 5) масштабирование
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]),
        columns=num_cols, index=X_train.index
    )
    X_test_num_scaled = pd.DataFrame(
        scaler.transform(X_test[num_cols]),
        columns=num_cols, index=X_test.index
    )

    # 6) собираем обратно
    if cat_cols:
        X_train_cat = X_train[cat_cols].astype("string").fillna("__NA__").reset_index(drop=True)
        X_test_cat  = X_test[cat_cols].astype("string").fillna("__NA__").reset_index(drop=True)

        X_train_scaled = pd.concat([X_train_num_scaled.reset_index(drop=True), X_train_cat], axis=1)
        X_test_scaled  = pd.concat([X_test_num_scaled.reset_index(drop=True),  X_test_cat],  axis=1)
    else:
        X_train_scaled = X_train_num_scaled.reset_index(drop=True)
        X_test_scaled  = X_test_num_scaled.reset_index(drop=True)

    # 7) сохраняем scaler
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(project_root)
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return (
        X_train_scaled.reset_index(drop=True),
        X_test_scaled.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )





def optimize_catboost(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
            Байес-опт по CatBoost с ЧЕСТНОЙ временной валидацией (Purged CV + Embargo).
            Пропускаем фолды, где train/val слишком малы — чтобы не учиться на шуме.
            Возвращаем лучший набор гиперов по среднему F1_macro на вал. фолдах.
            """
    from cv_utils import purged_cv_splits
    from config import EMBARGO_BARS, MIN_CV_TRAIN, MIN_CV_VAL, N_SPLITS_BO
    from sklearn.utils.class_weight import compute_sample_weight
    # ---- Prep before BayesOpt: выравниваем длины и чиним категориальные ----

    # санитизируем один раз весь train:
    X_train_sanitized, cat_cols = _sanitize_categoricals(X_train)

    # выравниваем длины X и y (если вдруг разошлись)
    if len(X_train_sanitized) != len(y_train):
        nX, nY = len(X_train_sanitized), len(y_train)
        logger.warning("[BayesOpt] Length mismatch before CV: X_train=%d, y_train=%d. Aligning to min %d.",
                       nX, nY, min(nX, nY))
        n_safe = min(nX, nY)
        X_train_sanitized = X_train_sanitized.iloc[:n_safe].reset_index(drop=True)
        y_train = y_train.iloc[:n_safe].reset_index(drop=True)

    n = len(X_train_sanitized)  # размер, с которым работаем дальше

    def evaluate(depth, learning_rate, l2_leaf_reg, bagging_temperature,
                 random_strength, rsm):

        # считаем число классов по тому y, с которым оптимизируем
        n_classes = int(pd.Series(y_train).nunique())  # <-- если у тебя y называется иначе, подставь его

        if n_classes == 2:
            loss_fn = "Logloss"
            eval_metric = "F1"
        else:
            loss_fn = "MultiClass"
            eval_metric = "TotalF1"

        params = {
            "iterations": 600,  # больше итераций, будет early stop
            "depth": int(depth),
            "learning_rate": float(learning_rate),
            "l2_leaf_reg": float(l2_leaf_reg),
            "bagging_temperature": float(bagging_temperature),
            "random_strength": float(random_strength),
            "rsm": float(rsm),  # feature subsampling
            "bootstrap_type": "Bayesian",
            "loss_function": loss_fn,
            "eval_metric": eval_metric,
            "verbose": False,
            "random_seed": 42,
        }


        scores = []
        used_folds = 0

        for fold_id, (tr_idx, val_idx) in enumerate(
                purged_cv_splits(n=n, n_splits=N_SPLITS_BO, embargo=EMBARGO_BARS), start=1
        ):
            tr_len, val_len = len(tr_idx), len(val_idx)
            if tr_len < MIN_CV_TRAIN or val_len < MIN_CV_VAL:
                logger.info(
                    "[BayesOpt][fold %d] skipped: train=%d (min %d), val=%d (min %d), embargo=%d",
                    fold_id, tr_len, MIN_CV_TRAIN, val_len, MIN_CV_VAL, EMBARGO_BARS
                )
                continue

            X_tr, X_val = X_train_sanitized.iloc[tr_idx], X_train_sanitized.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            sw = compute_sample_weight(class_weight="balanced", y=y_tr)

            model = cb.CatBoostClassifier(**params)
            try:
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_val, y_val),
                    sample_weight=sw,
                    cat_features=(cat_cols or None),
                    early_stopping_rounds=40,
                    verbose=False
                )
                preds = np.asarray(model.predict(X_val)).ravel().astype(int)
                score = f1_score(y_val, preds, average="macro", zero_division=0)
            except Exception as e:
                logger.warning("[BayesOpt][fold %d] failed: %s", fold_id, e)
                score = 0.0

            scores.append(float(score))
            used_folds += 1

        if not scores:
            logger.warning("[BayesOpt] No valid folds — returning 0.0 for this point")
            return 0.0

        mean_score = float(np.mean(scores))
        logger.debug("[BayesOpt] used_folds=%d | mean F1_macro=%.4f", used_folds, mean_score)
        return mean_score

    optimizer = BayesianOptimization(
        f=evaluate,
        pbounds={
            "depth": (4, 6),  # чуть глубже
            "learning_rate": (0.03, 0.12),  # средняя зона
            "l2_leaf_reg": (3.0, 20.0),  # помягче
            "bagging_temperature": (0.0, 1.0),
            "random_strength": (0.5, 20.0),  # снизили верх
            "rsm": (0.5, 0.95),  # не так низко
        },
        random_state=42,
        verbose=0,
    )

    optimizer.maximize(init_points=3, n_iter=7)

    best = optimizer.max["params"]
    best = {
        "depth": int(round(best["depth"])),
        "learning_rate": float(best["learning_rate"]),
        "l2_leaf_reg": float(best["l2_leaf_reg"]),
        "bagging_temperature": float(best["bagging_temperature"]),
        "random_strength": float(best["random_strength"]),
        "rsm": float(best["rsm"]),
    }
    logger.info("🔧 Best hyperparams (purged CV): %s", best)
    return best


def rolling_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_params: dict,
    n_splits: int = 5
):
    # --- ГАРД: выравнивание длин X / y ---
    nX, nY = len(X), len(y)
    if nX != nY:
        logger.warning("[RollingCV] Length mismatch: X=%d, y=%d. Aligning to min length.", nX, nY)
        n_safe = min(nX, nY)
        X = X.iloc[:n_safe].reset_index(drop=True)
        y = y.iloc[:n_safe].reset_index(drop=True)

    # --- Санитизация категориальных на весь X: str + fillna("__NA__") ---

    X, cat_cols = _sanitize_categoricals(X)

    """
    Walk-forward sanity CV с эмбарго и гардом минимальных размеров.
    - Без ресемплинга; баланс — через class_weight (если включено).
    - Эмбарго «вырезает» зазор между train и test (по времени).
    """
    from config import EMBARGO_BARS_ROLLING, MIN_ROLL_TRAIN, MIN_ROLL_TEST

    logger.info("Rolling CV started")
    scores: list[float] = []

    total_len = len(X)
    if total_len < (MIN_ROLL_TRAIN + MIN_ROLL_TEST + EMBARGO_BARS_ROLLING):
        logger.warning(
            "Dataset too small for rolling CV: len=%d (need >= %d)",
            total_len, MIN_ROLL_TRAIN + MIN_ROLL_TEST + EMBARGO_BARS_ROLLING
        )
        return scores

    # базовое окно: ~60% train, ~20% test, ~10% шаг — как раньше
    window_size = int(total_len * 0.6)
    test_size   = int(total_len * 0.2)
    step        = max(1, int(total_len * 0.1))

    # параметры модели
    params = model_params.copy()
    params["depth"] = int(params.get("depth", 6))
    params.update({
        "random_seed": 42,
        "iterations": 300,
        "verbose": False,
    })

    for i in range(n_splits):
        train_start = i * step
        train_end   = train_start + window_size
        gap_start   = train_end
        gap_end     = gap_start + EMBARGO_BARS_ROLLING
        test_start  = gap_end
        test_end    = test_start + test_size

        if test_end > total_len:
            break

        # гарды на размеры
        tr_len = train_end - train_start
        te_len = test_end - test_start
        if tr_len < MIN_ROLL_TRAIN or te_len < MIN_ROLL_TEST:
            logger.info(
                "Fold %d skipped: train=%d (min %d), test=%d (min %d), embargo=%d",
                i + 1, tr_len, MIN_ROLL_TRAIN, te_len, MIN_ROLL_TEST, EMBARGO_BARS_ROLLING
            )
            continue

        X_train_raw = X.iloc[train_start:train_end]
        y_train_ = y.iloc[train_start:train_end]
        X_test_raw = X.iloc[test_start:test_end]
        y_test_ = y.iloc[test_start:test_end]

        # санитизируем категориальные прямо на срезах
        X_train_s, cat_cols = _sanitize_categoricals(X_train_raw)
        X_test_s, _ = _sanitize_categoricals(X_test_raw)

        # веса классов (если включены)
        sample_weight = None
        try:
            from config import USE_CLASS_WEIGHTS, CLASS_WEIGHT_MODE
            if USE_CLASS_WEIGHTS and CLASS_WEIGHT_MODE == "balanced":
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_)
        except Exception:
            pass

        # Определяем тип задачи по числу классов в текущем y_train_
        n_classes_fold = int(pd.Series(y_train_).nunique())
        if n_classes_fold < 2:
            logger.warning("CV fold: only %d class in y_train_ — skip fold", n_classes_fold)
            # если это внутри цикла по фолдам — тут просто continue
            # если не в цикле, можно поднять исключение
            raise RuntimeError("Not enough classes in fold")

        if n_classes_fold == 2:
            loss_fn = "Logloss"
            eval_metric = "F1"
        else:
            loss_fn = "MultiClass"
            eval_metric = "TotalF1"

        local_params = params.copy()
        local_params["loss_function"] = loss_fn
        local_params["eval_metric"] = eval_metric

        model = cb.CatBoostClassifier(**local_params)
        model.fit(
            X_train_s,
            y_train_,
            cat_features=(cat_cols or None),
            sample_weight=sample_weight,
            verbose=False,
        )

        y_hat = np.asarray(model.predict(X_test_s)).ravel().astype(int)
        score = f1_score(y_test_, y_hat, average="macro", zero_division=0)
        logger.info(
            "Fold %d: train=%d, gap=%d, test=%d | F1_macro=%.4f",
            i + 1, tr_len, EMBARGO_BARS_ROLLING, te_len, score
        )
        scores.append(score)

    if scores:
        logger.info("Rolling CV complete. Mean F1_macro: %.4f", float(np.mean(scores)))
    else:
        logger.warning("Rolling CV produced no valid folds — check sizes/embargo")

    return scores


def train_final_model(
   X_train: pd.DataFrame,
   y_train: pd.Series,
   best_params: dict,
) -> "cb.CatBoostClassifier":
   """
   Финальный фит CatBoost с прод-гардами:
     - строгая выравниловка длин X/y и dtype целевого
     - санитизация категориальных (str + fillna('__NA__'))
     - (опц.) лёгкий feature bagging (по config.FEATURE_BAGGING_FRAC)
     - сохранение feature_columns.pkl и cat_features.pkl (алиасы в models/)
     - class reweight (по config.USE_CLASS_WEIGHTS / CLASS_WEIGHT_MODE) + безопасная нарезка на train-чанк
     - early stopping на "хвосте" трейна (последние 10%), guard при слишком малом вал-чане
     - итоговые диагностические метрики + отчёты (train)
     - сохранение модели в models/saved_model.cbm (алиас для пайплайна/архиватора)
   """
   import os, pickle
   import numpy as np
   from sklearn.metrics import (
       accuracy_score, f1_score, precision_score, recall_score, classification_report,
       ConfusionMatrixDisplay,
   )
   from sklearn.utils.class_weight import compute_sample_weight

   os.makedirs("models", exist_ok=True)
   os.makedirs("outputs", exist_ok=True)

   # --- 0) ГАРД: выравниваем длины X/y + тип y
   if len(X_train) != len(y_train):
       n_safe = min(len(X_train), len(y_train))
       logger.warning("[FinalFit] Length mismatch: X=%d, y=%d → aligning to %d",
                      len(X_train), len(y_train), n_safe)
       X_train = X_train.iloc[:n_safe].reset_index(drop=True)
       y_train = y_train.iloc[:n_safe].reset_index(drop=True)
   y_train = y_train.astype(int).reset_index(drop=True)

   # --- 1) Санитизация категориальных + список cat_features (по именам)
   X_train_s, cat_features = _sanitize_categoricals(X_train)
   cat_features_arg = cat_features or None

   # --- 2) (опц.) Feature bagging
   FEATURE_BAGGING_FRAC = None
   try:
       from config import FEATURE_BAGGING_FRAC as _FBF
       FEATURE_BAGGING_FRAC = _FBF
   except Exception:
       pass

   if FEATURE_BAGGING_FRAC and 0.0 < float(FEATURE_BAGGING_FRAC) < 1.0:
       feat_cols = X_train_s.columns.tolist()
       if len(feat_cols) >= 20:
           rng = np.random.default_rng(42)
           keep_n = max(1, int(len(feat_cols) * float(FEATURE_BAGGING_FRAC)))
           keep_cols = sorted(rng.choice(feat_cols, size=keep_n, replace=False).tolist())
           X_train_s = X_train_s[keep_cols]
           if cat_features:
               cat_features = [c for c in cat_features if c in keep_cols]
           cat_features_arg = (cat_features or None)
           logger.info("[FeatureBagging] kept %d/%d features", keep_n, len(feat_cols))
       else:
           logger.info("[FeatureBagging] skipped (features=%d < 20)", len(feat_cols))

   # --- 2.1) Сохраняем фактический набор/порядок фич + cat_features
   feature_columns = X_train_s.columns.tolist()
   try:
       with open(os.path.join("models", "feature_columns.pkl"), "wb") as f:
           pickle.dump(feature_columns, f)
       with open(os.path.join("models", "cat_features.pkl"), "wb") as f:
           pickle.dump(cat_features or [], f)
       logger.info(
           "[Artifacts] Saved feature_columns.pkl (%d) & cat_features.pkl (%d)",
           len(feature_columns), len(cat_features or []),
       )
   except Exception as e:
       logger.warning("[Artifacts] Failed to save feature_columns/cat_features: %s", e)

   # --- 3) Базовые параметры модели (без жёсткого loss_function)
   params = best_params.copy()
   params["depth"] = int(params.get("depth", 6))
   params.setdefault("random_seed", 42)
   params.setdefault("iterations", 600)
   params.setdefault("bootstrap_type", "Bayesian")
   params.setdefault("rsm", 0.8)
   params.setdefault("random_strength", 2.0)
   params.setdefault("verbose", False)

   # --- 4) Class weights (по конфигу)
   sample_weight_full = None
   try:
       from config import USE_CLASS_WEIGHTS, CLASS_WEIGHT_MODE
       if USE_CLASS_WEIGHTS and CLASS_WEIGHT_MODE == "balanced":
           sample_weight_full = compute_sample_weight(class_weight="balanced", y=y_train)
           logger.info("[ClassWeight] Enabled — balanced per-sample weights applied")
   except Exception:
       pass

   # --- 5) Early stopping на последнем куске трейна (10%)
   n = len(X_train_s)
   cut = max(1, int(n * 0.9))
   X_tr, y_tr = X_train_s.iloc[:cut], y_train.iloc[:cut]
   X_val, y_val = X_train_s.iloc[cut:], y_train.iloc[cut:]

   use_es = True
   if len(X_val) < 20:
       use_es = False
       logger.warning("[FinalFit] Val chunk too small (n=%d) → disabling early stopping", len(X_val))

   # --- корректно нарезаем веса
   sw_tr = sw_val = None
   if sample_weight_full is not None and len(sample_weight_full) == len(y_train):
       sw_tr = np.asarray(sample_weight_full[:cut])
       sw_val = np.asarray(sample_weight_full[cut:]) if use_es else None

   # --- 6) Настройка loss_function / eval_metric по числу классов
   n_classes = int(pd.Series(y_train).nunique())
   if n_classes == 2:
       loss_fn = "Logloss"
       eval_metric = "F1"
   else:
       loss_fn = "MultiClass"
       eval_metric = "TotalF1"

   local_params = params.copy()
   local_params["loss_function"] = loss_fn
   local_params["eval_metric"] = eval_metric

   # --- 7) Обучение
   model = cb.CatBoostClassifier(**local_params)

   fit_kwargs = dict(
       X=X_tr,
       y=y_tr,
       sample_weight=sw_tr,
       cat_features=([X_tr.columns.get_loc(c) for c in (cat_features or [])] or None),
       verbose=False,
   )

   if use_es:
       fit_kwargs.update(
           eval_set=(X_val, y_val),
           early_stopping_rounds=40,
       )

   model.fit(**fit_kwargs)

   # --- 8) Сохранение модели
   try:
       model.save_model("models/saved_model.cbm")
       logger.info("Model saved → models/saved_model.cbm")
   except Exception as e:
       logger.warning("Failed to save model alias: %s", e)

   # --- 9) Диагностика: train-чанк vs val-чанк
   try:
       y_pred_tr = model.predict(X_tr)
       y_pred_tr = np.asarray(y_pred_tr).ravel().astype(int)
       f1_tr = f1_score(y_tr, y_pred_tr, average="macro")
       if use_es:
           y_pred_val = model.predict(X_val)
           y_pred_val = np.asarray(y_pred_val).ravel().astype(int)
           f1_val = f1_score(y_val, y_pred_val, average="macro")
           logger.info(
               "[FinalFit diag] F1(train-chunk)=%.4f | F1(val-chunk)=%.4f | gap=%.4f",
               f1_tr, f1_val, f1_tr - f1_val,
           )
       else:
           logger.info("[FinalFit diag] F1(train-chunk)=%.4f | (no val-chunk)", f1_tr)
   except Exception:
       pass

   # --- 10) Отчёты по train
   try:
       proba_tr = np.asarray(model.predict_proba(X_train_s))
       y_pred_tr = np.asarray(model.predict(X_train_s)).ravel().astype(int)
       conf_tr = proba_tr.max(axis=1)

       f1 = f1_score(y_train, y_pred_tr, average="macro")
       acc = accuracy_score(y_train, y_pred_tr)

       labels_unique = sorted(np.unique(y_train))
       prec = precision_score(
           y_train, y_pred_tr,
           average=None, labels=labels_unique, zero_division=0
       )
       rec = recall_score(
           y_train, y_pred_tr,
           average=None, labels=labels_unique, zero_division=0
       )

       logger.info("Final model — Accuracy: %.4f, F1_macro: %.4f", acc, f1)
       name_map = {0: "Down", 1: "Up", 2: "Neutral"}
       for cls_id, p, r in zip(labels_unique, prec, rec):
           logger.info("Class %s — Precision: %.3f, Recall: %.3f",
                       name_map.get(cls_id, str(cls_id)), p, r)

       ConfusionMatrixDisplay.from_predictions(
           y_train, y_pred_tr, cmap="viridis", labels=labels_unique
       )
       plt.title("Confusion Matrix (train)")
       plt.savefig("outputs/confusion_matrix.png")
       plt.close()

       importances = model.get_feature_importance(prettified=True)
       feat_col = "Feature Id" if "Feature Id" in importances.columns else (
           "Feature" if "Feature" in importances.columns else importances.columns[0]
       )
       val_col = "Importances" if "Importances" in importances.columns else importances.columns[-1]
       plt.figure(figsize=(10, 6))
       plt.barh(importances[feat_col], importances[val_col])
       plt.tight_layout()
       plt.savefig("outputs/catboost_feature_importance.png")
       plt.close()

       # отчёт по классам
       target_names = [name_map.get(c, str(c)) for c in labels_unique]
       with open("outputs/classification_report.txt", "w", encoding="utf-8") as f:
           f.write(classification_report(
               y_train, y_pred_tr,
               labels=labels_unique,
               target_names=target_names,
               zero_division=0,
           ))

       try:
           from config import CONFIDENCE_THRESHOLDS
       except Exception:
           CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7]

       for th in CONFIDENCE_THRESHOLDS:
           idx = conf_tr >= th
           cov = float(np.mean(idx)) if len(idx) else 0.0
           if np.sum(idx) == 0:
               logger.warning("[Conf >= %.2f] Coverage: %.3f — нет уверенных предсказаний", th, cov)
               continue
           y_true_c = np.asarray(y_train)[idx]
           y_pred_c = y_pred_tr[idx]
           acc_c = accuracy_score(y_true_c, y_pred_c)
           f1_c = f1_score(y_true_c, y_pred_c, average="macro", zero_division=0)
           logger.info(
               "[Conf >= %.2f] Coverage: %.3f | Acc: %.4f | F1_macro: %.4f",
               th, cov, acc_c, f1_c,
           )

           ConfusionMatrixDisplay.from_predictions(
               y_true_c, y_pred_c, cmap="viridis", labels=labels_unique
           )
           plt.title(f"Confusion Matrix @ Confidence ≥ {th:.2f} (train)")
           plt.savefig(f"outputs/conf_matrix_conf_{int(th*100)}.png")
           plt.close()

   except Exception as e:
       logger.warning("[FinalFit reports] skipped due to: %s", e)

   return model




def load_model_and_scaler(
    model_path="models/saved_model.cbm",
    scaler_path="models/scaler.pkl",
    cat_features_path="models/cat_features.pkl",
    feature_columns_path="models/feature_columns.pkl",
):
    """
    Загружает модель, скейлер, список cat-фич и (если есть) фактический список колонок,
    использованный при финальном фите (feature bagging).
    Возвращает: model, scaler, cat_features, feature_columns (или None).
    """
    import os
    import pickle
    import catboost as cb

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_path)
    scaler_path = os.path.join(base_dir, scaler_path)
    cat_features_path = os.path.join(base_dir, cat_features_path)
    feature_columns_path = os.path.join(base_dir, feature_columns_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Scaler file not found: {scaler_path}")
    if not os.path.exists(cat_features_path):
        raise FileNotFoundError(f"❌ Cat features file not found: {cat_features_path}")

    model = cb.CatBoostClassifier()
    model.load_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(cat_features_path, "rb") as f:
        cat_features = pickle.load(f)

    feature_columns = None
    if os.path.exists(feature_columns_path):
        try:
            with open(feature_columns_path, "rb") as f:
                feature_columns = pickle.load(f)
        except Exception:
            feature_columns = None  # не критично

    print(f"[DEBUG] Загружена модель: {os.path.basename(model_path)}")
    print(f"[DEBUG] Загружен скейлер: {os.path.basename(scaler_path)}")
    print(f"[DEBUG] Загружены cat_features ({len(cat_features)}): {cat_features}")
    if feature_columns is not None:
        print(f"[DEBUG] Загружены feature_columns ({len(feature_columns)})")

    return model, scaler, cat_features, feature_columns




def predict_on_batch(model, X_input, cat_features=None, feature_columns=None):
    """
    Возвращает:
      preds: List[int]
      confidences: List[float] (max prob per row)
    Если передан feature_columns — X_input будет выровнен:
      - добавим недостающие колонки с NaN,
      - отсортируем в правильном порядке.
    """
    import numpy as np
    import pandas as pd
    import catboost as cb

    X = X_input.copy()

    # 1) Выровнять колонки под feature_columns (если есть)
    if feature_columns is not None:
        # добавить недостающие
        missing = [c for c in feature_columns if c not in X.columns]
        for c in missing:
            X[c] = np.nan
        # лишние оставить — CatBoost игнорировать не будет, поэтому отфильтруем
        X = X[feature_columns]

    # 2) Санитизировать категориальные: str + fillna("__NA__")
    if cat_features:
        for c in cat_features:
            if c in X.columns:
                X[c] = X[c].astype("string").fillna("__NA__")

    # 3) CatBoost: cat_features можно передать индексами
    cat_idx = None
    if cat_features:
        cat_idx = [X.columns.get_loc(c) for c in cat_features if c in X.columns]

    pool = cb.Pool(X, cat_features=cat_idx)
    probs = model.predict_proba(pool)        # (n, C)
    preds = model.predict(pool)              # (n,) или (n,1)
    preds = np.array(preds).astype(int).ravel().tolist()
    confs = np.max(probs, axis=1).tolist()
    return preds, confs
