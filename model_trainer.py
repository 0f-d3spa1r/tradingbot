import os
import pickle
import logging
from typing import Tuple, List

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
    RESAMPLING_STRATEGY,
    USE_CLASS_WEIGHTS,
    CLASS_WEIGHT_MODE,
    CONFIDENCE_THRESHOLDS,
)
from data_loader import get_processed_ohlcv
from feature_engineering import generate_target, select_features, generate_clustering

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
    logger.info("Target distribution: %s", y.value_counts(normalize=True).to_dict())

    # 2) сплит по времени (без shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    # 🔍 Лог дистрибуции таргета для контроля смещения
    logger.info("Train target dist: %s", y_train.value_counts(normalize=True).round(3).to_dict())
    logger.info("Test  target dist: %s", y_test.value_counts(normalize=True).round(3).to_dict())

    # --- ГАРД: выравниваем длины сразу после сплита (на всякий случай) ---
    if len(X_train) != len(y_train):
        n_safe = min(len(X_train), len(y_train))
        logger.warning("[prepare_data] Train length mismatch: X=%d, y=%d → aligning to %d",
                       len(X_train), len(y_train), n_safe)
        X_train = X_train.iloc[:n_safe].copy().reset_index(drop=True)
        y_train = y_train.iloc[:n_safe].copy().reset_index(drop=True)
    if len(X_test) != len(y_test):
        n_safe = min(len(X_test), len(y_test))
        logger.warning("[prepare_data] Test length mismatch: X=%d, y=%d → aligning to %d",
                       len(X_test), len(y_test), n_safe)
        X_test = X_test.iloc[:n_safe].copy().reset_index(drop=True)
        y_test = y_test.iloc[:n_safe].copy().reset_index(drop=True)

    # 3) разбиение по типам
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    if len(num_cols) == 0:
        raise ValueError("Нет числовых признаков для скейлинга — проверь select_features().")

    # --- Санитизация: убираем/заполняем NaN согласованно для X и y ---
    # (а) числовые NaN — выбрасываем строки (чтобы не портить распределения)
    mask_num_train = X_train[num_cols].notna().all(axis=1)
    mask_num_test  = X_test[num_cols].notna().all(axis=1)

    # (б) таргет без NaN
    mask_y_train = y_train.notna()
    mask_y_test  = y_test.notna()

    # (в) категориальные NaN для CatBoost — ЗАПОЛНЯЕМ строкой "__NA__" и приводим к str
    if cat_cols:
        X_train.loc[:, cat_cols] = X_train[cat_cols].astype("string").fillna("__NA__")
        X_test.loc[:, cat_cols]  = X_test[cat_cols].astype("string").fillna("__NA__")
        mask_cat_train = pd.Series(True, index=X_train.index)  # уже заполнено
        mask_cat_test  = pd.Series(True, index=X_test.index)
    else:
        mask_cat_train = pd.Series(True, index=X_train.index)
        mask_cat_test  = pd.Series(True, index=X_test.index)

    # применяем согласованные маски
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

    # 4) опциональный ресемплинг — ТОЛЬКО на train
    if USE_RESAMPLING and RESAMPLING_STRATEGY != "none":
        if RESAMPLING_STRATEGY == "smote":
            if SMOTE is None:
                logger.warning("[Resampling] imblearn не установлен — пропускаю SMOTE.")
            else:
                logger.info("[Resampling] Applying SMOTE on train set")
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

                # после SMOTE — убедимся, что категориальные остались string
                if cat_cols:
                    for c in cat_cols:
                        if c in X_train.columns:
                            X_train[c] = X_train[c].astype("string")
        elif RESAMPLING_STRATEGY == "undersample":
            if RandomUnderSampler is None:
                logger.warning("[Resampling] imblearn не установлен — пропускаю undersample.")
            else:
                logger.info("[Resampling] Applying RandomUnderSampler on train set")
                X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(
                    X_train.reset_index(drop=True), y_train.reset_index(drop=True)
                )
                y_train = y_train.astype(int)
        else:
            logger.info("[Resampling] strategy=none — пропускаю")
    else:
        logger.info("[Resampling] отключен — используем class weights на этапе обучения.")

    # 5) масштабируем числовые
    scaler = StandardScaler()
    X_train_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[num_cols]),
        columns=num_cols, index=X_train.index
    )
    X_test_num_scaled = pd.DataFrame(
        scaler.transform(X_test[num_cols]),
        columns=num_cols, index=X_test.index
    )

    # 6) собираем обратно с категориями (если они есть)
    if cat_cols:
        # гарантируем, что категории — string и без NaN
        X_train_cat = X_train[cat_cols].astype("string").fillna("__NA__").reset_index(drop=True)
        X_test_cat  = X_test[cat_cols].astype("string").fillna("__NA__").reset_index(drop=True)

        X_train_scaled = pd.concat([X_train_num_scaled.reset_index(drop=True), X_train_cat], axis=1)
        X_test_scaled  = pd.concat([X_test_num_scaled.reset_index(drop=True),  X_test_cat],  axis=1)
    else:
        X_train_scaled = X_train_num_scaled.reset_index(drop=True)
        X_test_scaled  = X_test_num_scaled.reset_index(drop=True)

    # 7) сохраняем scaler в PROJECT_ROOT/models/scaler.pkl
    project_root = os.path.dirname(os.path.abspath(__file__))  # model_trainer.py
    project_root = os.path.dirname(project_root)               # подняться на уровень проекта
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
        params = {
            "iterations": 600,  # больше итераций, будет early stop
            "depth": int(depth),
            "learning_rate": float(learning_rate),
            "l2_leaf_reg": float(l2_leaf_reg),
            "bagging_temperature": float(bagging_temperature),
            "random_strength": float(random_strength),
            "rsm": float(rsm),  # feature subsampling
            "bootstrap_type": "Bayesian",
            "loss_function": "MultiClass",
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
            "depth": (5, 7),  # чуть глубже
            "learning_rate": (0.05, 0.12),  # средняя зона
            "l2_leaf_reg": (4.0, 15.0),  # помягче
            "bagging_temperature": (0.1, 0.8),
            "random_strength": (0.5, 6.0),  # снизили верх
            "rsm": (0.70, 0.95),  # не так низко
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
        "loss_function": "MultiClass",
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

        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train_s, y_train_,
            cat_features=(cat_cols or None),
            sample_weight=sample_weight,
            verbose=False
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


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, best_params: dict):
    """
    Финальный фит CatBoost:
    - санитизация категориальных (str + fillna('__NA__'))
    - class weights (если включено в config)
    - early stopping на train (eval_set=train) — осознанно
    - сохранение модели, cat_features и диагностических отчётов
    """
    from sklearn.utils.class_weight import compute_sample_weight

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # --- ГАРД: выравниваем длины X/y (если кто-то тронул y по дороге) ---
    if len(X_train) != len(y_train):
        n_safe = min(len(X_train), len(y_train))
        logger.warning("[FinalFit] Length mismatch: X=%d, y=%d → aligning to %d",
                       len(X_train), len(y_train), n_safe)
        X_train = X_train.iloc[:n_safe].reset_index(drop=True)
        y_train = y_train.iloc[:n_safe].reset_index(drop=True)

    # --- тип y: строго int ---
    y_train = y_train.astype(int).reset_index(drop=True)

    # --- санитизация категориальных + список cat_features ---
    X_train_s, cat_features = _sanitize_categoricals(X_train)
    # CatBoost нормально принимает None вместо пустого списка
    cat_features_arg = cat_features or None

    # --- (опционально) лёгкий feature bagging: случайно «урезать» фичи на 5% ---
    try:
        from config import FEATURE_BAGGING_FRAC
    except Exception:
        FEATURE_BAGGING_FRAC = None

    if FEATURE_BAGGING_FRAC and 0.0 < FEATURE_BAGGING_FRAC < 1.0:
        feat_cols = X_train_s.columns.tolist()
        # не имеет смысла на очень маленьком числе признаков
        if len(feat_cols) >= 20:
            keep_n = max(1, int(len(feat_cols) * float(FEATURE_BAGGING_FRAC)))
            rng = np.random.default_rng(42)  # фиксируем отбор колонок
            keep_cols = sorted(rng.choice(feat_cols, size=keep_n, replace=False).tolist())

            # сужаем обучающую матрицу и список категориальных
            X_train_s = X_train_s[keep_cols]
            if cat_features:
                cat_features = [c for c in cat_features if c in keep_cols]
            cat_features_arg = (cat_features or None)

            logger.info("[FeatureBagging] kept %d/%d features", keep_n, len(feat_cols))

            # 🔒 Сохраняем список колонок, чтобы инференс использовал точно тот же порядок/набор
            os.makedirs("models", exist_ok=True)
            with open("models/feature_columns.pkl", "wb") as f:
                pickle.dump(keep_cols, f)
        else:
            logger.info("[FeatureBagging] skipped (features=%d < 20)", len(feat_cols))

    # Сохраняем фактический порядок/набор фич (для строгого выравнивания при инференсе)
    feature_columns = X_train_s.columns.tolist()
    try:
        with open(os.path.join("models", "feature_columns.pkl"), "wb") as f:
            pickle.dump(feature_columns, f)
        # (опционально) по-символьно:
        # with open(os.path.join("models", f"feature_columns_{symbol}.pkl"), "wb") as f:
        #     pickle.dump(feature_columns, f)
        logger.info("[Artifacts] Saved feature_columns.pkl (%d cols)", len(feature_columns))
    except Exception as e:
        logger.warning("Failed to save feature_columns.pkl: %s", e)

    # --- параметры модели ---
    params = best_params.copy()
    params["depth"] = int(params.get("depth", 6))
    params.update({
        "loss_function": "MultiClass",
        "random_seed": 42,
        "iterations": 500,
        "verbose": False,
    })

    # --- class weights (если включено в config) ---
    sample_weight = None
    try:
        from config import USE_CLASS_WEIGHTS, CLASS_WEIGHT_MODE
        if USE_CLASS_WEIGHTS and CLASS_WEIGHT_MODE == "balanced":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            logger.info("[ClassWeight] Enabled — balanced per-sample weights applied")
    except Exception:
        pass

    # --- обучение с ранним стопом по train (как договорились) ---
    # --- внутренний валид: последние 10% трейна ---
    # --- внутренний валид: последние 10% трейна ---
    n = len(X_train_s)
    cut = max(1, int(n * 0.9))
    X_tr, y_tr = X_train_s.iloc[:cut], y_train.iloc[:cut]
    X_val, y_val = X_train_s.iloc[cut:], y_train.iloc[cut:]

    # --- корректно нарезаем веса под train-часть ---
    sw_tr = None
    if 'sample_weight' in locals() and sample_weight is not None:
        # приводим к numpy и нарезаем по cut, если длины совпадают
        sw_arr = np.asarray(sample_weight)
        if len(sw_arr) == len(y_train):
            sw_tr = sw_arr[:cut]
        else:
            # на всякий: если вдруг уже совпадает с train-частью
            sw_tr = sw_arr

    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_tr, y_tr,
        sample_weight=sw_tr,
        cat_features=cat_features_arg,
        eval_set=(X_val, y_val),
        early_stopping_rounds=40,
        verbose=False,
    )

    # --- сохранение артефактов ---
    model.save_model("models/saved_model.cbm")
    with open("models/cat_features.pkl", "wb") as f:
        pickle.dump(cat_features, f)  # сохраняем фактический список (может быть пустым)

    # --- диагностика на train ---
    y_pred = np.asarray(model.predict(X_train_s)).ravel().astype(int)
    proba  = np.asarray(model.predict_proba(X_train_s))
    conf   = proba.max(axis=1)

    f1  = f1_score(y_train, y_pred, average="macro")
    acc = accuracy_score(y_train, y_pred)
    prec = precision_score(y_train, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    rec  = recall_score(y_train, y_pred, average=None, labels=[0, 1, 2], zero_division=0)

    logger.info("Final model — Accuracy: %.4f, F1_macro: %.4f", acc, f1)
    for i in range(3):
        logger.info("Class %d — Precision: %.3f, Recall: %.3f", i, prec[i], rec[i])

    # Confusion matrix (train)
    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, cmap="viridis")
    plt.title("Confusion Matrix (train)")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Feature importance (устойчиво к версиям CatBoost)
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

    # Текстовый отчёт
    with open("outputs/classification_report.txt", "w") as f:
        f.write(classification_report(y_train, y_pred, target_names=["Down", "Up", "Neutral"], zero_division=0))

    # Метрики на “уверенных” подмножествах
    for th in CONFIDENCE_THRESHOLDS:
        idx = conf >= th
        cov = float(np.mean(idx)) if len(idx) else 0.0
        if np.sum(idx) == 0:
            logger.warning("[Conf >= %.2f] Coverage: %.3f — нет уверенных предсказаний", th, cov)
            continue
        y_true_c = np.asarray(y_train)[idx]
        y_pred_c = y_pred[idx]
        acc_c = accuracy_score(y_true_c, y_pred_c)
        f1_c  = f1_score(y_true_c, y_pred_c, average="macro", zero_division=0)
        logger.info("[Conf >= %.2f] Coverage: %.3f | Acc: %.4f | F1_macro: %.4f", th, cov, acc_c, f1_c)

        ConfusionMatrixDisplay.from_predictions(y_true_c, y_pred_c, cmap="viridis")
        plt.title(f"Confusion Matrix @ Confidence ≥ {th:.2f} (train)")
        plt.savefig(f"outputs/conf_matrix_conf_{int(th*100)}.png")
        plt.close()


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
