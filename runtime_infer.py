# runtime_infer.py
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import catboost as cb

from model_trainer import (
    _sanitize_categoricals,        # df, cat_cols -> df_sanitized
    apply_isotonic_confidence,        # (ir, conf) -> calibrated_conf (uses .predict + clamping)
)
from pipeline import _apply_temperature_scaling       # (proba, T) -> proba_T

logger = logging.getLogger(__name__)
MODEL_DIR = Path("models")

# Если хотите «валить» инференс при отсутствии cat_features.pkl — поставьте True
STRICT_MISSING_CAT_PKL = False


# ---------------- paths by levels ----------------

def _level_paths(symbol: str, ts: Optional[str]) -> List[Dict[str, Path]]:
    """Приоритет уровней: run_dir -> per-symbol aliases -> global fallback."""
    levels: List[Dict[str, Path]] = []

    # 1) run_dir
    if ts:
        run = MODEL_DIR / symbol / ts
        levels.append({
            "label": f"run_dir:{symbol}/{ts}",
            "model": run / "model.cbm",
            "feat_cols": run / "feature_columns.pkl",
            "cat_feats": run / "cat_features.pkl",
            "scaler": run / "scaler.pkl",
            "temperature": run / "temperature.json",
            "calibrator": run / "confidence_calibrator.pkl",
        })

    # 2) per-symbol aliases
    levels.append({
        "label": f"alias:{symbol}",
        "model": MODEL_DIR / f"model_{symbol}.cbm",
        "feat_cols": MODEL_DIR / f"feature_columns_{symbol}.pkl",
        "cat_feats": MODEL_DIR / f"cat_features_{symbol}.pkl",
        "scaler": MODEL_DIR / f"scaler_{symbol}.pkl",
        "temperature": MODEL_DIR / f"temperature_{symbol}.json",
        "calibrator": MODEL_DIR / f"confidence_calibrator_{symbol}.pkl",
    })

    # 3) global fallback
    levels.append({
        "label": "global",
        "model": MODEL_DIR / "saved_model.cbm",
        "feat_cols": MODEL_DIR / "feature_columns.pkl",
        "cat_feats": MODEL_DIR / "cat_features.pkl",
        "scaler": MODEL_DIR / "scaler.pkl",
        "temperature": MODEL_DIR / "temperature.json",
        "calibrator": MODEL_DIR / "confidence_calibrator.pkl",
    })

    return levels


def _exists(p: Optional[Path]) -> bool:
    return bool(p and p.exists())


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------- artifacts loading (consistent level) ----------------

def load_artifacts(symbol: str, ts: Optional[str]) -> Dict[str, object]:
    """
    Берём артефакты ЕДИНОГО уровня (не миксуем):
      1) run_dir/{symbol}/{ts}
      2) per-symbol aliases
      3) global fallback
    Уровень валиден, если есть и model, и feature_columns.
    Остальные — опциональны (cat/scaler/temperature/calibrator).
    """
    levels = _level_paths(symbol, ts)

    chosen: Optional[Dict[str, Path]] = None
    for lvl in levels:
        if _exists(lvl["model"]) and _exists(lvl["feat_cols"]):
            chosen = lvl
            break

    if chosen is None:
        rows = []
        for lvl in levels:
            rows.append(
                f"  - {lvl['label']}: model={_exists(lvl['model'])}, feature_columns={_exists(lvl['feat_cols'])}"
            )
        raise RuntimeError(
            "[Infer] No consistent artifact level found (need both model & feature_columns):\n"
            + "\n".join(rows)
        )

    # обязательные
    feature_columns = _load_pickle(chosen["feat_cols"])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise RuntimeError(f"[Infer] Invalid feature_columns at {chosen['feat_cols']}")
    model = cb.CatBoostClassifier()
    model.load_model(str(chosen["model"]))

    # опциональные
    cat_features = None
    if _exists(chosen["cat_feats"]):
        try:
            v = _load_pickle(chosen["cat_feats"])
            if isinstance(v, list):
                cat_features = v
        except Exception:
            cat_features = None

    scaler = None
    if _exists(chosen["scaler"]):
        try:
            scaler = _load_pickle(chosen["scaler"])
        except Exception:
            scaler = None

    T = 1.0
    if _exists(chosen["temperature"]):
        try:
            with open(chosen["temperature"], "r", encoding="utf-8") as f:
                T = float(json.load(f).get("T", 1.0))
        except Exception:
            T = 1.0

    calibrator = None
    if _exists(chosen["calibrator"]):
        try:
            calibrator = _load_pickle(chosen["calibrator"])
        except Exception:
            calibrator = None

    logger.info("[Infer] Using artifacts level: %s", chosen["label"])
    return {
        "level_label": chosen["label"],
        "paths_used": {k: str(v) for k, v in chosen.items() if k != "label"},
        "feature_columns": feature_columns,
        "cat_features": cat_features,
        "scaler": scaler,
        "temperature": T,
        "calibrator": calibrator,
        "model": model,
    }


# ---------------- alignment & scaling ----------------

def _align_and_scale_for_infer(
    X_raw: pd.DataFrame,
    feature_columns: List[str],
    cat_features: Optional[List[str]],
    scaler: Optional[object],
) -> Tuple[pd.DataFrame, List[int]]:
    """
    1) Жёстко выравниваем X_raw под feature_columns (добавляем недостающие нулями, дропаем лишние, упорядочиваем).
    2) Категориальные: если cat_features есть — пересечение; если нет — Fallback по dtypes (string/category)
       + WARNING о риске int-категориальных.
    3) Скейлинг: ТОЛЬКО если у скейлера есть feature_names_in_ и список числовых фич
       **ровно** совпадает по составу и порядку. Иначе — WARNING и без скейлинга.
    """
    if not feature_columns:
        raise RuntimeError("[Infer] Empty feature_columns — cannot align data.")

    X = X_raw.copy()

    # 1) выравнивание под feature_columns
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X.reindex(columns=feature_columns, copy=True)

    # 2) категориальные
    if cat_features is None:
        inferred_cats = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if STRICT_MISSING_CAT_PKL:
            raise RuntimeError(
                "[Infer] cat_features.pkl is missing and STRICT_MISSING_CAT_PKL=True. "
                "Cannot reliably reconstruct categorical schema (int-coded categories possible)."
            )
        logger.warning(
            "[Infer] cat_features.pkl not found. Falling back to dtype-based cats: %s "
            "(⚠ int-coded categories will NOT be detected!)",
            inferred_cats[:10],
        )
        cat_cols = inferred_cats
    else:
        cat_cols = [c for c in cat_features if c in X.columns]

    X = _sanitize_categoricals(X, cat_cols)

    # 3) скейлинг числовых
    if scaler is not None:
        numeric_cols = [c for c in X.columns if c not in cat_cols]
        if hasattr(scaler, "feature_names_in_"):
            required = list(scaler.feature_names_in_)
            current = list(numeric_cols)
            if required != current:
                logger.warning(
                    "[Infer][Scaler] Skipping scaling: numeric feature set mismatch.\n"
                    "  required (scaler): %s\n"
                    "  current (X):       %s",
                    required, current
                )
            else:
                X.loc[:, numeric_cols] = scaler.transform(X[numeric_cols])
        else:
            # Без feature_names_in_ нельзя гарантировать совпадение порядка/набора — безопаснее пропустить.
            logger.warning(
                "[Infer][Scaler] Skipping scaling: scaler has no feature_names_in_ attribute "
                "(cannot ensure exact feature alignment)."
            )

    # индексы катфичей
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    return X, cat_idx


# ---------------- predict & postprocess ----------------

def predict_proba_catboost(
    model: cb.CatBoostClassifier, X: pd.DataFrame, cat_idx: List[int]
) -> np.ndarray:
    pool = cb.Pool(X, cat_features=cat_idx if cat_idx else None)
    proba = model.predict_proba(pool)
    return np.asarray(proba)


def postprocess_confidence(
    proba: np.ndarray, T: float, calibrator
) -> Tuple[np.ndarray, np.ndarray]:
    """temperature → max(proba) → Isotonic.predict (если есть)."""
    proba_T = _apply_temperature_scaling(proba, T if T and T > 0 else 1.0)
    conf = proba_T.max(axis=1)
    if calibrator is not None:
        conf = apply_isotonic_confidence(calibrator, conf)  # .predict + clamping [0,1]
    return proba_T, conf


# ---------------- high-level facade ----------------

def infer_batch(
    X_raw: pd.DataFrame,
    symbol: str,
    ts: Optional[str] = None,
) -> Dict[str, object]:
    """
    1) Грузим артефакты консистентного уровня,
    2) жёстко выравниваем X, санитизируем категории, (опц.) скейлим,
    3) CatBoost → proba → Temperature → Isotonic → y_pred, conf.
    """
    arts = load_artifacts(symbol, ts)

    X_aligned, cat_idx = _align_and_scale_for_infer(
        X_raw,
        feature_columns=arts["feature_columns"],
        cat_features=arts["cat_features"],
        scaler=arts["scaler"],
    )
    proba = predict_proba_catboost(arts["model"], X_aligned, cat_idx)
    proba_T, conf = postprocess_confidence(proba, arts["temperature"], arts["calibrator"])
    y_pred = proba_T.argmax(axis=1).astype(int)

    return {
        "y_pred": y_pred,
        "proba": proba_T,
        "conf": conf,
        "X_used": X_aligned,
        "cat_idx": cat_idx,
        "artifacts_level": arts["level_label"],
        "artifacts_paths": arts["paths_used"],
    }
