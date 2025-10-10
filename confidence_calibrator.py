# confidence_calibrator.py
import os
import pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression

def fit_confidence_calibrator(raw_conf: np.ndarray, is_correct: np.ndarray):
    """
    raw_conf: shape (n,) — max(proba) из модели (0..1)
    is_correct: shape (n,) — 1 если предсказание верное, иначе 0
    """
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(raw_conf.astype(float), is_correct.astype(float))
    return ir

def save_calibrator(ir, path="models/confidence_calibrator.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ir, f)

def load_calibrator(path="models/confidence_calibrator.pkl"):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def apply_calibration(ir, raw_conf: np.ndarray):
    if ir is None:
        return raw_conf
    return ir.predict(raw_conf.astype(float))
