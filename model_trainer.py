import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
import logging

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

from config import USE_RESAMPLING, RESAMPLING_STRATEGY, CONFIDENCE_THRESHOLDS
from data_loader import get_processed_ohlcv
from feature_engineering import generate_target, select_features
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)

UP_THRESHOLD = 0.002
DOWN_THRESHOLD = 0.0015

def prepare_data(symbol: str, interval: str, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = get_processed_ohlcv(symbol, interval)

    from feature_engineering import generate_clustering
    df = generate_clustering(df)

    X, y = select_features(df)

    logger.info("Target distribution: %s", y.value_counts(normalize=True).to_dict())

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    num_cols = X_train.select_dtypes(include=["number"]).columns

    # 🔁 Балансировка только на трейне
    if USE_RESAMPLING:
        if RESAMPLING_STRATEGY == "smote":
            logger.info("[Resampling] Applying SMOTE on train set")

            X_train_num = X_train[num_cols]
            X_train_cat = X_train[cat_cols].reset_index(drop=True)

            smote = SMOTE(random_state=42)
            X_num_res, y_train_res = smote.fit_resample(X_train_num, y_train)

            # 🧩 Безопасная репликация категориальных
            rep = max(1, int(np.ceil(len(y_train_res) / max(1, len(X_train_cat)))))
            X_cat_rep = pd.concat([X_train_cat] * rep, ignore_index=True).iloc[:len(y_train_res)]

            X_train = pd.concat(
                [pd.DataFrame(X_num_res, columns=num_cols), X_cat_rep.reset_index(drop=True)],
                axis=1
            )
            y_train = y_train_res.reset_index(drop=True)


        elif RESAMPLING_STRATEGY == "undersample":
            logger.info("[Resampling] Applying RandomUnderSampler on train set")
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)

    # ♻️ Масштабируем числовые фичи
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols)

    # ⛓ Объединяем с категориальными
    X_train_scaled = pd.concat([X_train_scaled, X_train[cat_cols].reset_index(drop=True)], axis=1)
    X_test_scaled = pd.concat([X_test_scaled, X_test[cat_cols].reset_index(drop=True)], axis=1)

    # всегда сохраняем рядом с model_trainer.py → PROJECT_ROOT/models/scaler.pkl
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)




def optimize_catboost(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    def evaluate(depth, learning_rate, l2_leaf_reg, bagging_temperature):
        params = {
            "iterations": 300,
            "depth": int(depth),
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "bagging_temperature": bagging_temperature,
            "loss_function": "MultiClass",
            "verbose": False,
            "random_state": 42
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = cb.CatBoostClassifier(**params)
            try:
                model.fit(
                    X_t, y_t,
                    eval_set=(X_v, y_v),
                    cat_features=X_t.select_dtypes(include=["category", "object"]).columns.tolist(),
                    early_stopping_rounds=20
                )
                score = f1_score(y_v, model.predict(X_v), average="macro")
            except Exception as e:
                logger.warning(f"[BayesOpt] fold failed: {e}")
                score = 0.0
            scores.append(score)
        return np.mean(scores)

    optimizer = BayesianOptimization(
        f=evaluate,
        pbounds={"depth": (3, 8), "learning_rate": (0.01, 0.2), "l2_leaf_reg": (1, 10), "bagging_temperature": (0, 1)},
        random_state=42
    )
    optimizer.maximize(init_points=3, n_iter=7)
    logger.info(f"Best CatBoost params: {optimizer.max['params']}")
    return optimizer.max['params']

def rolling_cross_validation(X: pd.DataFrame, y: pd.Series, model_params: dict, n_splits: int = 5):
    logger.info("Rolling CV started")
    scores = []
    total_len = len(X)
    window_size = int(total_len * 0.6)
    test_size = int(total_len * 0.2)

    # Копируем и преобразуем параметры
    params = model_params.copy()
    params["depth"] = int(params["depth"])  # ✅ Приведение к int

    for i in range(n_splits):
        train_start = int(i * total_len * 0.1)
        train_end = train_start + window_size
        test_end = train_end + test_size
        if test_end > total_len:
            break

        X_train, X_test = X.iloc[train_start:train_end], X.iloc[train_end:test_end]
        y_train, y_test = y.iloc[train_start:train_end], y.iloc[train_end:test_end]

        model = cb.CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            cat_features=X_train.select_dtypes(include=["category", "object"]).columns.tolist(),
            verbose=False
        )

        score = f1_score(y_test, model.predict(X_test), average="macro")
        logger.info(f"Fold {i+1}: F1_macro={score:.4f}")
        scores.append(score)

    logger.info(f"Rolling CV complete. Mean F1_macro: {np.mean(scores):.4f}")
    return scores


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, best_params: dict):
    best_params["depth"] = int(best_params["depth"])
    best_params.update({
        "loss_function": "MultiClass",
        "random_seed": 42,
        "iterations": 500,
        "verbose": False
    })

    model = cb.CatBoostClassifier(**best_params)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    cat_features = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        cat_features=cat_features,
        eval_set=(X_train, y_train),
        early_stopping_rounds=20
    )

    os.makedirs("models", exist_ok=True)
    model.save_model("models/saved_model.cbm")

    # 🆕 Сохраняем список категориальных фичей
    with open("models/cat_features.pkl", "wb") as f:
        pickle.dump(cat_features, f)

    y_pred = model.predict(X_train)
    proba = model.predict_proba(X_train)
    confidence = np.max(proba, axis=1)

    f1 = f1_score(y_train, y_pred, average="macro")
    acc = accuracy_score(y_train, y_pred)
    prec = precision_score(y_train, y_pred, average=None, labels=[0, 1, 2])
    rec = recall_score(y_train, y_pred, average=None, labels=[0, 1, 2])

    logger.info(f"Final model — Accuracy: {acc:.4f}, F1_macro: {f1:.4f}")
    for i in range(3):
        logger.info(f"Class {i} — Precision: {prec[i]:.3f}, Recall: {rec[i]:.3f}")

    ConfusionMatrixDisplay.from_predictions(y_train, y_pred, cmap="viridis")
    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    importances = model.get_feature_importance(prettified=True)
    plt.figure(figsize=(10, 6))
    plt.barh(importances["Feature Id"], importances["Importances"])
    plt.tight_layout()
    plt.savefig("outputs/catboost_feature_importance.png")
    plt.close()

    with open("outputs/classification_report.txt", "w") as f:
        f.write(classification_report(y_train, y_pred, target_names=["Down", "Up", "Neutral"]))

    for threshold in CONFIDENCE_THRESHOLDS:
        idx = confidence >= threshold
        if np.sum(idx) == 0:
            logger.warning(f"[Confidence >= {threshold:.2f}] No confident predictions.")
            continue
        y_conf = y_pred[idx]
        y_true_conf = y_train[idx]
        acc_conf = accuracy_score(y_true_conf, y_conf)
        f1_conf = f1_score(y_true_conf, y_conf, average="macro")
        logger.info(f"[Confidence >= {threshold:.2f}] Accuracy: {acc_conf:.4f}, F1_macro: {f1_conf:.4f}")
        ConfusionMatrixDisplay.from_predictions(y_true_conf, y_conf, cmap="viridis")
        plt.title(f"Confusion Matrix @ Confidence ≥ {threshold:.2f}")
        plt.savefig(f"outputs/conf_matrix_conf_{int(threshold*100)}.png")
        plt.close()



def load_model_and_scaler(
    model_path="models/saved_model.cbm",
    scaler_path="models/scaler.pkl",
    cat_features_path="models/cat_features.pkl"
):

    """
    Загружает обученную модель, скейлер и список категориальных признаков.
    Работает независимо от текущей директории (использует абсолютные пути).
    """
    import os
    import pickle
    import catboost as cb

    # Определяем абсолютные пути
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, model_path)
    scaler_path = os.path.join(base_dir, scaler_path)
    cat_features_path = os.path.join(base_dir, cat_features_path)

    # Проверяем наличие файлов
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"❌ Scaler file not found: {scaler_path}")
    if not os.path.exists(cat_features_path):
        raise FileNotFoundError(f"❌ Cat features file not found: {cat_features_path}")

    # Загружаем модель
    model = cb.CatBoostClassifier()
    model.load_model(model_path)

    # Загружаем скейлер
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Загружаем категориальные признаки
    with open(cat_features_path, "rb") as f:
        cat_features = pickle.load(f)

    print(f"[DEBUG] Загружена модель: {os.path.basename(model_path)}")
    print(f"[DEBUG] Загружен скейлер: {os.path.basename(scaler_path)}")
    print(f"[DEBUG] Загружены cat_features ({len(cat_features)}): {cat_features}")

    return model, scaler, cat_features



def predict_on_batch(model, X_input, cat_features=None):
    """
    Выполняет предсказание и возвращает классы и уверенность (макс. вероятность по строке).

    Возвращает:
        preds (List[int]): предсказанные классы
        confidences (List[float]): уверенность (max(prob) по строке)
    """
    import numpy as np
    import catboost as cb

    # Индексы категориальных (если список имён передали)
    cat_idx = None
    if cat_features is not None:
        cat_idx = [X_input.columns.get_loc(c) for c in cat_features]

    pool = cb.Pool(X_input, cat_features=cat_idx)

    probs = model.predict_proba(pool)             # shape (n, C)
    preds = model.predict(pool)                   # CatBoost возвращает shape (n,1) или (n,)
    preds = np.array(preds).astype(int).ravel().tolist()

    # ✅ Уверенность — максимум вероятности по каждой строке
    confs = np.max(probs, axis=1).tolist()

    return preds, confs