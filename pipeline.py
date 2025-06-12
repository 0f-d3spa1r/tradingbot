import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, ConfusionMatrixDisplay
)
import catboost as cb
from pybit.unified_trading import HTTP  # ✅ заменено

from model_trainer import prepare_data, optimize_catboost, train_final_model
from data_loader import set_client
from config import CONFIDENCE_THRESHOLDS
from config import BYBIT_API_KEY, BYBIT_API_SECRET
from model_trainer import rolling_cross_validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # === Настройки ===
    symbol = "BTCUSDT"
    interval = "15"
    threshold = 0.0015

    # === Инициализация Bybit (Pybit) API ===
    client = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
    set_client(client)

    # === Подготовка данных ===
    X_train, X_test, y_train, y_test = prepare_data(symbol, interval, threshold)
    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))
    logger.info(f"Categorical features: {X_test.select_dtypes(include=['object', 'category']).columns.tolist()}")

    # === Оптимизация CatBoost ===
    best_params = optimize_catboost(X_train, y_train)

    # === Rolling Cross-Validation (по желанию) ===
    USE_ROLLING_CV = True

    if USE_ROLLING_CV:
        logger.info("⚙️  Выполняем rolling cross-validation...")
        scores = rolling_cross_validation(X_train, y_train, best_params, n_splits=5)
        logger.info("Средний F1_macro на rolling CV: %.4f", np.mean(scores))

    # === Обучение модели ===
    train_final_model(X_train, y_train, best_params)

    # === Загрузка модели ===
    model = cb.CatBoostClassifier()
    try:
        model.load_model("models/saved_model.cbm")
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise

    # === Предсказания ===
    proba = model.predict_proba(X_test)
    confidence = np.max(proba, axis=1)
    y_pred = np.argmax(proba, axis=1)

    os.makedirs("outputs", exist_ok=True)

    # === Метрики по всей выборке ===
    logger.info("=" * 30 + " [FINAL METRICS] " + "=" * 30)
    logger.info("[All] Accuracy: %.4f", accuracy_score(y_test, y_pred))
    logger.info("[All] F1 macro: %.4f", f1_score(y_test, y_pred, average='macro'))

    # === Classification Report (Full) ===
    labels_order = [0, 1, 2]
    target_names = ["Down", "Up", "Neutral"]
    report = classification_report(
        y_test, y_pred, labels=labels_order, target_names=target_names, zero_division=0
    )

    logger.info("\n" + report)

    # Также сохранить в файл
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    # === Матрица ошибок ===
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='viridis')
    plt.title("Confusion Matrix (All)")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # === Распределение предсказаний ===
    unique, counts = np.unique(y_pred, return_counts=True)
    logger.info("Predicted class distribution: %s", dict(zip(unique, counts)))

    # === Уверенные предсказания ===
    for threshold in CONFIDENCE_THRESHOLDS:
        confident_idx = confidence >= threshold
        y_conf = y_pred[confident_idx]
        y_test_conf = y_test[confident_idx]

        if len(y_conf) > 0:
            acc = accuracy_score(y_test_conf, y_conf)
            f1 = f1_score(y_test_conf, y_conf, average='macro')
            logger.info(f"[Conf >= {threshold:.2f}] Accuracy: {acc:.4f}")
            logger.info(f"[Conf >= {threshold:.2f}] F1 macro: {f1:.4f}")
            logger.info(f"[Conf >= {threshold:.2f}] Predicted distribution: {dict(zip(*np.unique(y_conf, return_counts=True)))}")

            ConfusionMatrixDisplay.from_predictions(y_test_conf, y_conf, cmap='viridis')
            plt.title(f"Conf Matrix @ Confidence ≥ {threshold}")
            plt.savefig(f"outputs/conf_matrix_conf_{int(threshold*100)}.png")
            plt.close()
        else:
            logger.warning(f"Нет уверенных прогнозов при пороге {threshold:.2f}")

    # === Classification Report ===
    labels_order = [0, 1, 2]
    target_names = ["Down", "Up", "Neutral"]
    report = classification_report(y_test, y_pred, labels=labels_order, target_names=target_names, zero_division=0)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)

    # === Важность признаков ===
    feature_importances = model.get_feature_importance(prettified=True)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Feature Id"], feature_importances["Importances"])
    plt.title("CatBoost Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("outputs/catboost_feature_importance.png")
    plt.close()

    # === Placeholder для Derivatives API ===
    # client.get_kline(category="linear", ...) — если потребуется переключение с spot на futures.
