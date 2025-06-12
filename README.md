# TradingBot

📈 Алгоритмический трейдинг-бот на основе машинного обучения (CatBoost) с feature engineering, кластеризацией и кросс-валидацией.

## 📂 Структура проекта

- `pipeline.py` — основной скрипт запуска модели
- `data_loader.py` — загрузка данных через Pybit API
- `feature_engineering.py` — генерация признаков
- `model_trainer.py` — обучение и оценка модели
- `.env` — переменные окружения (API-ключи)
- `.gitignore` — исключённые файлы

## 🧠 Метрики

- F1_macro (test): ~0.71
- Accuracy: ~0.79
- Confidence Thresholding поддерживается

## 🚧 В разработке

- Визуализация предсказаний
- Telegram-интеграция
- Live-режим
