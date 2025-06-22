
---

```markdown
🤖 TradingBot — v0.1.0

Алгоритмический трейдинг-бот на основе машинного обучения (CatBoost). Поддерживает feature engineering, кластеризацию, отбор активов, адаптивный скоринг и кросс-валидацию. Спроектирован с учётом модульности, масштабируемости и прозрачности.

---

## 📂 Структура проекта


├── pipeline.py                  # Основной скрипт запуска пайплайна
├── config.py                    # Все глобальные параметры и константы
├── data\_loader.py               # Загрузка и обработка данных с API (Pybit)
├── feature\_engineering.py       # Генерация фичей и целевых переменных
├── model\_trainer.py             # Обучение CatBoost модели
├── pair\_selector.py             # Оценка и ранжирование торговых пар
├── pair\_finder.py               # Поиск и фильтрация кандидатов для анализа
├── logs/                        # Логи фильтрации и диагностики
├── models/                      # Сохранённые модели
├── outputs/                     # Графики, отчёты, confusion matrix
├── .env                         # Секреты и API-ключи (не коммитится)
├── .gitignore                   # Игнорируемые файлы
└── README.md                    # Документация

````

---

## ⚙️ Основной функционал

| Компонент         | Назначение                                                                 |
|------------------|----------------------------------------------------------------------------|
| `pair_finder`     | Автоматически отбирает пары по ликвидности, истории, корректности данных |
| `pair_selector`   | Оценивает пары по метрикам (объём, ATR, EMA, ADX, качество сигналов)     |
| `model_trainer`   | Обучает CatBoost с байесовской оптимизацией гиперпараметров              |
| `pipeline`        | Финальный пайплайн: от данных до обучения/инференса                       |

---

## 📐 Метрики качества (CatBoost)

| Метрика           | Значение (train/test) |
|------------------|-----------------------|
| F1 Macro          | ~0.71                 |
| Accuracy          | ~0.79                 |
| Confidence Logic  | Да (веса по уверенности) |

---

## 🧠 Алгоритм отбора пар

Система проходит через следующие этапы:

1. **`pair_finder.py`**
   - Получает список всех тикеров через API
   - Фильтрует пары по:
     - quote asset (например, USDT)
     - средней ликвидности (`MIN_AVG_VOLUME`)
     - длине истории (`MIN_HISTORY_BARS`)
     - наличию в `blacklist.txt`
     - корректности данных OHLCV

2. **`pair_selector.py`**
   - Прогоняет каждую пару через CatBoost модель
   - Извлекает сигналы, считает технические метрики
   - Рассчитывает общий `score`
   - Возвращает `top_n` лучших пар

---

## 🧩 Используемые библиотеки

- `catboost`
- `pandas`, `numpy`, `scikit-learn`, `ta-lib`, `imbalanced-learn`
- `pybit` (Binance/Bybit unified API)
- `bayesian-optimization`
- `matplotlib` (для визуализаций)

---

## 🛠️ Установка

```bash
git clone https://github.com/yourname/tradingbot.git
cd tradingbot
pip install -r requirements.txt
````

Создайте `.env` с API-ключами:

```
API_KEY=your_key
API_SECRET=your_secret
```

---

## 🚀 Быстрый запуск

```bash
python pipeline.py
```

---

## 🧪 Проверка выбора пар вручную

```python
from pair_finder import get_candidate_pairs, set_client
from pybit.unified_trading import HTTP

client = HTTP(testnet=False)
set_client(client)

pairs = get_candidate_pairs()
print(pairs)
```

---

## ⚙️ Конфигурация (`config.py`)

```python
MIN_HISTORY_BARS = 200
MIN_AVG_VOLUME = 500_000
QUOTE_ASSETS = ['USDT']
INTERVAL = "15"
BLACKLIST_PATH = "blacklist.txt"
```

---

## 📥 Черный список пар

* Файл `blacklist.txt` можно создавать вручную.
* Пары в этом списке не будут обрабатываться.
* Причины исключения логируются в `logs/filter_exclusions.csv`.

---

## 📌 To-Do / В планах

* ✅ Pair Scoring + Filtering
* ✅ CatBoost + Confidence Thresholding
* ✅ Logging исключённых пар
* 🔄 Telegram/Discord нотификации
* 🔄 Live-инференс (бот)
* 🔄 Web-дэшборд (Streamlit/Gradio)

---

## 📎 Версия

```
v0.1.0 — First full-featured release
```

---

## 👤 Автор

*GitHub:* [@0f-d3spa1r](https://github.com/0f-d3spa1r)

---

## 📄 Лицензия

MIT License
