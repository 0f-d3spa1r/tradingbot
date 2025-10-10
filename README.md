---

```markdown
# 🤖 TradingBot — v0.2.0-dev

Алго-бот на CatBoost: сбор OHLCV (Bybit/Pybit), фичи (TA-индикаторы, свечи, время, кластеризация), обучение с Bayes-оптимизацией, отбор ликвидных пар и скоринг (объём/ATR/EMA/ADX + качество сигналов). Есть калибровка уверенности и набор смоук-тестов.

---

## 📂 Структура проекта

```

├── pipeline.py                 # Тренировка по символам + архивация артефактов
├── config.py                   # Конфиги/пороговые значения/ключи из .env
├── data_loader.py              # Pybit клиент, fetch_ohlcv(), set_client()
├── feature_engineering.py      # TA/свечи/время/derivatives/KMeans, target
├── model_trainer.py            # prepare_data(), BayesOpt, RollingCV, train/save
├── pair_finder.py              # Поиск кандидатов (сырые свечи + фильтры)
├── pair_selector.py            # Скоринг пар + инференс модели
├── confidence_calibrator.py    # Калибровка max(proba) (Isotonic)
├── cv_utils.py                 # Purged CV сплиты (заготовка)
├── labels.py                   # Triple-barrier разметка (заготовка)
├── tests/                      # Смоук-тесты (env/loader/finder/selector/pipeline/E2E)
├── models/                     # ⭐ артефакты (модели/скейлер/катфичи, НЕ в git)
├── outputs/                    # графики/отчёты, НЕ в git
├── logs/                       # логи, НЕ в git
├── blacklist.txt               # ручной чёрный список тикеров
├── .env                        # секреты (НИКОГДА не коммитим)
├── .env.example                # шаблон с именами переменных
└── README.md

```

> В репозитории `.gitignore` уже исключает `models/`, `outputs/`, `logs/`, `data/` и артефакты (`*.cbm`, `*.pkl`, `*.png`, …).

---

## ⚙️ Основные компоненты

| Модуль            | Что делает                                                                                                 |
|------------------|-------------------------------------------------------------------------------------------------------------|
| `pair_finder`     | Подбирает пары: квота (USDT), история, валидность, средний объём (24h предфильтр + проверка по свечам)     |
| `pair_selector`   | Считает признаки → прогон через модель → калибрует уверенность → агрегирует скор (объём/ATR/EMA/ADX/сигналы) |
| `model_trainer`   | Подготовка данных, SMOTE/undersample (опционально), BayesOpt, Rolling CV, финальное обучение + сохранения |
| `pipeline`        | Запуск тренировки по символам, калибровка уверенности, архивация артефактов по таймстемпу                  |

---

## 🧪 Метрики (типично на быстрых запусках)

- `Accuracy` тест: ~0.60–0.80  
- `F1_macro` тест: ~0.32–0.70 (зависит от окна/символа/порогов)  
- Метрики растут при повышении порога уверенности (меньше покрытие, выше качество)

> В отчётах сохраняются: classification report, confusion matrix, feature importance.

---

## 🔑 Переменные окружения (`.env`)

Создай `.env` (шаблон см. `.env.example`):

```

BYBIT_API_KEY=your_key_here
BYBIT_API_SECRET=your_secret_here

````

---

## 🛠️ Установка

```bash
git clone https://github.com/0f-d3spa1r/tradingbot.git
cd tradingbot
pip install -r requirements.txt
````

> Для `TA-Lib` на Windows/Ubuntu удобнее поставить готовый wheel/библиотеку (см. документацию пакета, если pip ругается).

---

## 🚀 Быстрый старт

### 1) Мини-проверка окружения и API

```bash
python tests/smoke_env.py
python tests/smoke_data_loader.py
```

### 2) Поиск и отбор пар

```bash
python tests/test_pair_finder.py     # выдаёт shortlist
python tests/test_pair_selector.py   # скорит пары, печатает топ
```

### 3) Обучение модели (pipeline)

```bash
python pipeline.py
```

По умолчанию тренирует выбранные символы (см. конец `pipeline.py`), сохраняет артефакты в `models/`, отчёты в `outputs/`, калибратор уверенности — `models/confidence_calibrator.pkl`. Также архивирует комплект в `models/<SYMBOL>/<YYYYMMDD_HHMMSS>/`.

---

## 🧩 Как всё работает (коротко)

1. **Сбор данных**: `data_loader.fetch_ohlcv(symbol, interval)` → чистый OHLCV (UTC, индекс по времени).
2. **Фичи**: `feature_engineering.select_features(df)` → TA/свечи/время/derivatives/KMeans + `target`.
3. **Тренировка**: `model_trainer.prepare_data()` → SMOTE/undersample (опция), скейлинг численных;
   `optimize_catboost()` → BayesOpt; `rolling_cross_validation()`; `train_final_model()` → сохранение `saved_model.cbm`, `scaler.pkl`, `cat_features.pkl`.
4. **Калибровка уверенности**: в `pipeline.evaluate_model()` обучается Isotonic на `max(proba)` и флаге корректности предсказаний; сохраняется калибратор.
5. **Отбор пар**: `pair_finder.get_candidate_pairs()` → предфильтр (24h turnover) + проверка истории/объёма/валидности/blacklist.
   `pair_selector.evaluate_pairs()` → пересчёт фичей, скейлинг под сохранённый `scaler`, инференс CatBoost, калибровка уверенности, скоринг и топ-N.

---

## 🧠 Скоринг пары

Факторы: нормализованные **объём (50-MA)**, **ATR%**, **EMA(5–20) diff**, **ADX**, **частота сигналов** при калиброванном пороге, **средняя уверенность**, **стабильность (std)**.
Весовые коэффициенты настраиваются (см. `compute_pair_score()`).

---

## ⚙️ Конфигурация (`config.py`)

```python
USE_RESAMPLING = True
RESAMPLING_STRATEGY = "smote"   # "smote" | "undersample" | "none"
CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7]

MIN_HISTORY_BARS = 200
MIN_AVG_VOLUME   = 500_000
QUOTE_ASSETS     = ["USDT"]
INTERVAL         = "15"

BLACKLIST_PATH   = "blacklist.txt"
```

---

## 🧪 Смоук-набор (ручной запуск)

```bash
python tests/test_pair_finder.py
python tests/test_pair_selector.py
python tests/smoke_training_pipeline.py
python tests/smoke_end_to_end.py
# опционально: python tests/mega_smoke.py
```

Все тесты **онлайн**, используют твои `.env` ключи Bybit.

---

## 🧭 Дорожная карта (R&D)

* Purged/Embargoed CV (Lopez de Prado) для устойчивой валидации
* Triple-Barrier labels + Meta-labeling
* Калибровка вероятностей и адаптивные пороги в селекторе (включено частично)
* Лёгкий ансамбль (несколько сидов/окон)
* Бэктест с комиссиями/скольжением
* Pair discovery → Signal Manager → Risk/Execution
* Телеметрия/алерты (Telegram)

---

## 🧹 Git-гигиена

* `.env` не коммитим; используем `.env.example`.
* `models/`, `outputs/`, `logs/`, `data/` — **вне git** (см. `.gitignore`).
* Если артефакты попали в репозиторий:
  `git rm -r --cached models outputs logs *.pkl *.cbm *.png *.json && git commit -m "cleanup artifacts"`

---

## 📎 Версия

```
v0.2.0-dev — selector/finder refactor, calibrated confidence, smoke tests, training archiver
```

---

## 👤 Автор

GitHub: [@0f-d3spa1r](https://github.com/0f-d3spa1r)

---

## 📄 Лицензия

MIT

```

— конец файла —

::contentReference[oaicite:0]{index=0}
```
