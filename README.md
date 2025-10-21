---

```markdown
# 🤖 TradingBot — v0.2.0-dev

Algo-bot on CatBoost: OHLCV fetching (Bybit/Pybit), feature generation (TA indicators, candlesticks, time, clustering), training with Bayesian optimization, liquid-pair filtering and scoring (volume/ATR/EMA/ADX + signal quality).  
Includes confidence calibration and a set of smoke tests.

---

## 📂 Project Structure

```

├── pipeline.py                 # Training by symbols + artifact archiving
├── config.py                   # Configs / thresholds / keys from .env
├── data_loader.py              # Pybit client, fetch_ohlcv(), set_client()
├── feature_engineering.py      # TA / candles / time / derivatives / KMeans, target
├── model_trainer.py            # prepare_data(), BayesOpt, RollingCV, train/save
├── pair_finder.py              # Finds candidate pairs (raw candles + filters)
├── pair_selector.py            # Pair scoring + model inference
├── confidence_calibrator.py    # max(proba) calibration (Isotonic)
├── cv_utils.py                 # Purged CV splits (stub)
├── labels.py                   # Triple-barrier labeling (stub)
├── tests/                      # Smoke tests (env/loader/finder/selector/pipeline/E2E)
├── models/                     # ⭐ artifacts (models/scaler/cat_features, NOT in git)
├── outputs/                    # plots/reports, NOT in git
├── logs/                       # logs, NOT in git
├── blacklist.txt               # manual ticker blacklist
├── .env                        # secrets (NEVER commit)
├── .env.example                # template with variable names
└── README.md

```

> `.gitignore` already excludes `models/`, `outputs/`, `logs/`, `data/` and artifacts (`*.cbm`, `*.pkl`, `*.png`, …).

---

## ⚙️ Main Components

| Module           | Purpose                                                                                                    |
|------------------|------------------------------------------------------------------------------------------------------------|
| `pair_finder`    | Selects pairs: quote asset (USDT), history, validity, avg 24h volume (prefilter + candle-based validation) |
| `pair_selector`  | Builds features → runs model inference → calibrates confidence → aggregates score (volume/ATR/EMA/ADX/signals) |
| `model_trainer`  | Data prep, optional SMOTE/undersample, BayesOpt, Rolling CV, final training + artifact saving              |
| `pipeline`       | Runs per-symbol training, confidence calibration, artifact archiving by timestamp                           |

---

## 🧪 Metrics (typical on quick runs)

- `Accuracy` (test): ~0.60–0.80  
- `F1_macro` (test): ~0.32–0.70 (depends on window/symbol/thresholds)  
- Metrics improve as confidence threshold increases (less coverage, higher quality)

> Reports include: classification report, confusion matrix, feature importance.

---

## 🔑 Environment Variables (`.env`)

Create `.env` (template: `.env.example`):

```

BYBIT_API_KEY=your_key_here
BYBIT_API_SECRET=your_secret_here

````

---

## 🛠️ Installation

```bash
git clone https://github.com/0f-d3spa1r/tradingbot.git
cd tradingbot
pip install -r requirements.txt
````

> For `TA-Lib` on Windows/Ubuntu, it’s easier to install a prebuilt wheel/library (see package docs if pip fails).

---

## 🚀 Quick Start

### 1) Environment & API sanity checks

```bash
python tests/smoke_env.py
python tests/smoke_data_loader.py
```

### 2) Pair search and selection

```bash
python tests/test_pair_finder.py     # produces shortlist
python tests/test_pair_selector.py   # scores pairs, prints top results
```

### 3) Model training (pipeline)

```bash
python pipeline.py
```

By default, it trains the configured symbols (see end of `pipeline.py`), saves artifacts to `models/`, reports to `outputs/`, and the confidence calibrator to `models/confidence_calibrator.pkl`.
Each run is archived under `models/<SYMBOL>/<YYYYMMDD_HHMMSS>/`.

---

## 🧩 How It Works (short version)

1. **Data collection** — `data_loader.fetch_ohlcv(symbol, interval)` → clean OHLCV (UTC, time-indexed).
2. **Feature generation** — `feature_engineering.select_features(df)` → TA/candle/time/derivatives/KMeans + `target`.
3. **Training** — `model_trainer.prepare_data()` → optional SMOTE/undersample, scaling;
   `optimize_catboost()` → BayesOpt; `rolling_cross_validation()`; `train_final_model()` → saves `saved_model.cbm`, `scaler.pkl`, `cat_features.pkl`.
4. **Confidence calibration** — `pipeline.evaluate_model()` trains an Isotonic regressor on `max(proba)` and prediction correctness; saves calibrator.
5. **Pair selection** — `pair_finder.get_candidate_pairs()` → prefilter (24h turnover) + history/volume/validity/blacklist checks.
   `pair_selector.evaluate_pairs()` → rebuilds features, scales with saved scaler, runs CatBoost inference, calibrates confidence, scores and ranks top-N.

---

## 🧠 Pair Scoring

Factors: normalized **volume (50-MA)**, **ATR%**, **EMA(5–20) diff**, **ADX**, **signal frequency** at calibrated threshold, **avg confidence**, **stability (std)**.
Weight coefficients are configurable (see `compute_pair_score()`).

---

## ⚙️ Configuration (`config.py`)

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

## 🧪 Smoke Suite (manual run)

```bash
python tests/test_pair_finder.py
python tests/test_pair_selector.py
python tests/smoke_training_pipeline.py
python tests/smoke_end_to_end.py
# optional: python tests/mega_smoke.py
```

All tests are **online**, using your Bybit `.env` keys.

---

## 🧭 Roadmap (R&D)

* Purged/Embargoed CV (Lopez de Prado) for robust validation
* Triple-Barrier labels + Meta-labeling
* Probability calibration & adaptive thresholds in selector (partially implemented)
* Lightweight ensemble (multiple seeds/windows)
* Backtesting with fees/slippage
* Pair Discovery → Signal Manager → Risk/Execution
* Telemetry/alerts (Telegram)

---

## 🧹 Git Hygiene

* Never commit `.env`; use `.env.example`.
* `models/`, `outputs/`, `logs/`, `data/` — **excluded from git** (`.gitignore`).
* If artifacts accidentally got committed:

  ```bash
  git rm -r --cached models outputs logs *.pkl *.cbm *.png *.json
  git commit -m "cleanup artifacts"
  ```

---

## 📎 Version

```
v0.2.0-dev — selector/finder refactor, calibrated confidence, smoke tests, training archiver
```

---

## 👤 Author

GitHub: [@0f-d3spa1r](https://github.com/0f-d3spa1r)

---

## 📄 License

MIT

```

— end of file —
```
