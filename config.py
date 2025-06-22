import os
from dotenv import load_dotenv

load_dotenv()

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")


# === Ресемплирование ===
USE_RESAMPLING = True                    # включить SMOTE/undersample или нет
RESAMPLING_STRATEGY = "smote"            # допустимые значения: "smote", "undersample", "none"

# === Порог уверенности для сигналов ===
CONFIDENCE_THRESHOLDS = [0.5, 0.6, 0.7]

# config.py

# --- Pair Finder Settings ---
MIN_HISTORY_BARS = 200           # Минимум баров в истории
MIN_AVG_VOLUME = 500_000         # Минимальный средний объём
QUOTE_ASSETS = ['USDT']          # Только пары к USDT (можно расширить: ['USDT', 'BTC', 'ETH'])
INTERVAL = "15"                  # Интервал, используемый для анализа

# --- Путь к blacklist файлу ---
BLACKLIST_PATH = "blacklist.txt"
