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
