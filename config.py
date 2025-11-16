# config.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Literal, List
from pathlib import Path
from dotenv import load_dotenv

# --- Load .env once ---
load_dotenv()

# ---------------- API / Secrets ----------------
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY") or ""
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET") or ""

# Fail fast in runtime tasks that require auth (не падаем при оффлайн-тренинге без ключей)
REQUIRE_API_KEYS: bool = False  # выставляй True в рантайме, False — при оффлайн обучении/тестах

# ---------------- Training / ML ----------------
USE_RESAMPLING: bool = False                # 🚫 отключаем SMOTE по умолчанию
RESAMPLING_STRATEGY: Literal["smote", "undersample", "none"] = "none"

# Включаем класс-весы вместо SMOTE
USE_CLASS_WEIGHTS: bool = True              # ✅ включить балансировку через веса
CLASS_WEIGHT_MODE: Literal["balanced", "none"] = "balanced"


# Пороги уверенности (используются в отчётах/фильтрах сигналов)
CONFIDENCE_THRESHOLDS = [0.50, 0.60, 0.70]

# ---------------- Pair Finder ----------------
MIN_HISTORY_BARS: int = 200
MIN_AVG_VOLUME: float = 500_000.0
QUOTE_ASSETS: List[str] = ["USDT"]  # можно расширить: ["USDT","BTC","ETH"]
INTERVAL: str = "15"               # строка для Bybit Spot API

# ---------------- Pair Selector / Discovery ----------------
TOP_N: int = 5
DISCOVERY_LIMIT: int = 100  # сколько пар из finder посылать в selector

# ---------------- Paths / IO ----------------
BLACKLIST_PATH: str = "blacklist.txt"
OUTPUTS_DIR: str = "outputs"
LOGS_DIR: str = "logs"
MODELS_DIR: str = "models"

# Безопасное логирование: не печатать секреты/пути к ним
SAFE_LOG: bool = True


# ===== Helper: settings object with validation =====
@dataclass(frozen=True)
class Settings:
    bybit_api_key: str
    bybit_api_secret: str
    require_api_keys: bool

    use_resampling: bool
    resampling_strategy: Literal["smote", "undersample", "none"]
    confidence_thresholds: List[float]

    min_history_bars: int
    min_avg_volume: float
    quote_assets: List[str]
    interval: str

    top_n: int
    discovery_limit: int

    blacklist_path: Path
    outputs_dir: Path
    logs_dir: Path
    models_dir: Path
    safe_log: bool

    def validate(self) -> None:
        # resampling
        if self.resampling_strategy not in {"smote", "undersample", "none"}:
            raise ValueError(f"Invalid RESAMPLING_STRATEGY={self.resampling_strategy}")
        # thresholds
        if not self.confidence_thresholds or any(not (0 < x < 1) for x in self.confidence_thresholds):
            raise ValueError("CONFIDENCE_THRESHOLDS must be in (0,1)")
        # finder
        if self.min_history_bars < 50:
            raise ValueError("MIN_HISTORY_BARS too small; set >= 50")
        if self.min_avg_volume <= 0:
            raise ValueError("MIN_AVG_VOLUME must be > 0")
        if not self.quote_assets:
            raise ValueError("QUOTE_ASSETS cannot be empty")
        # interval: Bybit принимает строки с минутами/часами; базовая проверка — числовая строка
        if not self.interval.isdigit():
            raise ValueError("INTERVAL should be a numeric string like '1','5','15','60'")

        # api keys if required
        if self.require_api_keys and (not self.bybit_api_key or not self.bybit_api_secret):
            raise RuntimeError("Bybit API keys required but missing")

        # paths
        for p in (self.outputs_dir, self.logs_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)

    @property
    def interval_int(self) -> int:
        # Иногда удобно как int (например, для окон)
        return int(self.interval)


def get_settings() -> Settings:
    s = Settings(
        bybit_api_key=BYBIT_API_KEY,
        bybit_api_secret=BYBIT_API_SECRET,
        require_api_keys=REQUIRE_API_KEYS,

        use_resampling=USE_RESAMPLING,
        resampling_strategy=RESAMPLING_STRATEGY,
        confidence_thresholds=CONFIDENCE_THRESHOLDS,

        min_history_bars=MIN_HISTORY_BARS,
        min_avg_volume=MIN_AVG_VOLUME,
        quote_assets=QUOTE_ASSETS,
        interval=INTERVAL,

        top_n=TOP_N,
        discovery_limit=DISCOVERY_LIMIT,

        blacklist_path=Path(BLACKLIST_PATH),
        outputs_dir=Path(OUTPUTS_DIR),
        logs_dir=Path(LOGS_DIR),
        models_dir=Path(MODELS_DIR),
        safe_log=SAFE_LOG,
    )
    s.validate()
    return s


# --- CV / Validation ---
EMBARGO_BARS: int = 60        # уже используется в BayesOpt/purged CV
# Для rolling CV (sanity) — можно использовать тот же эмбарго
EMBARGO_BARS_ROLLING: int = EMBARGO_BARS

# Минимальные размеры выборок на фолде (чтобы не оптимизировать по шуму)
MIN_CV_TRAIN: int = 120          # минимальный размер train в фолде
MIN_CV_VAL: int = 150            # минимальный размер val в фолде
N_SPLITS_BO: int = 2             # число фолдов в байес-опте

# Минимальные размеры окон в rolling CV (чтобы не мерить шум)
MIN_ROLL_TRAIN: int = 300
MIN_ROLL_TEST: int = 100

# --- Calibration / regularization ---
TEMPERATURE_SCALING: bool = True           # включить temperature scaling на holdout
TEMPERATURE_MIN: float = 0.8               # минимум T
TEMPERATURE_MAX: float = 3.0               # максимум T
TEMPERATURE_STEP: float = 0.1              # шаг перебора T

# --- (опционально) лёгкий feature bagging на финальном фите ---
FEATURE_BAGGING_FRAC: float | None = 0.95  # None to disable; 0.95 keeps 95% random features
