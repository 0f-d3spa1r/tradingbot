# labels.py
import numpy as np
import pandas as pd


def compute_atr_pct(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    ATR в долях цены (процент от close): ATR(window) / close.
    Требуются колонки: high, low, close.
    """
    high = df["high"].to_numpy()
    low  = df["low"].to_numpy()
    close= df["close"].to_numpy()

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = np.abs(high - low)
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low  - prev_close)
    tr = np.maximum.reduce([tr1, tr2, tr3])

    atr = pd.Series(tr, index=df.index).rolling(window, min_periods=window).mean()
    atr_pct = (atr / df["close"]).astype(float)
    return atr_pct



def triple_barrier_labels(
    df: pd.DataFrame,
    atr_pct: pd.Series,
    tp_mul: float = 1.5,
    sl_mul: float = 1.0,
    max_horizon: int = 20,
) -> pd.Series:
    """
    Triple-barrier labeling with ATR normalization (ATR% already provided).

    Returns pd.Series of int labels (index=df.index), name='target':
        1 = UP (take-profit hit first)
        0 = DOWN (stop-loss hit first)
        2 = NEUTRAL (no barrier hit within horizon)
    """
    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    atrv  = atr_pct.to_numpy()

    n = len(df)
    out = np.full(n, 2, dtype=int)  # NEUTRAL=2

    for i in range(n):
        # нет горизонта впереди или нет ATR — оставляем NEUTRAL
        if (i + 1 >= n) or (i + max_horizon >= n) or np.isnan(atrv[i]):
            continue

        entry = close[i]
        tp = entry * (1 + tp_mul * atrv[i])
        sl = entry * (1 - sl_mul * atrv[i])

        hit_tp = hit_sl = None
        end = min(i + max_horizon, n - 1)

        # ищем первое срабатывание любой из границ
        for j in range(i + 1, end + 1):
            if hit_tp is None and high[j] >= tp:
                hit_tp = j
            if hit_sl is None and low[j]  <= sl:
                hit_sl = j
            if (hit_tp is not None) or (hit_sl is not None):
                break

        if hit_tp is None and hit_sl is None:
            out[i] = 2
        elif hit_tp is not None and (hit_sl is None or hit_tp < hit_sl):
            out[i] = 1
        else:
            out[i] = 0

    return pd.Series(out, index=df.index, name="target")


def generate_triple_barrier_target(
    df: pd.DataFrame,
    tp_mul: float = 1.5,
    sl_mul: float = 1.0,
    max_horizon: int = 20,
    atr_window: int = 14,
    atr_pct: pd.Series | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Main entrypoint — attaches 'target' using triple-barrier logic (0=DOWN, 1=UP, 2=NEUTRAL).
    - Если atr_pct не передан, считает его через compute_atr_pct(df, atr_window).
    - По умолчанию возвращает копию df с добавленной колонкой 'target'.
      Если inplace=True — добавляет в исходный df и возвращает его же.

    Требуются колонки: ['high','low','close'].
    Зависит от: compute_atr_pct(...), triple_barrier_labels(...).
    """
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"generate_triple_barrier_target: missing columns: {sorted(missing)}")

    base = df if inplace else df.copy()

    # 1) Получаем ATR% (в долях цены)
    if atr_pct is None:
        atr_pct = compute_atr_pct(base, window=int(atr_window))
    else:
        # выравниваем индекс на всякий случай
        atr_pct = atr_pct.reindex(base.index)

    # 2) Считаем метки triple-barrier
    target = triple_barrier_labels(
        df=base,
        atr_pct=atr_pct,
        tp_mul=float(tp_mul),
        sl_mul=float(sl_mul),
        max_horizon=int(max_horizon),
    ).astype(int)

    base["target"] = target
    return base



def triple_barrier_from_candles(
    df: pd.DataFrame,
    tp_mul: float = 1.5,
    sl_mul: float = 1.0,
    max_horizon: int = 20,
    atr_window: int = 14,
) -> pd.Series:
    """
    Удобный враппер: сам считает ATR% по свечам и вызывает triple_barrier_labels.
    Сигнатура сохранена: вернёт Series 'target' с 0/1/2.
    """
    atr_pct = compute_atr_pct(df, window=atr_window)
    return triple_barrier_labels(
        df=df,
        atr_pct=atr_pct,
        tp_mul=tp_mul,
        sl_mul=sl_mul,
        max_horizon=max_horizon,
    )


def triple_barrier_labels_with_h(
    df: pd.DataFrame,
    atr_pct: pd.Series,
    tp_mul: float = 1.5,
    sl_mul: float = 1.0,
    max_horizon: int = 20,
) -> pd.DataFrame:
    """
    Возвращает DataFrame с:
      - 'target' ∈ {0,1,2}
      - 'tb_h'  ∈ [0..max_horizon] — сколько баров заняло событие/таймаут
    """
    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    atrv  = atr_pct.to_numpy()

    n = len(df)
    labels = np.full(n, 2, dtype=int)
    holds  = np.zeros(n, dtype=int)

    for i in range(n):
        if (i + 1 >= n) or (i + max_horizon >= n) or np.isnan(atrv[i]):
            continue

        entry = close[i]
        tp = entry * (1 + tp_mul * atrv[i])
        sl = entry * (1 - sl_mul * atrv[i])

        hit_tp = hit_sl = None
        end = min(i + max_horizon, n - 1)

        for j in range(i + 1, end + 1):
            if hit_tp is None and high[j] >= tp:
                hit_tp = j
            if hit_sl is None and low[j]  <= sl:
                hit_sl = j
            if (hit_tp is not None) or (hit_sl is not None):
                break

        if hit_tp is None and hit_sl is None:
            labels[i] = 2
            holds[i]  = end - i
        elif hit_tp is not None and (hit_sl is None or hit_tp < hit_sl):
            labels[i] = 1
            holds[i]  = hit_tp - i
        else:
            labels[i] = 0
            holds[i]  = hit_sl - i

    return pd.DataFrame(
        {"target": labels, "tb_h": holds},
        index=df.index
    )
