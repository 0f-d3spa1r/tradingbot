# labels.py
import numpy as np
import pandas as pd

def generate_triple_barrier_labels(df: pd.DataFrame, pt_mult=1.5, sl_mult=1.0, max_h=4, atr_window=14):
    """
    pt/sl в ATR-мультипликаторах. Возвращает df с колонками: 'tb_label' (1/-1/0), 'tb_h'.
    tb_label:  1 — PT раньше SL/горизонта; -1 — SL раньше PT/горизонта; 0 — neutral/timeout
    max_h — горизонт в барах (напр. 4*15m ≈ 1 час)
    """
    df = df.copy()

    tr1 = (df['high'] - df['low']).abs()
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low']  - df['close'].shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_window, min_periods=atr_window).mean().ffill()

    close = df['close']
    pt = close + pt_mult * atr
    sl = close - sl_mult * atr

    labels = np.zeros(len(df), dtype=int)
    hold   = np.zeros(len(df), dtype=int)

    for i in range(len(df)):
        end = min(i + max_h, len(df) - 1)
        if i + 1 <= end:
            hit_pt = (df['high'].iloc[i+1:end+1] >= pt.iloc[i]).any()
            hit_sl = (df['low'].iloc[i+1:end+1]  <= sl.iloc[i]).any()
        else:
            hit_pt = False
            hit_sl = False

        if hit_pt and not hit_sl:
            labels[i] = 1
        elif hit_sl and not hit_pt:
            labels[i] = -1
        else:
            labels[i] = 0

        hold[i] = end - i

    df['tb_label'] = labels
    df['tb_h'] = hold
    return df
