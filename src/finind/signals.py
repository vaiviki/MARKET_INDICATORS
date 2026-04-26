from __future__ import annotations
import pandas as pd
from .core import to_series
from .indicators import sma

def crossovers(fast: pd.Series, slow: pd.Series) -> pd.DataFrame:
    """
    Returns boolean crossover events:
    - cross_up: fast crosses above slow
    - cross_down: fast crosses below slow
    """
    f = to_series(fast, "fast")
    s = to_series(slow, "slow")

    aligned = pd.concat([f, s], axis=1).dropna()
    f2 = aligned.iloc[:, 0]
    s2 = aligned.iloc[:, 1]

    prev = (f2.shift(1) - s2.shift(1))
    now = (f2 - s2)

    cross_up = (prev <= 0) & (now > 0)
    cross_down = (prev >= 0) & (now < 0)

    return pd.DataFrame(
        {"cross_up": cross_up, "cross_down": cross_down},
        index=aligned.index
    )

def golden_cross(x, fast_window: int = 50, slow_window: int = 200) -> pd.Series:
    """
    Golden cross: SMA(fast) crosses above SMA(slow)
    """
    s_fast = sma(x, fast_window)
    s_slow = sma(x, slow_window)
    crosses = crossovers(s_fast, s_slow)
    return crosses["cross_up"].rename(f"GOLDEN_CROSS_{fast_window}_{slow_window}")

def death_cross(x, fast_window: int = 50, slow_window: int = 200) -> pd.Series:
    """
    Death cross: SMA(fast) crosses below SMA(slow)
    """
    s_fast = sma(x, fast_window)
    s_slow = sma(x, slow_window)
    crosses = crossovers(s_fast, s_slow)
    return crosses["cross_down"].rename(f"DEATH_CROSS_{fast_window}_{slow_window}")

def higher_highs(x: pd.Series, window: int = 3) -> pd.Series:
    x = to_series(x, "price")

    sh = swing_high(x, window)

    # Extract only swing highs
    swing_vals = x.where(sh)

    # Previous swing high (forward filled)
    prev_swing = swing_vals.shift(1).ffill()

    hh = (x > prev_swing) & sh

    return hh.rename("HIGHER_HIGH")

def lower_lows(x: pd.Series, window: int = 3) -> pd.Series:
    x = to_series(x, "price")

    sl = swing_low(x, window)

    # Extract only swing lows
    swing_vals = x.where(sl)

    # Previous swing low (forward filled)
    prev_swing = swing_vals.shift(1).ffill()

    ll = (x < prev_swing) & sl

    return ll.rename("LOWER_LOW")

def rsi_signal(rsi: pd.Series, overbought=70, oversold=30) -> pd.DataFrame:
    r = to_series(rsi, "rsi")

    return pd.DataFrame({
        "rsi_overbought": r > overbought,
        "rsi_oversold": r < oversold
    }, index=r.index)

def macd_crossover(macd: pd.Series, signal: pd.Series) -> pd.DataFrame:
    m = to_series(macd, "macd")
    s = to_series(signal, "signal")

    return crossovers(m, s).rename(columns={
        "cross_up": "macd_bullish",
        "cross_down": "macd_bearish"
    })
    



def swing_high(x: pd.Series, window: int = 3) -> pd.Series:
    """
    Detect swing highs using centered rolling window.
    """
    x = to_series(x, "price")
    return (x == x.rolling(window, center=True).max())


def higher_highs(x: pd.Series, window: int = 3) -> pd.Series:
    """
    True Higher High detection:
    - Identify swing highs
    - Compare current swing high with PREVIOUS swing high
    """
    x = to_series(x, "price")

    sh = swing_high(x, window)

    swing_points = x[sh]

    # Compare consecutive swing highs
    hh_points = swing_points > swing_points.shift(1)

    # Map back to full index
    hh = pd.Series(False, index=x.index)
    hh.loc[hh_points.index] = hh_points

    return hh.rename("HIGHER_HIGH")

def swing_low(x: pd.Series, window: int = 3) -> pd.Series:
    x = to_series(x, "price")
    return (x == x.rolling(window, center=True).min())


def lower_lows(x: pd.Series, window: int = 3) -> pd.Series:
    """
    True Lower Low detection using swing lows
    """
    x = to_series(x, "price")

    sl = swing_low(x, window)

    swing_points = x[sl]

    ll_points = swing_points < swing_points.shift(1)

    ll = pd.Series(False, index=x.index)
    ll.loc[ll_points.index] = ll_points

    return ll.rename("LOWER_LOW")