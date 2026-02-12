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
