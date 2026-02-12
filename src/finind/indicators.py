from __future__ import annotations
import numpy as np
import pandas as pd
from .core import to_series

def sma(x, window: int = 20, min_periods: int | None = None) -> pd.Series:
    s = to_series(x)
    if window <= 0:
        raise ValueError("window must be > 0")
    mp = min_periods if min_periods is not None else window
    return s.rolling(window=window, min_periods=mp).mean().rename(f"SMA_{window}")

def ema(x, span: int = 20, adjust: bool = False, min_periods: int = 0) -> pd.Series:
    s = to_series(x)
    if span <= 0:
        raise ValueError("span must be > 0")
    out = s.ewm(span=span, adjust=adjust, min_periods=min_periods).mean()
    return out.rename(f"EMA_{span}")

def rsi(x, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing (EMA with alpha=1/period).
    """
    s = to_series(x)
    if period <= 0:
        raise ValueError("period must be > 0")

    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing = EMA(alpha=1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.rename(f"RSI_{period}")
