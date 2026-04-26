
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

def atr(high, low, close, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) using Wilder's smoothing.

    Parameters:
    - high, low, close: price series
    - period: lookback period (default 14)

    Returns:
    - ATR series
    """
    h = to_series(high, "high")
    l = to_series(low, "low")
    c = to_series(close, "close")

    if period <= 0:
        raise ValueError("period must be > 0")

    # Align inputs
    aligned = pd.concat([h, l, c], axis=1).dropna()
    h2, l2, c2 = aligned.iloc[:, 0], aligned.iloc[:, 1], aligned.iloc[:, 2]

    prev_close = c2.shift(1)

    # True Range (TR)
    tr = pd.concat([
        (h2 - l2),
        (h2 - prev_close).abs(),
        (l2 - prev_close).abs()
    ], axis=1).max(axis=1)

    # ATR using Wilder smoothing
    atr_val = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    return atr_val.rename(f"ATR_{period}")