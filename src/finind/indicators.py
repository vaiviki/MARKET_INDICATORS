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
    h = to_series(high)
    l = to_series(low)
    c = to_series(close)

    if period <= 0:
        raise ValueError("period must be > 0")

    prev_close = c.shift(1)

    tr = pd.concat([
        (h - l),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr_val.rename(f"ATR_{period}")

def bollinger(x, window: int = 20, num_std: float = 2.0):
    s = to_series(x)

    if window <= 0:
        raise ValueError("window must be > 0")

    mean = s.rolling(window).mean()
    std = s.rolling(window).std()

    upper = mean + num_std * std
    lower = mean - num_std * std

    return pd.DataFrame({
        f"BB_MID_{window}": mean,
        f"BB_UPPER_{window}": upper,
        f"BB_LOWER_{window}": lower
    })
    
def macd(x, fast: int = 12, slow: int = 26, signal: int = 9):
    s = to_series(x)

    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("periods must be > 0")

    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_SIGNAL": signal_line,
        "MACD_HIST": hist
    })
    
    
def vwap(high, low, close, volume):
    h = to_series(high)
    l = to_series(low)
    c = to_series(close)
    v = to_series(volume)

    typical_price = (h + l + c) / 3
    cum_tp_vol = (typical_price * v).cumsum()
    cum_vol = v.cumsum()

    vwap_val = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap_val.rename("VWAP")

def obv(close, volume):
    c = to_series(close)
    v = to_series(volume)

    direction = np.sign(c.diff()).fillna(0)
    obv_val = (direction * v).cumsum()

    return obv_val.rename("OBV")

def roc(x, period: int = 12):
    s = to_series(x)

    if period <= 0:
        raise ValueError("period must be > 0")

    roc_val = ((s - s.shift(period)) / s.shift(period)) * 100
    return roc_val.rename(f"ROC_{period}")
