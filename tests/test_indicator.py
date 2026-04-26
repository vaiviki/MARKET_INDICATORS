
import pandas as pd
from finind import sma, ema, rsi, golden_cross

def test_sma_length():
    s = pd.Series([1,2,3,4,5])
    out = sma(s, 3)
    assert len(out) == 5

def test_ema_length():
    s = pd.Series([1,2,3,4,5])
    out = ema(s, 3)
    assert len(out) == 5

def test_rsi_range():
    s = pd.Series(range(1, 200))
    out = rsi(s, 14).dropna()
    assert (out >= 0).all() and (out <= 100).all()

def test_golden_cross_returns_boolish():
    s = pd.Series(range(1, 400))
    out = golden_cross(s, 5, 10).dropna()
    assert out.dtype == bool
