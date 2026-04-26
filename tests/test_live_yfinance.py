import pytest
import yfinance as yf
import pandas as pd

from finind import sma, ema, rsi, golden_cross

@pytest.mark.slow
def test_indicators_on_live_data():
    symbol = "^NSEI"
    df = yf.download(symbol, start="2020-01-01", progress=False).reset_index()
    

    # If Yahoo returns empty, skip instead of failing your package tests
    if df.empty:
        pytest.skip("No live data returned (API throttling / network / symbol issue).")
    df.columns=df.columns.droplevel(level=1)
    # Compute indicators
    df["SMA20"] = sma(df, 20)
    df["EMA20"] = ema(df, 20)
    df["RSI14"] = rsi(df, 14)
    df["GC_50_200"] = golden_cross(df, 50, 200)

    # Basic sanity assertions
    assert "SMA20" in df.columns
    assert df["SMA20"].notna().sum() > 10
    assert df["EMA20"].notna().sum() > 10

    r = df["RSI14"].dropna()
    assert len(r) > 10
    assert (r >= 0).all() and (r <= 100).all()

    # Golden cross should be boolean where defined (may be all False, that's fine)
    gc = df["GC_50_200"].dropna()
    if len(gc) > 0:
        # Ensure every non-null value is a real bool
        assert gc.map(lambda x: isinstance(x, (bool,))).all()

