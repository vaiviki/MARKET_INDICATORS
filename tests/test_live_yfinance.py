import pytest
import yfinance as yf
import pandas as pd

from finind import (
    sma, ema, rsi, roc,
    atr, bollinger,
    vwap, obv,
    macd,
    golden_cross, death_cross
)

@pytest.mark.slow
def test_all_indicators_on_live_data():
    symbol = "^NSEI"
    df = yf.download(symbol, start="2020-01-01", progress=False)

    if df.empty:
        pytest.skip("No live data returned.")

    df = df.reset_index()

    # Handle MultiIndex columns (yfinance issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Ensure required columns exist
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        pytest.skip("Missing required OHLCV columns.")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ======================
    # Trend
    # ======================
    df["SMA20"] = sma(close, 20)
    df["EMA20"] = ema(close, 20)

    # ======================
    # Momentum
    # ======================
    df["RSI14"] = rsi(close, 14)
    df["ROC12"] = roc(close, 12)

    # ======================
    # Volatility
    # ======================
    df["ATR14"] = atr(high, low, close, 14)

    bb = bollinger(close, 20)
    df = pd.concat([df, bb], axis=1)

    # ======================
    # Volume
    # ======================
    df["VWAP"] = vwap(high, low, close, volume)
    df["OBV"] = obv(close, volume)

    # ======================
    # MACD
    # ======================
    macd_df = macd(close)
    df = pd.concat([df, macd_df], axis=1)

    # ======================
    # Cross Signals
    # ======================
    df["GC_50_200"] = golden_cross(close, 50, 200)
    df["DC_50_200"] = death_cross(close, 50, 200)

    # ======================
    # Assertions
    # ======================

    # Basic checks
    assert df["SMA20"].notna().sum() > 10
    assert df["EMA20"].notna().sum() > 10

    # RSI range
    r = df["RSI14"].dropna()
    assert len(r) > 10
    assert (r >= 0).all() and (r <= 100).all()

    # ROC exists
    assert df["ROC12"].notna().sum() > 5

    # ATR exists
    assert df["ATR14"].notna().sum() > 5

    # Bollinger bands
    assert "BB_UPPER_20" in df.columns
    assert "BB_LOWER_20" in df.columns

    # VWAP / OBV
    assert df["VWAP"].notna().sum() > 5
    assert df["OBV"].notna().sum() > 5

    # MACD
    assert "MACD" in df.columns
    assert "MACD_SIGNAL" in df.columns
    assert "MACD_HIST" in df.columns

    # Cross signals (boolean check)
    gc = df["GC_50_200"].dropna()
    if len(gc) > 0:
        assert gc.map(lambda x: isinstance(x, bool)).all()