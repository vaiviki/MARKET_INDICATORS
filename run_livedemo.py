
from tracemalloc import start

import yfinance as yf
import pandas as pd

from finind import *
from finind.signals import *
from finind.indicators import *

def fetch(symbol: str, start="2012-01-01") -> pd.DataFrame:
    import datetime as dt

    
    end = dt.datetime.today() + dt.timedelta(days=1)
    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        raise RuntimeError(f"No data returned for symbol={symbol}")
    df.columns=df.columns.droplevel(level=1)
    df = df.reset_index()
    # Ensure expected columns exist
    need = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    return df

def main():
    symbol = "^NSEI"  # NIFTY 50
    df = fetch(symbol, start="2024-01-01")

    df["SMA20"] = sma(df, 20)
    df["EMA20"] = ema(df, 20)
    df["RSI14"] = rsi(df, 14)

    # Golden/Death cross events (booleans at crossover dates)
    df["GoldenCross_50_200"] = golden_cross(df, 50, 200)
    df["DeathCross_50_200"] = death_cross(df, 50, 200)

    df["ATR14"] = atr(df["High"], df["Low"], df["Close"], 14)
    df["HigherHigh"] = higher_highs(df["Close"])
    df["LowerLow"] = lower_lows(df["Close"])
    # Show latest values
    cols = ["Date", "Close", "SMA20", "EMA20", "RSI14", "GoldenCross_50_200", "DeathCross_50_200"]
    print(df[cols].tail(15).to_string(index=False))

    # Latest signal summary
    last = df.iloc[-1]
    print("\n--- Latest Snapshot ---")
    print("Date:", last["Date"])
    print("Close:", float(last["Close"]))
    print("RSI14:", None if pd.isna(last["RSI14"]) else round(float(last["RSI14"]), 2))
    print("GoldenCross today?:", bool(last["GoldenCross_50_200"]) if pd.notna(last["GoldenCross_50_200"]) else False)
    print("DeathCross today?:", bool(last["DeathCross_50_200"]) if pd.notna(last["DeathCross_50_200"]) else False)
    print("HigherHigh today?:", bool(last["HigherHigh"]) if pd.notna(last["HigherHigh"]) else False)
    print("LowerLow today?:", bool(last["LowerLow"]) if pd.notna(last["LowerLow"]) else False)

    print(df.tail(20))
    df.to_csv("nifty50_latest.csv", index=False)
    
if __name__ == "__main__":
    main()
