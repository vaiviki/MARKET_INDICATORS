import yfinance as yf
import pandas as pd

from finind import sma, ema, rsi, golden_cross, death_cross

def fetch(symbol: str, start="2015-01-01") -> pd.DataFrame:
    df = yf.download(symbol, start=start, progress=False)
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
    df = fetch(symbol, start="2007-01-01")

    df["SMA20"] = sma(df, 20)
    df["EMA20"] = ema(df, 20)
    df["RSI14"] = rsi(df, 14)

    # Golden/Death cross events (booleans at crossover dates)
    df["GoldenCross_50_200"] = golden_cross(df, 50, 200)
    df["DeathCross_50_200"] = death_cross(df, 50, 200)

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
    
    print(df.tail(50))
    print(df.columns)
    print(df[df["GoldenCross_50_200"]==True])

if __name__ == "__main__":
    main()
