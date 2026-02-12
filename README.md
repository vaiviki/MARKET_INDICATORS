# finind

Lightweight financial indicators and signals: SMA, EMA, RSI, Golden Cross.

## Install (local)
pip install -e .

## Usage
```python
import pandas as pd
from finind import sma, ema, rsi, golden_cross

df = pd.read_csv("prices.csv")  # must have Close column
df["SMA20"] = sma(df, 20)
df["EMA20"] = ema(df, 20)
df["RSI14"] = rsi(df, 14)
df["GoldenCross"] = golden_cross(df, 50, 200)

print(df.tail())
