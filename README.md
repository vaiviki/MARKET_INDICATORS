# finind 📊

A lightweight, fast, and clean **financial indicators & signal generation library** built on top of pandas.

Designed for:

* Quant research
* Trading systems
* Streamlit dashboards
* Backtesting pipelines

## 📜 Changelog

See full version history in [CHANGELOG.md](./CHANGELOG.md)

---

## 🚀 Features

### 📈 Indicators

* SMA (Simple Moving Average)
* EMA (Exponential Moving Average)
* RSI (Wilder’s method)
* ATR (Average True Range)

### 🔁 Crossover Signals

* Golden Cross (e.g., 50/200)
* Death Cross
* Generic crossover detection

### 📊 Price Structure Signals

* Higher Highs (HH)
* Lower Lows (LL)
* Swing High / Swing Low detection

### ⚡ Momentum Signals

* RSI Overbought / Oversold
* MACD Bullish / Bearish Crossovers

---

## 🆕 What's New (v0.2.1)

* Added **ATR indicator**
* Added **Higher High / Lower Low detection**
* Added **Swing High / Swing Low logic**
* Added **MACD crossover signals**
* Improved **crossover alignment logic**
* Cleaner boolean signal outputs

---

## 📦 Installation

```bash
pip install finind
```

---

## 🧠 Quick Example

```python
import pandas as pd
import yfinance as yf

from finind import sma, ema, rsi, atr
from finind.signals import (
    golden_cross, higher_highs, lower_lows,
    rsi_signal, macd_crossover
)

# Fetch data
df = yf.download("^NSEI", start="2022-01-01")

# Indicators
df["SMA20"] = sma(df, 20)
df["EMA20"] = ema(df, 20)
df["RSI14"] = rsi(df, 14)
df["ATR14"] = atr(df["High"], df["Low"], df["Close"])

# Signals
df["GoldenCross"] = golden_cross(df, 50, 200)
df["HigherHigh"] = higher_highs(df["Close"])
df["LowerLow"] = lower_lows(df["Close"])

rsi_flags = rsi_signal(df["RSI"])

# Example MACD (user computes MACD first)
df["EMA12"] = ema(df, 12)
df["EMA26"] = ema(df, 26)
df["MACD"] = df["EMA12"] - df["EMA26"]
df["Signal"] = ema(df["MACD"], 9)

macd_flags = macd_crossover(df["MACD"], df["Signal"])

df = pd.concat([df, rsi_flags, macd_flags], axis=1)

print(df.tail())
```

---

## 🔍 Indicator Details

### 📊 SMA

```python
sma(df, 20)
```

---

### 📊 EMA

```python
ema(df, 20)
```

---

### 📊 RSI (Wilder)

```python
rsi(df, 14)
df["RSI14"] = rsi(df, 14)
df["RSI25"] = rsi(df, 25)
```

---

### 📊 ATR

```python
atr(df["High"], df["Low"], df["Close"], period=14)
df["ATR14"] = atr(df["High"], df["Low"], df["Close"], 14)
```

Measures volatility using **True Range with Wilder smoothing**.

---

## 🔁 Signal Details

### 🔁 Golden Cross

```python
golden_cross(df, 50, 200)
```

✔ Short-term SMA crosses above long-term SMA

---

### 📉 Death Cross

```python
death_cross(df, 50, 200)
```

✔ Short-term SMA crosses below long-term SMA

---

### 📈 Higher Highs

```python
higher_highs(df["Close"], window=3)
```

✔ Detects trend continuation using swing highs

---

### 📉 Lower Lows

```python
lower_lows(df["Close"], window=3)
```

✔ Detects downtrend continuation using swing lows

---

### 🔥 RSI Signals

```python
rsi_signal(df["RSI"], overbought=70, oversold=30)
```

Returns:

* `rsi_overbought`
* `rsi_oversold`

---

### ⚡ MACD Crossovers

```python
from finind import ema
from finind.signals import macd_crossover

# Step 1: MACD
df["EMA12"] = ema(df, 12)
df["EMA26"] = ema(df, 26)
df["MACD"] = df["EMA12"] - df["EMA26"]

# Step 2: Signal line
df["Signal"] = ema(df["MACD"], 9)

# Step 3: Crossovers
macd_signals = macd_crossover(df["MACD"], df["Signal"])

df = pd.concat([df, macd_signals], axis=1)
```

Returns:

* `macd_bullish`
* `macd_bearish`

---

## 🧩 Design Philosophy

* ✅ Works directly with **pandas Series / DataFrames**
* ✅ Minimal dependencies
* ✅ Clean boolean outputs for easy backtesting
* ✅ No black-box logic
* ✅ Fully vectorized (fast)

---

## 📜 Version History

### v0.2.0

* Added ATR indicator
* Added Higher High / Lower Low detection
* Added MACD crossover signals
* Improved crossover logic

### v0.1.0

* Initial release
* SMA, EMA, RSI
* Golden / Death cross

---

## ⚠️ Disclaimer

This library is for **educational and research purposes only**.
Not financial advice.

---

## 🔥 Roadmap

* MACD (full indicator)
* Bollinger Bands
* ADX / Trend strength indicators
* Volume indicators (OBV, VWAP)
* Multi-timeframe support


---
