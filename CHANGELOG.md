# Changelog

All notable changes to this project will be documented in this file.

The format is based on **Keep a Changelog**
and this project adheres to **Semantic Versioning**.

---

## [0.2.1] - 2026-04-26

### 🚀 Added

* ATR (Average True Range) indicator using Wilder smoothing
* MACD crossover signals (`macd_bullish`, `macd_bearish`)
* RSI signal flags:

  * `rsi_overbought`
  * `rsi_oversold`
* Price structure detection:

  * Higher High (`higher_highs`)
  * Lower Low (`lower_lows`)
* Swing point detection:

  * `swing_high`
  * `swing_low`

### 🔁 Improved

* Generic `crossovers()` function for reusable signal detection
* Golden Cross / Death Cross built on top of crossover engine
* Better alignment of signals with price index
* Consistent boolean outputs for all signal functions

### 🧠 Enhanced

* Cleaner API for signal generation
* Improved readability and modular structure of `signals.py`
* Better integration between indicators and signals

### 🐛 Fixed

* Boolean dtype inconsistency in crossover signals
* Misalignment issues when merging signals with price data
* Duplicate function definitions for:

  * `higher_highs`
  * `lower_lows`

---

## [0.1.1]

### 🎉 Initial Release

### 📈 Added

* SMA (Simple Moving Average)
* EMA (Exponential Moving Average)
* RSI (Wilder’s method)

### 🔁 Signals

* Golden Cross
* Death Cross
* Basic crossover detection

### 🧩 Core

* `to_series()` helper for flexible input handling
* Pandas-based vectorized computations

---
