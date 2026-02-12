from __future__ import annotations
import pandas as pd

def to_series(x, name: str = "value") -> pd.Series:
    """
    Accepts pd.Series or pd.DataFrame (uses 'Close' if present).
    Returns a pd.Series.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
        s.name = x.name or name
        return s

    if isinstance(x, pd.DataFrame):
        if "Close" not in x.columns:
            raise ValueError("DataFrame input must contain a 'Close' column.")
        s = x["Close"].copy()
        s.name = "Close"
        return s

    raise TypeError("Input must be a pandas Series or DataFrame.")
