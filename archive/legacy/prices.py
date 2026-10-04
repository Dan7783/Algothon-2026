"""Price data I/O for pairs trading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices(path: str | Path = "prices.txt") -> pd.DataFrame:
    """Load whitespace-separated prices.txt (ticker header + daily rows)."""
    return pd.read_csv(path, sep=r"\s+", header=0)


def extract_pair(
    prices: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, x) float arrays for a named pair."""
    if y_ticker not in prices.columns:
        raise KeyError(f"Unknown y ticker: {y_ticker}")
    if x_ticker not in prices.columns:
        raise KeyError(f"Unknown x ticker: {x_ticker}")
    y = prices[y_ticker].to_numpy(dtype=float)
    x = prices[x_ticker].to_numpy(dtype=float)
    return y, x
