"""
yfinance client — batch OHLCV download for all portfolio tickers.

Design decisions:
- Single yf.download() call for all tickers (1 API hit, not N)
- Returns per-ticker DataFrames with columns: Open, High, Low, Close, Volume
- Close is adjusted (auto_adjust=True)
- Filters out any tickers yfinance couldn't fetch (cash symbols, delisted, etc.)
"""
from __future__ import annotations

import yfinance as yf
import pandas as pd


def fetch_price_history(
    tickers: list[str],
    period: str = "1y",
) -> dict[str, pd.DataFrame]:
    """
    Batch-download 1yr daily OHLCV for all tickers in a single API call.

    Returns:
        { "TICKER": DataFrame(columns=[Open, High, Low, Close, Volume], index=DatetimeIndex) }
        Tickers that yfinance couldn't fetch are omitted silently.
    """
    if not tickers:
        return {}

    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    result: dict[str, pd.DataFrame] = {}

    # yfinance 1.x always returns MultiIndex columns: (field, ticker)
    for ticker in tickers:
        try:
            df = raw.xs(ticker, level=1, axis=1)[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if not df.empty:
                result[ticker] = df
        except KeyError:
            pass  # ticker not in response (cash symbol, delisted, etc.)

    return result


def latest_price(history: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Return the most recent Close price for each ticker."""
    return {
        ticker: float(df["Close"].iloc[-1])
        for ticker, df in history.items()
        if not df.empty
    }
