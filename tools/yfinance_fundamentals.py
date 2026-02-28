"""
Fundamental data fetcher using yfinance .info

Primary source for Agent 4 — no API key required.
Fetches per-ticker (yfinance doesn't batch .info like it batches OHLCV).
"""
from __future__ import annotations

import yfinance as yf


def fetch_raw_info(tickers: list[str]) -> dict[str, dict]:
    """
    Call yf.Ticker(ticker).info for each ticker.

    Returns dict[ticker, info_dict].
    Skips tickers where .info is missing or empty (e.g. cash symbols).
    """
    result: dict[str, dict] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            # Sanity check — cash symbols return a near-empty dict
            if info and len(info) > 5:
                result[ticker] = info
            else:
                print(f"  [skip] {ticker}: .info returned empty dict")
        except Exception as e:
            print(f"  [warn] {ticker}: yfinance .info failed — {e}")
    return result
