"""
ETF data fetcher using yfinance .info + funds_data

Primary source for Agent 4 (ETF branch) — no API key required.
Fetches per-ticker (yfinance doesn't batch .info).
"""
from __future__ import annotations

import yfinance as yf


def fetch_etf_info(tickers: list[str]) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """
    Call yf.Ticker(ticker).info for each ETF ticker,
    plus attempt to get top holdings from funds_data.

    Returns:
        (raw_info_dict, top_holdings_map)
        raw_info_dict:    { ticker: info_dict }
        top_holdings_map: { ticker: [holding_name, ...] }
    """
    raw_info: dict[str, dict] = {}
    top_holdings: dict[str, list[str]] = {}

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if info and len(info) > 5:
                raw_info[ticker] = info
            else:
                print(f"  [skip] {ticker}: .info returned empty dict")
                continue

            # Try to get top holdings
            try:
                fd = t.funds_data
                if fd is not None:
                    th = fd.top_holdings
                    if th is not None and not th.empty:
                        names = th.index.tolist()[:10]
                        top_holdings[ticker] = [str(n) for n in names]
            except Exception:
                pass  # top_holdings is optional

        except Exception as e:
            print(f"  [warn] {ticker}: yfinance ETF fetch failed — {e}")

    return raw_info, top_holdings
