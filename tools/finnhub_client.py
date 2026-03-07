"""
Finnhub news client — /company-news endpoint.

PRIMARY news source for Agent 5.
Free tier: 60 API calls/min (shared across all Finnhub usage).
API key from FINNHUB_API_KEY env var.

Returns same article format as yfinance_news:
    {"title": str, "publisher": str, "age_hours": float, "url": str}
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import requests

_API_KEY = os.getenv("FINNHUB_API_KEY", "")
_BASE_URL = "https://finnhub.io/api/v1"


def fetch_news(
    tickers: list[str],
    max_per_ticker: int = 8,
    max_age_days: int = 7,
) -> dict[str, list[dict]]:
    """
    Fetch company news from Finnhub for each ticker.

    Returns dict[ticker, list[article]] in the standard article format.
    Returns empty dict if no API key is configured.
    """
    if not _API_KEY:
        return {}

    now = datetime.now(tz=timezone.utc)
    date_from = (now - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    date_to = now.strftime("%Y-%m-%d")

    result: dict[str, list[dict]] = {}

    for ticker in tickers:
        try:
            resp = requests.get(
                f"{_BASE_URL}/company-news",
                params={
                    "symbol": ticker,
                    "from": date_from,
                    "to": date_to,
                    "token": _API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()

            if not isinstance(raw, list):
                continue

            articles = []
            for item in raw:
                title = (item.get("headline") or "").strip()
                if not title:
                    continue

                pub_ts = item.get("datetime", 0)
                age_sec = now.timestamp() - pub_ts
                if age_sec > max_age_days * 86400 or age_sec < 0:
                    continue

                articles.append({
                    "title": title,
                    "publisher": item.get("source", "Unknown"),
                    "age_hours": round(age_sec / 3600, 1),
                    "url": item.get("url", ""),
                })

                if len(articles) >= max_per_ticker:
                    break

            if articles:
                result[ticker] = articles

        except Exception as e:
            print(f"  [warn] {ticker}: Finnhub news failed — {e}")

    return result
