"""
FMP (Financial Modeling Prep) news client — /stock_news endpoint.

SECONDARY news source for Agent 5.
Free tier: 250 API calls/day.
API key from FMP_API_KEY env var.

Returns same article format as yfinance_news:
    {"title": str, "publisher": str, "age_hours": float, "url": str}
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

_API_KEY = os.getenv("FMP_API_KEY", "")
_BASE_URL = "https://financialmodelingprep.com/api/v3"


def fetch_news(
    tickers: list[str],
    max_per_ticker: int = 8,
    max_age_days: int = 7,
) -> dict[str, list[dict]]:
    """
    Fetch stock news from FMP for each ticker.

    Returns dict[ticker, list[article]] in the standard article format.
    Returns empty dict if no API key is configured.
    """
    if not _API_KEY:
        return {}

    now = datetime.now(tz=timezone.utc)
    max_age_sec = max_age_days * 86400

    result: dict[str, list[dict]] = {}

    failed = False
    for ticker in tickers:
        try:
            resp = requests.get(
                f"{_BASE_URL}/stock_news",
                params={
                    "tickers": ticker,
                    "limit": max_per_ticker * 2,  # fetch extra, filter by age
                    "apikey": _API_KEY,
                },
                timeout=10,
            )
            if resp.status_code in (402, 403):
                # Endpoint not available on free plan — stop trying
                if not failed:
                    print("  [info] FMP news endpoint not available on current plan — skipping")
                    failed = True
                continue
            resp.raise_for_status()
            raw = resp.json()

            if not isinstance(raw, list):
                continue

            articles = []
            for item in raw:
                title = (item.get("title") or "").strip()
                if not title:
                    continue

                pub_str = item.get("publishedDate", "")
                try:
                    pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    age_sec = now.timestamp() - pub_dt.timestamp()
                except Exception:
                    continue

                if age_sec > max_age_sec or age_sec < 0:
                    continue

                articles.append({
                    "title": title,
                    "publisher": item.get("site", "Unknown"),
                    "age_hours": round(age_sec / 3600, 1),
                    "url": item.get("url", ""),
                })

                if len(articles) >= max_per_ticker:
                    break

            if articles:
                result[ticker] = articles

        except Exception as e:
            print(f"  [warn] {ticker}: FMP news failed — {e}")

    return result
