"""
News fetcher using yfinance .news

Primary source for Agent 5 — no API key required.
yfinance 1.x returns: [{"id": ..., "content": {"title", "pubDate", "provider", "canonicalUrl", ...}}, ...]

Fetches per-ticker (no batch API for news).
"""
from __future__ import annotations

from datetime import datetime, timezone

import yfinance as yf


def _parse_pub_date(pub_date: str) -> float:
    """Parse ISO 8601 pubDate string → Unix timestamp. Returns 0 on failure."""
    try:
        dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return 0.0


def fetch_news(
    tickers: list[str],
    max_per_ticker: int = 8,
    max_age_days: int = 7,
) -> dict[str, list[dict]]:
    """
    Fetch recent news headlines per ticker from yfinance.

    Args:
        tickers:        equity ticker symbols (cash excluded by caller)
        max_per_ticker: max headlines to keep per ticker
        max_age_days:   skip articles older than this many days

    Returns:
        dict[ticker, list[article]]
        article: {"title": str, "publisher": str, "age_hours": float, "url": str}
    """
    now = datetime.now(tz=timezone.utc).timestamp()
    max_age_sec = max_age_days * 86400

    result: dict[str, list[dict]] = {}

    for ticker in tickers:
        try:
            raw_news = yf.Ticker(ticker).news or []
            articles = []

            for item in raw_news:
                # yfinance 1.x wraps everything under "content"
                content = item.get("content") or item  # fallback: item itself (older format)

                title = content.get("title", "").strip()
                if not title:
                    continue

                pub_ts = _parse_pub_date(content.get("pubDate", ""))
                if pub_ts == 0:
                    # Fallback: try old-style providerPublishTime (unix int)
                    pub_ts = float(content.get("providerPublishTime", 0))

                age_sec = now - pub_ts
                if age_sec > max_age_sec:
                    continue

                provider = content.get("provider") or {}
                publisher = (
                    provider.get("displayName")
                    or content.get("publisher", "Unknown")
                )

                canon = content.get("canonicalUrl") or {}
                url = (
                    canon.get("url")
                    or content.get("link", "")
                    or content.get("url", "")
                )

                articles.append({
                    "title":      title,
                    "publisher":  publisher,
                    "age_hours":  round(age_sec / 3600, 1),
                    "url":        url,
                })

                if len(articles) >= max_per_ticker:
                    break

            if articles:
                result[ticker] = articles

        except Exception as e:
            print(f"  [warn] {ticker}: news fetch failed — {e}")

    return result
