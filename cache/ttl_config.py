"""
TTL constants for the cache layer.

Each data category has a TTL that balances freshness against API rate limits.
"""

# Seconds
TTL_REAL_TIME = 0           # No cache — always live
TTL_HOURLY    = 3600        # 1 hour
TTL_DAILY     = 86400       # 24 hours
TTL_WEEKLY    = 604800      # 7 days (168 hours)
TTL_MONTHLY   = 2592000     # 30 days (720 hours)

# Per-category TTLs
TTL_MAP = {
    # Agent 3 — Market Data
    "price_history":          TTL_HOURLY * 12,  # 12 hours

    # Agent 4 — Fundamentals
    "fundamentals":           TTL_DAILY,       # yfinance .info (P/E, ROE, etc.)

    # Agent 5 — News
    "news_finnhub":           TTL_HOURLY,      # Finnhub company news
    "news_fmp":               TTL_HOURLY,      # FMP stock news
    "news_yfinance":          TTL_HOURLY,      # yfinance .news

    # Agent 6 — Earnings
    "earnings_calendar":      TTL_DAILY,       # Finnhub earnings calendar

    # Agent 7 — Insider
    "insider_transactions":   TTL_DAILY * 10,  # 10 days
    "insider_sentiment":      TTL_DAILY * 10,  # 10 days
}


def cache_key(category: str, ticker: str) -> str:
    """Build a cache key from category and ticker."""
    return f"{category}:{ticker.upper()}"


def batch_cache_key(category: str, tickers: list[str]) -> str:
    """Build a cache key for a batch request (e.g. earnings calendar)."""
    sorted_tickers = sorted(t.upper() for t in tickers)
    return f"{category}:batch:{','.join(sorted_tickers)}"
