"""
Agent 5 — News & Sentiment

Fetches recent headlines from multiple sources and computes keyword-based
sentiment per ticker. Cash symbols are automatically excluded.

Source priority (all fetched, then merged + deduplicated):
  PRIMARY:   Finnhub /company-news  (needs FINNHUB_API_KEY, 60/min)
  SECONDARY: FMP /stock_news        (needs FMP_API_KEY, 250/day)
  FALLBACK:  yfinance .news         (no key, unlimited)

Data pipeline:
  tools/finnhub_client.py          →  fetch_news()
  tools/fmp_client.py              →  fetch_news()
  tools/yfinance_news.py           →  fetch_news()
  _merge_and_dedup()               →  combined articles per ticker
  analysis/news_sentiment.py       →  compute_all()  →  NewsSnapshot per ticker
  state["news_data"]               →  dict[ticker, NewsSnapshot]
"""
from __future__ import annotations

import time

from analysis.news_sentiment import NewsSnapshot, compute_all
from tools import finnhub_client, fmp_client
from tools.yfinance_news import fetch_news as yf_fetch_news
from cache.cache_manager import cache
from state.models import Holding


# ── Deduplication ─────────────────────────────────────────────────────────────

def _title_words(title: str) -> set[str]:
    """Extract lowercase word tokens from a headline."""
    return set(title.lower().replace(",", " ").replace(".", " ").split())


def _is_duplicate(title_a: str, title_b: str, threshold: float = 0.8) -> bool:
    """Check if two headlines are duplicates based on word overlap."""
    words_a = _title_words(title_a)
    words_b = _title_words(title_b)
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    smaller = min(len(words_a), len(words_b))
    return (overlap / smaller) >= threshold


def _merge_and_dedup(
    sources: list[dict[str, list[dict]]],
    max_per_ticker: int = 8,
) -> dict[str, list[dict]]:
    """
    Merge articles from multiple sources and deduplicate by title similarity.

    Sources are ordered by priority — earlier sources win on duplicates.
    Returns at most max_per_ticker articles per ticker.
    """
    # Collect all tickers across sources
    all_tickers: set[str] = set()
    for src in sources:
        all_tickers.update(src.keys())

    result: dict[str, list[dict]] = {}

    for ticker in sorted(all_tickers):
        merged: list[dict] = []

        for src in sources:
            for article in src.get(ticker, []):
                # Check if this headline is a duplicate of one already kept
                title = article["title"]
                if any(_is_duplicate(title, kept["title"]) for kept in merged):
                    continue
                merged.append(article)

        # Sort by recency (lowest age_hours first), cap at limit
        merged.sort(key=lambda a: a["age_hours"])
        result[ticker] = merged[:max_per_ticker]

    return result


# ── Source tracking ───────────────────────────────────────────────────────────

def _count_per_source(
    ticker: str,
    merged: list[dict],
    finnhub: dict[str, list[dict]],
    fmp: dict[str, list[dict]],
    yfinance: dict[str, list[dict]],
) -> str:
    """Build a source breakdown string like '4 Finnhub + 2 FMP + 2 yfinance'."""
    parts = []

    # Count how many merged articles came from each source by matching titles
    merged_titles = {a["title"] for a in merged}

    for name, src in [("Finnhub", finnhub), ("FMP", fmp), ("yfinance", yfinance)]:
        src_titles = {a["title"] for a in src.get(ticker, [])}
        count = len(merged_titles & src_titles)
        if count > 0:
            parts.append(f"{count} {name}")

    return " + ".join(parts) if parts else "0"


# ── Standalone entry point ────────────────────────────────────────────────────

def _cached_fetch(category: str, fetch_fn, tickers: list[str]) -> dict[str, list[dict]]:
    """Fetch with per-ticker caching for a news source."""
    cached_data, stale = cache.partition(category, tickers)
    if stale:
        fresh = fetch_fn(stale)
        for ticker, articles in fresh.items():
            cache.set(category, ticker, articles)
        return {**cached_data, **fresh}
    return cached_data


def fetch(tickers: list[str]) -> dict[str, NewsSnapshot]:
    """
    Fetch news from all available sources, merge, deduplicate,
    and compute sentiment for the given equity tickers.
    Uses per-source caching (daily TTL).
    """
    # Fetch from all sources with caching (missing API key → returns {})
    finnhub_data = _cached_fetch("news_finnhub", finnhub_client.fetch_news, tickers)
    fmp_data = _cached_fetch("news_fmp", fmp_client.fetch_news, tickers)
    yf_data = _cached_fetch("news_yfinance", yf_fetch_news, tickers)

    # Print source stats
    fh_count = sum(len(v) for v in finnhub_data.values())
    fmp_count = sum(len(v) for v in fmp_data.values())
    yf_count = sum(len(v) for v in yf_data.values())
    print(f"  Raw headlines — Finnhub: {fh_count}, FMP: {fmp_count}, yfinance: {yf_count}")

    # Merge in priority order and deduplicate
    merged = _merge_and_dedup([finnhub_data, fmp_data, yf_data])

    # Per-ticker source breakdown
    for ticker in sorted(merged.keys()):
        breakdown = _count_per_source(ticker, merged[ticker], finnhub_data, fmp_data, yf_data)
        print(f"  {ticker}: {breakdown} = {len(merged[ticker])} headlines")

    return compute_all(merged)


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node — fetches news for all equity positions.
    Adds state["news_data"] = dict[ticker, NewsSnapshot].
    Cash positions are excluded (asset_type == "cash").
    """
    holdings: list[Holding] = state["holdings"]
    equity_tickers = [h.ticker for h in holdings if h.asset_type != "cash"]

    print("\n" + "="*70)
    print("AGENT 5 — News & Sentiment")
    print("="*70)
    print(f"Fetching news for: {', '.join(equity_tickers)}")

    t0 = time.time()
    news_data = fetch(equity_tickers)
    elapsed = time.time() - t0

    total = sum(s.headline_count for s in news_data.values())
    print(f"Done in {elapsed:.1f}s — {total} headlines across {len(news_data)} tickers")

    return {**state, "news_data": news_data}
