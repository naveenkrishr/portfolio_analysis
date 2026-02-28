"""
Agent 5 — News & Sentiment

Fetches recent headlines for all equity positions via yfinance .news
and computes keyword-based sentiment per ticker.
Cash symbols are automatically excluded.

Data pipeline:
  tools/yfinance_news.py       →  fetch_news()
  analysis/news_sentiment.py   →  compute_all()  →  NewsSnapshot per ticker
  state["news_data"]           →  dict[ticker, NewsSnapshot]

Source waterfall (future):
  PRIMARY:   yfinance .news  (no key, unlimited)
  SECONDARY: Finnhub /company-news  (needs FINNHUB_API_KEY, 60/min)
  TERTIARY:  NewsData.io     (needs NEWSDATA_API_KEY, 200 credits/day)
"""
from __future__ import annotations

import time

from analysis.news_sentiment import NewsSnapshot, compute_all
from tools.yfinance_news import fetch_news
from state.models import Holding


# ── Standalone entry point (testable without LangGraph) ───────────────────────

def fetch(tickers: list[str]) -> dict[str, NewsSnapshot]:
    """
    Fetch news and compute sentiment for the given equity tickers.

    Args:
        tickers: list of equity ticker symbols (no cash symbols)

    Returns:
        { ticker: NewsSnapshot }
    """
    raw = fetch_news(tickers)
    return compute_all(raw)


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
