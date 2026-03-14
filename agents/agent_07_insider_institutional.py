"""
Agent 7 — Insider & Institutional

Fetches insider transactions (Form 4) and insider sentiment via Finnhub,
then produces InsiderSnapshot per ticker for LLM context.

Data pipeline:
  tools/finnhub_insider.py   →  fetch_insider_transactions()  (1 call per ticker)
  tools/finnhub_insider.py   →  fetch_insider_sentiment()     (1 call per ticker)
  analysis/insider.py        →  compute_all()                 →  InsiderSnapshot per ticker
  state["insider_data"]      →  dict[ticker, InsiderSnapshot]

No LLM required — pure data fetch + transformation.
"""
from __future__ import annotations

import time

from analysis.insider import InsiderSnapshot, compute_all
from tools.finnhub_insider import fetch_insider_transactions, fetch_insider_sentiment
from cache.cache_manager import cache
from state.models import Holding


def _cached_per_ticker(category: str, fetch_fn, tickers: list[str]) -> dict[str, any]:
    """Fetch with per-ticker caching."""
    cached_data, stale = cache.partition(category, tickers)
    if cached_data:
        print(f"  Cache hit ({category}): {list(cached_data.keys())}")
    if stale:
        fresh = fetch_fn(stale)
        for ticker, data in fresh.items():
            cache.set(category, ticker, data)
        return {**cached_data, **fresh}
    return cached_data


# ── Standalone entry point (testable without LangGraph) ───────────────────────

def fetch(tickers: list[str]) -> dict[str, InsiderSnapshot]:
    """
    Fetch insider trading data for the given equity tickers.
    Uses per-ticker caching (monthly TTL — insider data changes infrequently).

    Args:
        tickers: list of equity ticker symbols (no cash symbols)

    Returns:
        { ticker: InsiderSnapshot }
    """
    raw_transactions = _cached_per_ticker(
        "insider_transactions", fetch_insider_transactions, tickers
    )
    raw_sentiment = _cached_per_ticker(
        "insider_sentiment",
        lambda t: fetch_insider_sentiment(t),
        tickers,
    )
    return compute_all(tickers, raw_transactions, raw_sentiment)


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node — fetches insider trading data for all equity positions.
    Adds state["insider_data"] = dict[ticker, InsiderSnapshot].
    Cash positions are excluded (asset_type == "cash").
    """
    holdings: list[Holding] = state["holdings"]
    # Only fetch insider data for stocks — ETFs don't have Form 4 filings
    equity_tickers = [h.ticker for h in holdings if h.asset_type == "stock"]

    print("\n" + "="*70)
    print("AGENT 7 — Insider & Institutional")
    print("="*70)
    skipped_etfs = [h.ticker for h in holdings if h.asset_type == "etf"]
    if skipped_etfs:
        print(f"Skipping ETFs (no insider data): {', '.join(skipped_etfs)}")
    print(f"Fetching insider data for: {', '.join(equity_tickers) or '(none)'}")

    t0 = time.time()
    insider_data = fetch(equity_tickers)
    elapsed = time.time() - t0

    bullish = sum(1 for s in insider_data.values() if s.signal == "bullish")
    bearish = sum(1 for s in insider_data.values() if s.signal == "bearish")
    with_data = sum(
        1 for s in insider_data.values()
        if s.total_buys > 0 or s.total_sells > 0
    )
    print(f"Done in {elapsed:.1f}s — {with_data}/{len(insider_data)} tickers with insider activity")
    print(f"  Signals: {bullish} bullish, {bearish} bearish, {len(insider_data) - bullish - bearish} neutral/mixed")
    for ticker, snap in insider_data.items():
        print(f"  {snap.summary()}")

    return {**state, "insider_data": insider_data}
