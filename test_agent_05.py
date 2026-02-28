#!/usr/bin/env python3
"""
Standalone test for agents/agent_05_news_sentiment.py
Run: ./venv/bin/python3 test_agent_05.py
"""
from agents.agent_05_news_sentiment import fetch

# Equity tickers from live portfolio (cash excluded)
TICKERS = ["VOO", "SNOW", "QQMG", "NVDA", "TSLA"]

print(f"\nFetching news for: {', '.join(TICKERS)}\n")
snapshots = fetch(TICKERS)

print(f"\n{'='*70}")
print(f"Agent 5 — News & Sentiment  ({len(snapshots)} tickers)")
print(f"{'='*70}\n")

for ticker, snap in snapshots.items():
    print(f"{'─'*60}")
    print(snap.summary())
    print()

# Show full sample block for one ticker
print(f"{'='*70}")
print("Sample LLM context block (TSLA):")
print(f"{'='*70}")
if "TSLA" in snapshots:
    print(snapshots["TSLA"].summary())
else:
    print("TSLA not in results")
