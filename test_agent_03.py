#!/usr/bin/env python3
"""
Standalone test for agents/agent_03_market_data.py
Run: ./venv/bin/python3 test_agent_03.py
"""
from agents.agent_03_market_data import fetch

# Equity tickers from live portfolio (cash excluded)
TICKERS = ["VOO", "SNOW", "QQMG", "NVDA", "TSLA"]

snapshots = fetch(TICKERS)

print(f"\n{'='*70}")
print(f"Agent 3 — Market Data  ({len(snapshots)} tickers)")
print(f"{'='*70}\n")

for ticker, snap in snapshots.items():
    print(f"{'─'*60}")
    print(snap.summary())
    print()

# Show what the LLM context block will look like for one ticker
print(f"{'='*70}")
print("Sample LLM context block (TSLA):")
print(f"{'='*70}")
if "TSLA" in snapshots:
    print(snapshots["TSLA"].summary())
