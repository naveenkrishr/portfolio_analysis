#!/usr/bin/env python3
"""
Standalone test for agents/agent_04_fundamental.py
Run: ./venv/bin/python3 test_agent_04.py
"""
from agents.agent_04_fundamental import fetch

# Equity tickers from live portfolio (cash excluded)
TICKERS = ["VOO", "SNOW", "QQMG", "NVDA", "TSLA"]

print(f"\nFetching fundamentals for: {', '.join(TICKERS)}\n")
snapshots = fetch(TICKERS)

print(f"\n{'='*70}")
print(f"Agent 4 — Fundamental Data  ({len(snapshots)} tickers)")
print(f"{'='*70}\n")

for ticker, snap in snapshots.items():
    print(f"{'─'*60}")
    print(snap.summary())
    print()

# Show what the LLM context block looks like for one ticker
print(f"{'='*70}")
print("Sample LLM context block (NVDA):")
print(f"{'='*70}")
if "NVDA" in snapshots:
    print(snapshots["NVDA"].summary())
else:
    print("NVDA not in results")
