#!/usr/bin/env python3
"""
Standalone test for analysis/technicals.py
Run: ./venv/bin/python3 test_technicals.py
"""
from tools.yfinance_client import fetch_price_history
from analysis.technicals import compute

TICKERS = ["VOO", "SNOW", "NVDA", "TSLA", "QQMG"]

print(f"Fetching history for: {TICKERS}")
history = fetch_price_history(TICKERS)

print(f"Computing technicals for {len(history)} tickers...\n")
snapshots = compute(history)

for ticker, snap in snapshots.items():
    print(f"{'─'*60}")
    print(snap.summary())
    print()
