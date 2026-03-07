#!/usr/bin/env python3
"""
Standalone test for agents/agent_07_insider_institutional.py
Run: ./venv/bin/python3 test_agent_07.py
"""
from dotenv import load_dotenv
load_dotenv()

from agents.agent_07_insider_institutional import fetch

# Equity tickers from live portfolio (cash excluded)
TICKERS = ["VOO", "SNOW", "QQMG", "NVDA", "TSLA"]

print(f"\nFetching insider data for: {', '.join(TICKERS)}\n")
snapshots = fetch(TICKERS)

print(f"\n{'='*70}")
print(f"Agent 7 — Insider & Institutional  ({len(snapshots)} tickers)")
print(f"{'='*70}\n")

for ticker, snap in snapshots.items():
    print(f"{'─'*60}")
    print(snap.summary())
    if snap.recent_transactions:
        print(f"  Recent transactions ({len(snap.recent_transactions)}):")
        for txn in snap.recent_transactions[:3]:
            direction = "BUY" if txn.get("change", 0) > 0 else "SELL"
            val = txn.get("value")
            val_str = f"  ${val:,.0f}" if val else ""
            print(f"    {txn['filing_date']}  {direction}  {abs(txn.get('change', 0)):,} sh{val_str}  — {txn['name']}")
    print()

# Show full sample block for one ticker
print(f"{'='*70}")
print("Sample LLM context block (NVDA):")
print(f"{'='*70}")
if "NVDA" in snapshots:
    print(snapshots["NVDA"].summary())
else:
    print("NVDA not in results")
