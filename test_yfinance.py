#!/usr/bin/env python3
"""
Standalone test for tools/yfinance_client.py
Run: ./venv/bin/python3 test_yfinance.py
"""
from tools.yfinance_client import fetch_price_history, latest_price

TICKERS = ["VOO", "SNOW", "NVDA", "TSLA", "QQMG"]

print(f"Fetching price history for: {TICKERS}\n")
history = fetch_price_history(TICKERS)

print(f"Fetched {len(history)} tickers:\n")
for ticker, df in history.items():
    start = df.index[0].date()
    end   = df.index[-1].date()
    close = df["Close"].iloc[-1]
    print(f"  {ticker:6s}  {len(df)} bars  {start} → {end}  last close: ${close:.2f}")

print()
prices = latest_price(history)
print("Latest prices:")
for t, p in prices.items():
    print(f"  {t}: ${p:.2f}")
