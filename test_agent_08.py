#!/usr/bin/env python3
"""
Standalone test for agents/agent_08_risk_analysis.py
Run: ./venv/bin/python3 test_agent_08.py
"""
from state.models import Holding
from agents.agent_08_risk_analysis import fetch

# Mock holdings (approximate real portfolio)
HOLDINGS = [
    Holding(ticker="VOO", name="Vanguard S&P 500 ETF", shares=298, price=550.0, value=163900, account="Fidelity", asset_type="etf"),
    Holding(ticker="SNOW", name="Snowflake Inc", shares=100, price=165.0, value=16500, account="Fidelity", asset_type="stock"),
    Holding(ticker="NVDA", name="NVIDIA Corp", shares=50, price=180.0, value=9000, account="Robinhood", asset_type="stock"),
    Holding(ticker="TSLA", name="Tesla Inc", shares=20, price=420.0, value=8400, account="Robinhood", asset_type="stock"),
    Holding(ticker="QQMG", name="Invesco NASDAQ 100 ETF", shares=1118, price=22.0, value=24596, account="Fidelity", asset_type="etf"),
    Holding(ticker="SPAXX", name="Fidelity Cash", shares=1, price=15000, value=15000, account="Fidelity", asset_type="cash"),
]

TOTAL_VALUE = sum(h.value for h in HOLDINGS)

print(f"\nPortfolio: {len(HOLDINGS)} positions, ${TOTAL_VALUE:,.0f} total")
print(f"Computing risk metrics...\n")

risk = fetch(HOLDINGS, TOTAL_VALUE)

print(f"\n{'='*70}")
print("Agent 8 — Risk Analysis")
print(f"{'='*70}\n")
print(risk.summary())
