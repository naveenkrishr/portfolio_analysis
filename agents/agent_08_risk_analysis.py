"""
Agent 8 — Risk Analysis

Computes quantitative portfolio risk metrics from price history.
Fetches its own price data via yfinance (single batch call including SPY benchmark).

Metrics produced:
  Per-ticker:  beta, volatility, max drawdown, Sharpe ratio
  Portfolio:   VaR(95%), CVaR(95%), portfolio volatility, Sharpe, beta,
               concentration (HHI), correlation highlights, risk rating

No LLM required — pure computation.

Data pipeline:
  tools/yfinance_client.py  →  fetch_price_history()  (1 batch call, tickers + SPY)
  analysis/risk.py          →  compute()              →  PortfolioRiskSnapshot
  state["risk_data"]        →  PortfolioRiskSnapshot
"""
from __future__ import annotations

import time

from analysis.risk import PortfolioRiskSnapshot, compute
from tools.yfinance_client import fetch_price_history
from state.models import Holding


# ── Standalone entry point (testable without LangGraph) ───────────────────────

def fetch(
    holdings: list[Holding],
    total_value: float,
) -> PortfolioRiskSnapshot:
    """
    Compute portfolio risk metrics for the given holdings.

    Args:
        holdings: all holdings (cash included for weight calculation)
        total_value: total portfolio value

    Returns:
        PortfolioRiskSnapshot
    """
    equity = [h for h in holdings if h.asset_type != "cash"]
    equity_tickers = [h.ticker for h in equity]

    # Build weights (fraction of total portfolio)
    weights = {h.ticker: h.value / total_value for h in equity if total_value > 0}

    # Fetch price history for portfolio tickers + SPY benchmark
    all_tickers = list(set(equity_tickers + ["SPY"]))
    history = fetch_price_history(all_tickers)

    # Extract benchmark
    benchmark = history.pop("SPY", None)

    return compute(history, weights, benchmark)


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: dict) -> dict:
    """
    LangGraph node — computes quantitative risk metrics for the portfolio.
    Adds state["risk_data"] = PortfolioRiskSnapshot.
    """
    holdings: list[Holding] = state["holdings"]
    total_value: float = state["total_value"]

    print("\n" + "="*70)
    print("AGENT 8 — Risk Analysis")
    print("="*70)

    equity_tickers = [h.ticker for h in holdings if h.asset_type != "cash"]
    print(f"Computing risk metrics for: {', '.join(equity_tickers)}")

    t0 = time.time()
    risk_data = fetch(holdings, total_value)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s")
    if risk_data.portfolio_var_95 is not None:
        print(f"  VaR(95%): {risk_data.portfolio_var_95*100:.2f}%  |  "
              f"Vol: {risk_data.portfolio_vol*100:.1f}%  |  "
              f"Beta: {risk_data.portfolio_beta:.2f}  |  "
              f"Sharpe: {risk_data.portfolio_sharpe:.2f}")
    print(f"  Concentration: HHI={risk_data.hhi:.0f} ({risk_data.concentration_rating})")
    print(f"  Risk rating: {risk_data.risk_rating}")
    for tr in risk_data.ticker_risks.values():
        print(f"    {tr.summary()}")

    return {**state, "risk_data": risk_data}
