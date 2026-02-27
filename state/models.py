"""
Pydantic models for portfolio data and analysis output.
Step 1: minimal — just Holding + PortfolioAnalysis.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel


class Holding(BaseModel):
    ticker: str
    name: str
    shares: float
    price: float            # current price per share
    value: float            # current market value
    account: str            # e.g. "Robinhood", "Fidelity-Z24", "Fidelity-Z31"
    asset_type: Literal["stock", "etf", "cash"]


class TickerAnalysis(BaseModel):
    ticker: str
    recommendation: Literal["BUY", "ADD", "HOLD", "REDUCE", "SELL"]
    priority: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    action_summary: str     # one-line action for the recommended actions section
    analysis: str           # full analysis paragraph


class PortfolioAnalysis(BaseModel):
    executive_summary: str
    recommended_actions: list[str]   # formatted strings, priority-sorted
    ticker_analyses: list[TickerAnalysis]
    risk_summary: str
    raw_llm_output: Optional[str] = None   # preserved for debugging
