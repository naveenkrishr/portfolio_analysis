"""
LangGraph state for the portfolio analysis pipeline.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from state.models import Holding, PortfolioAnalysis


class PortfolioState(TypedDict):
    # Input — populated by Agent 1 (or mock_data for Step 1)
    holdings: list[Holding]
    tickers: list[str]
    total_value: float

    # Data freshness warnings from Agent 1 (cache fallbacks, partial failures)
    data_warnings: list[str]

    # Agent 3 output — dict[ticker, TechnicalSnapshot]
    market_data: Optional[dict[str, Any]]

    # Agent 4 output — dict[ticker, FundamentalSnapshot]
    fundamentals: Optional[dict[str, Any]]

    # Agent 4 output — dict[ticker, ETFSnapshot] (ETF-specific data)
    etf_data: Optional[dict[str, Any]]

    # Agent 5 output — dict[ticker, NewsSnapshot]
    news_data: Optional[dict[str, Any]]

    # Agent 6 output — dict[ticker, EarningsSnapshot]
    earnings_data: Optional[dict[str, Any]]

    # Agent 7 output — dict[ticker, InsiderSnapshot]
    insider_data: Optional[dict[str, Any]]

    # Agent 8 output — PortfolioRiskSnapshot
    risk_data: Optional[Any]

    # Agent 9 output
    analysis: Optional[PortfolioAnalysis]

    # Optional per-user email override (used by main_spreadsheet.py)
    recipient_email: Optional[str]
