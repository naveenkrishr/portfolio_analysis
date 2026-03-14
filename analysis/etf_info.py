"""
ETFSnapshot — structured ETF-specific data for one ticker.

Input:  raw yfinance .info dict  (from tools/yfinance_etf.py)
Output: ETFSnapshot with .summary() for LLM context

Fields sourced from yfinance .info:
  Fund:       netExpenseRatio, totalAssets, fundFamily, fundInceptionDate, category
  Yield:      trailingAnnualDividendYield
  Risk:       beta3Year / beta
  Holdings:   (from funds_data.top_holdings if available)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ETFSnapshot:
    ticker: str

    # Fund identity
    fund_family: Optional[str] = None
    category: Optional[str] = None          # e.g. "Large Blend"
    inception_date: Optional[str] = None

    # Cost & size
    expense_ratio: Optional[float] = None   # decimal, e.g. 0.0003 = 0.03%
    total_assets: Optional[float] = None    # AUM in dollars

    # Yield
    dividend_yield: Optional[float] = None  # decimal, e.g. 0.015 = 1.5%

    # Risk
    beta: Optional[float] = None

    # Holdings & sectors
    top_holdings: list[str] = field(default_factory=list)    # top holding names/tickers
    sector_weights: dict[str, float] = field(default_factory=dict)  # sector -> weight %

    # 52-week range
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None

    def summary(self) -> str:
        """One-paragraph text summary suitable for LLM context."""
        parts = [f"{self.ticker} ETF data:"]

        if self.fund_family:
            parts.append(f"Fund family: {self.fund_family}.")
        if self.category:
            parts.append(f"Category: {self.category}.")

        if self.expense_ratio is not None:
            parts.append(f"Expense ratio: {self.expense_ratio * 100:.2f}%.")

        if self.total_assets is not None:
            aum = self.total_assets
            if aum >= 1e12:
                aum_str = f"${aum / 1e12:.2f}T"
            elif aum >= 1e9:
                aum_str = f"${aum / 1e9:.1f}B"
            elif aum >= 1e6:
                aum_str = f"${aum / 1e6:.0f}M"
            else:
                aum_str = f"${aum:,.0f}"
            parts.append(f"AUM: {aum_str}.")

        if self.dividend_yield is not None and self.dividend_yield > 0:
            parts.append(f"Dividend yield: {self.dividend_yield * 100:.2f}%.")

        if self.beta is not None:
            parts.append(f"Beta: {self.beta:.2f}.")

        if self.inception_date:
            parts.append(f"Inception: {self.inception_date}.")

        if self.top_holdings:
            top = ", ".join(self.top_holdings[:10])
            parts.append(f"Top holdings: {top}.")

        if self.sector_weights:
            sectors = ", ".join(
                f"{s}={w:.1f}%" for s, w in
                sorted(self.sector_weights.items(), key=lambda x: -x[1])[:5]
            )
            parts.append(f"Sectors: {sectors}.")

        if self.week52_high and self.week52_low:
            parts.append(f"52-wk range ${self.week52_low:.2f}-${self.week52_high:.2f}.")

        return " ".join(parts)


# ── Builders ───────────────────────────────────────────────────────────────────

def _sf(info: dict, key: str) -> Optional[float]:
    val = info.get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _ss(info: dict, key: str) -> Optional[str]:
    val = info.get(key)
    return str(val) if val is not None else None


def from_raw_info(ticker: str, info: dict, top_holdings: list[str] | None = None) -> ETFSnapshot:
    """Build an ETFSnapshot from a yfinance .info dict + optional top holdings."""
    # Sector weights from yfinance .info (if present)
    sector_weights: dict[str, float] = {}
    for key in info:
        if key.startswith("sector") and key.endswith("_weight"):
            sector_name = key.replace("_weight", "").replace("sector", "").strip("_")
            val = _sf(info, key)
            if val is not None:
                sector_weights[sector_name] = val

    return ETFSnapshot(
        ticker=ticker,
        fund_family=_ss(info, "fundFamily"),
        category=_ss(info, "category"),
        inception_date=_ss(info, "fundInceptionDate"),
        expense_ratio=_sf(info, "netExpenseRatio") or _sf(info, "annualReportExpenseRatio"),
        total_assets=_sf(info, "totalAssets"),
        dividend_yield=_sf(info, "trailingAnnualDividendYield") or _sf(info, "yield"),
        beta=_sf(info, "beta3Year") or _sf(info, "beta"),
        top_holdings=top_holdings or [],
        sector_weights=sector_weights,
        week52_high=_sf(info, "fiftyTwoWeekHigh"),
        week52_low=_sf(info, "fiftyTwoWeekLow"),
    )


def compute(tickers: list[str], raw_info: dict[str, dict], top_holdings_map: dict[str, list[str]] | None = None) -> dict[str, ETFSnapshot]:
    """
    Convert raw yfinance info dicts -> ETFSnapshot objects.

    Args:
        tickers:          ETF tickers to process
        raw_info:         output of tools.yfinance_etf.fetch_etf_info()
        top_holdings_map: optional {ticker: [holding_names]}

    Returns:
        { ticker: ETFSnapshot } — only includes tickers with data
    """
    top_map = top_holdings_map or {}
    result: dict[str, ETFSnapshot] = {}
    for ticker in tickers:
        if ticker in raw_info:
            result[ticker] = from_raw_info(ticker, raw_info[ticker], top_map.get(ticker))
    return result
