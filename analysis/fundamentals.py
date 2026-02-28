"""
FundamentalSnapshot — structured fundamental data for one ticker.

Input:  raw yfinance .info dict  (from tools/yfinance_fundamentals.py)
Output: FundamentalSnapshot with .summary() for LLM context

Fields sourced from yfinance .info:
  Valuation:     trailingPE, forwardPE, priceToBook, enterpriseToEbitda, priceToSalesTrailing12Months
  Profitability: returnOnEquity, returnOnAssets, grossMargins, operatingMargins, profitMargins
  Growth:        revenueGrowth, earningsGrowth
  Balance sheet: debtToEquity, currentRatio
  Cash:          freeCashflow, totalCash, totalDebt
  EPS:           trailingEps, forwardEps
  Analyst:       recommendationKey, recommendationMean, numberOfAnalystOpinions,
                 targetMeanPrice, targetLowPrice, targetHighPrice
  Dividend:      dividendYield, dividendRate, payoutRatio
  Risk:          beta
  52-week:       fiftyTwoWeekHigh, fiftyTwoWeekLow
  Identity:      sector, industry
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class FundamentalSnapshot:
    ticker: str

    # Identity
    sector:   Optional[str]
    industry: Optional[str]

    # Size
    market_cap:       Optional[float]   # $
    enterprise_value: Optional[float]   # $

    # Valuation
    trailing_pe:    Optional[float]
    forward_pe:     Optional[float]
    price_to_book:  Optional[float]
    ev_to_ebitda:   Optional[float]
    price_to_sales: Optional[float]

    # Profitability (all as decimals, e.g. 0.25 = 25%)
    roe:              Optional[float]
    roa:              Optional[float]
    gross_margin:     Optional[float]
    operating_margin: Optional[float]
    profit_margin:    Optional[float]

    # Growth (YoY, decimal)
    revenue_growth:  Optional[float]
    earnings_growth: Optional[float]

    # Balance sheet
    debt_to_equity: Optional[float]
    current_ratio:  Optional[float]

    # Cash
    free_cash_flow: Optional[float]   # $
    total_cash:     Optional[float]   # $
    total_debt:     Optional[float]   # $

    # EPS
    trailing_eps: Optional[float]
    forward_eps:  Optional[float]

    # Analyst
    analyst_rating:     Optional[str]    # "buy", "hold", "sell", "strong_buy" etc.
    analyst_mean_score: Optional[float]  # 1.0=strong buy … 5.0=sell
    analyst_count:      Optional[int]
    target_price_mean:  Optional[float]
    target_price_low:   Optional[float]
    target_price_high:  Optional[float]

    # Dividend
    dividend_yield: Optional[float]   # decimal (0.015 = 1.5%)
    dividend_rate:  Optional[float]   # $/yr
    payout_ratio:   Optional[float]   # decimal

    # Risk
    beta: Optional[float]

    # 52-week range
    week52_high: Optional[float]
    week52_low:  Optional[float]

    # ── summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-paragraph text summary suitable for LLM context."""
        parts = [f"{self.ticker} fundamentals:"]

        if self.sector:
            sector_str = self.sector
            if self.industry and self.industry != self.sector:
                sector_str += f" / {self.industry}"
            parts.append(f"Sector: {sector_str}.")

        if self.market_cap is not None:
            mc = self.market_cap
            if mc >= 1e12:
                mc_str = f"${mc/1e12:.2f}T"
            elif mc >= 1e9:
                mc_str = f"${mc/1e9:.1f}B"
            else:
                mc_str = f"${mc/1e6:.0f}M"
            parts.append(f"Market cap: {mc_str}.")

        # Valuation
        val = []
        if self.trailing_pe is not None: val.append(f"P/E(ttm)={self.trailing_pe:.1f}")
        if self.forward_pe is not None:  val.append(f"P/E(fwd)={self.forward_pe:.1f}")
        if self.price_to_book is not None: val.append(f"P/B={self.price_to_book:.1f}")
        if self.ev_to_ebitda is not None: val.append(f"EV/EBITDA={self.ev_to_ebitda:.1f}")
        if val:
            parts.append("Valuation: " + ", ".join(val) + ".")

        # EPS
        eps = []
        if self.trailing_eps is not None: eps.append(f"EPS(ttm)=${self.trailing_eps:.2f}")
        if self.forward_eps is not None:  eps.append(f"EPS(fwd)=${self.forward_eps:.2f}")
        if eps:
            parts.append(" ".join(eps) + ".")

        # Profitability
        prof = []
        if self.roe is not None:           prof.append(f"ROE={self.roe*100:.1f}%")
        if self.profit_margin is not None: prof.append(f"Net margin={self.profit_margin*100:.1f}%")
        if self.gross_margin is not None:  prof.append(f"Gross margin={self.gross_margin*100:.1f}%")
        if prof:
            parts.append("Profitability: " + ", ".join(prof) + ".")

        # Growth
        growth = []
        if self.revenue_growth is not None:  growth.append(f"Rev growth={self.revenue_growth*100:+.1f}%")
        if self.earnings_growth is not None: growth.append(f"EPS growth={self.earnings_growth*100:+.1f}%")
        if growth:
            parts.append("Growth (YoY): " + ", ".join(growth) + ".")

        # Balance sheet
        bs = []
        if self.debt_to_equity is not None: bs.append(f"D/E={self.debt_to_equity:.1f}")
        if self.current_ratio is not None:  bs.append(f"Current ratio={self.current_ratio:.1f}")
        if bs:
            parts.append("Balance sheet: " + ", ".join(bs) + ".")

        # FCF
        if self.free_cash_flow is not None:
            fcf = self.free_cash_flow
            if abs(fcf) >= 1e9:
                fcf_str = f"${fcf/1e9:.1f}B"
            else:
                fcf_str = f"${fcf/1e6:.0f}M"
            parts.append(f"FCF: {fcf_str}.")

        # Analyst
        if self.analyst_rating and self.target_price_mean is not None:
            analyst_str = (
                f"Analysts ({self.analyst_count or '?'}): consensus={self.analyst_rating.upper()}, "
                f"mean target=${self.target_price_mean:.2f}"
            )
            if self.target_price_low and self.target_price_high:
                analyst_str += f" (range ${self.target_price_low:.2f}–${self.target_price_high:.2f})"
            analyst_str += "."
            parts.append(analyst_str)
        elif self.analyst_rating:
            parts.append(f"Analyst consensus: {self.analyst_rating.upper()} ({self.analyst_count} analysts).")

        # Dividend (yfinance returns dividendYield already in percentage form, e.g. 1.11 = 1.11%)
        if self.dividend_yield and self.dividend_yield > 0:
            div_str = f"Dividend: {self.dividend_yield:.2f}% yield"
            if self.dividend_rate:
                div_str += f" (${self.dividend_rate:.2f}/yr)"
            div_str += "."
            parts.append(div_str)

        # Beta + 52-week
        misc = []
        if self.beta is not None:
            misc.append(f"Beta={self.beta:.2f}")
        if self.week52_high and self.week52_low:
            misc.append(f"52-wk range ${self.week52_low:.2f}–${self.week52_high:.2f}")
        if misc:
            parts.append(" | ".join(misc) + ".")

        return " ".join(parts)


# ── Builders ───────────────────────────────────────────────────────────────────

def _sf(info: dict, key: str) -> Optional[float]:
    val = info.get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _si(info: dict, key: str) -> Optional[int]:
    val = info.get(key)
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _ss(info: dict, key: str) -> Optional[str]:
    val = info.get(key)
    return str(val) if val is not None else None


def from_raw_info(ticker: str, info: dict) -> FundamentalSnapshot:
    """Build a FundamentalSnapshot from a yfinance .info dict."""
    return FundamentalSnapshot(
        ticker=ticker,
        sector=_ss(info, "sector"),
        industry=_ss(info, "industry"),
        market_cap=_sf(info, "marketCap"),
        enterprise_value=_sf(info, "enterpriseValue"),
        trailing_pe=_sf(info, "trailingPE"),
        forward_pe=_sf(info, "forwardPE"),
        price_to_book=_sf(info, "priceToBook"),
        ev_to_ebitda=_sf(info, "enterpriseToEbitda"),
        price_to_sales=_sf(info, "priceToSalesTrailing12Months"),
        roe=_sf(info, "returnOnEquity"),
        roa=_sf(info, "returnOnAssets"),
        gross_margin=_sf(info, "grossMargins"),
        operating_margin=_sf(info, "operatingMargins"),
        profit_margin=_sf(info, "profitMargins"),
        revenue_growth=_sf(info, "revenueGrowth"),
        earnings_growth=_sf(info, "earningsGrowth"),
        debt_to_equity=_sf(info, "debtToEquity"),
        current_ratio=_sf(info, "currentRatio"),
        free_cash_flow=_sf(info, "freeCashflow"),
        total_cash=_sf(info, "totalCash"),
        total_debt=_sf(info, "totalDebt"),
        trailing_eps=_sf(info, "trailingEps"),
        forward_eps=_sf(info, "forwardEps"),
        analyst_rating=_ss(info, "recommendationKey"),
        analyst_mean_score=_sf(info, "recommendationMean"),
        analyst_count=_si(info, "numberOfAnalystOpinions"),
        target_price_mean=_sf(info, "targetMeanPrice"),
        target_price_low=_sf(info, "targetLowPrice"),
        target_price_high=_sf(info, "targetHighPrice"),
        dividend_yield=_sf(info, "dividendYield"),
        dividend_rate=_sf(info, "dividendRate"),
        payout_ratio=_sf(info, "payoutRatio"),
        beta=_sf(info, "beta"),
        week52_high=_sf(info, "fiftyTwoWeekHigh"),
        week52_low=_sf(info, "fiftyTwoWeekLow"),
    )


def compute(tickers: list[str], raw_info: dict[str, dict]) -> dict[str, FundamentalSnapshot]:
    """
    Convert raw yfinance info dicts → FundamentalSnapshot objects.

    Args:
        tickers:  equity tickers to process (cash symbols already excluded)
        raw_info: output of tools.yfinance_fundamentals.fetch_raw_info()

    Returns:
        { ticker: FundamentalSnapshot }  — only includes tickers with data
    """
    result: dict[str, FundamentalSnapshot] = {}
    for ticker in tickers:
        if ticker in raw_info:
            result[ticker] = from_raw_info(ticker, raw_info[ticker])
    return result
