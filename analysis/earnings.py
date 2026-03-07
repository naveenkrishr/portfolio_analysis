"""
EarningsSnapshot — upcoming earnings dates + estimates for one ticker.

Input:  list of event dicts from tools/finnhub_earnings.py
Output: EarningsSnapshot with .summary() for LLM context

Designed to give the LLM awareness of upcoming catalysts so it can
factor earnings timing into buy/hold/sell recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class EarningsSnapshot:
    ticker: str
    next_earnings_date: str | None       # "2026-04-29" or None
    next_earnings_hour: str | None       # "bmo", "amc", or ""
    quarter: int | None                  # fiscal quarter (1-4)
    year: int | None                     # fiscal year
    eps_estimate: float | None           # consensus EPS estimate
    revenue_estimate: float | None       # consensus revenue estimate
    days_until: int | None               # days until next earnings
    past_events: list[dict] = field(default_factory=list)  # events with actuals

    def summary(self) -> str:
        """Text summary suitable for LLM context."""
        if not self.next_earnings_date:
            return f"{self.ticker}: no upcoming earnings date found."

        hour_str = ""
        if self.next_earnings_hour == "bmo":
            hour_str = " (before market open)"
        elif self.next_earnings_hour == "amc":
            hour_str = " (after market close)"

        parts = [
            f"{self.ticker} next earnings: {self.next_earnings_date}{hour_str}"
        ]

        if self.quarter and self.year:
            parts[0] += f", Q{self.quarter} {self.year}"

        if self.days_until is not None:
            parts.append(f"({self.days_until} days away)")

        if self.eps_estimate is not None:
            parts.append(f"EPS est: ${self.eps_estimate:.2f}")

        if self.revenue_estimate is not None:
            if self.revenue_estimate >= 1e9:
                parts.append(f"Rev est: ${self.revenue_estimate/1e9:.1f}B")
            elif self.revenue_estimate >= 1e6:
                parts.append(f"Rev est: ${self.revenue_estimate/1e6:.0f}M")

        return " | ".join(parts)


def _days_until(date_str: str) -> int | None:
    """Calculate days from today until a date string (YYYY-MM-DD)."""
    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        return (target - now).days
    except Exception:
        return None


def compute(ticker: str, events: list[dict]) -> EarningsSnapshot:
    """Build an EarningsSnapshot from a list of earnings events."""
    if not events:
        return EarningsSnapshot(
            ticker=ticker,
            next_earnings_date=None,
            next_earnings_hour=None,
            quarter=None,
            year=None,
            eps_estimate=None,
            revenue_estimate=None,
            days_until=None,
        )

    # First event is the soonest (already sorted by date in fetch)
    nxt = events[0]

    return EarningsSnapshot(
        ticker=ticker,
        next_earnings_date=nxt.get("date"),
        next_earnings_hour=nxt.get("hour"),
        quarter=nxt.get("quarter"),
        year=nxt.get("year"),
        eps_estimate=nxt.get("eps_estimate"),
        revenue_estimate=nxt.get("revenue_estimate"),
        days_until=_days_until(nxt.get("date", "")),
        past_events=[e for e in events[1:] if e.get("eps_actual") is not None],
    )


def compute_all(
    tickers: list[str],
    raw_events: dict[str, list[dict]],
) -> dict[str, EarningsSnapshot]:
    """
    Compute EarningsSnapshots for all tickers.

    Args:
        tickers: all equity tickers (ensures every ticker gets a snapshot)
        raw_events: output of tools.finnhub_earnings.fetch_earnings()

    Returns:
        { ticker: EarningsSnapshot }
    """
    return {
        ticker: compute(ticker, raw_events.get(ticker, []))
        for ticker in tickers
    }
