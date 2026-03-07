"""
InsiderSnapshot — insider trading activity summary for one ticker.

Input:  list of transaction dicts from tools/finnhub_insider.py
        + optional sentiment dict from fetch_insider_sentiment()
Output: InsiderSnapshot with .summary() for LLM context

Gives the LLM awareness of insider buying/selling patterns so it can
factor insider confidence into buy/hold/sell recommendations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InsiderSnapshot:
    ticker: str

    # Aggregated stats (last 6 months)
    total_buys: int = 0
    total_sells: int = 0
    total_buy_value: float = 0.0       # $ total purchases
    total_sell_value: float = 0.0      # $ total sales
    net_shares: int = 0                # net share change (positive = net buying)
    notable_names: list[str] = field(default_factory=list)  # top insider names

    # Recent transactions (up to 5 most recent)
    recent_transactions: list[dict] = field(default_factory=list)

    # Finnhub MSPR (Monthly Share Purchase Ratio)
    # Positive = net buying, Negative = net selling
    mspr: Optional[float] = None

    # Derived signal
    signal: str = "neutral"  # "bullish", "bearish", "neutral", "mixed"

    def summary(self) -> str:
        """Text summary suitable for LLM context."""
        if self.total_buys == 0 and self.total_sells == 0:
            return f"{self.ticker}: no insider transactions in last 6 months."

        parts = [f"{self.ticker} insider activity (6mo):"]

        # Buy/sell counts
        parts.append(f"{self.total_buys} buys, {self.total_sells} sells.")

        # Values
        val_parts = []
        if self.total_buy_value > 0:
            val_parts.append(f"bought ${_fmt_value(self.total_buy_value)}")
        if self.total_sell_value > 0:
            val_parts.append(f"sold ${_fmt_value(self.total_sell_value)}")
        if val_parts:
            parts.append("Value: " + ", ".join(val_parts) + ".")

        # Net shares
        if self.net_shares != 0:
            direction = "net buying" if self.net_shares > 0 else "net selling"
            parts.append(f"Net: {abs(self.net_shares):,} shares ({direction}).")

        # MSPR
        if self.mspr is not None and self.mspr != 0:
            mspr_label = "positive (buying)" if self.mspr > 0 else "negative (selling)"
            parts.append(f"MSPR: {self.mspr:.2f} ({mspr_label}).")

        # Notable names
        if self.notable_names:
            parts.append(f"Key insiders: {', '.join(self.notable_names[:3])}.")

        # Signal
        parts.append(f"Signal: {self.signal.upper()}.")

        # Recent notable transactions
        if self.recent_transactions:
            txn = self.recent_transactions[0]
            txn_type = "bought" if txn.get("change", 0) > 0 else "sold"
            name = txn.get("name", "Unknown")
            val = txn.get("value")
            val_str = f" (${_fmt_value(val)})" if val else ""
            date = txn.get("filing_date", "")
            parts.append(f"Latest: {name} {txn_type} {abs(txn.get('change', 0)):,} shares{val_str} on {date}.")

        return " ".join(parts)


def _fmt_value(v: float) -> str:
    """Format dollar value for display."""
    if v >= 1e9:
        return f"{v / 1e9:.1f}B"
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:,.0f}"


def _classify_signal(buys: int, sells: int, net_shares: int, mspr: float | None) -> str:
    """Classify insider activity as bullish/bearish/neutral/mixed."""
    if buys == 0 and sells == 0:
        return "neutral"

    # MSPR is a strong signal if available
    if mspr is not None:
        if mspr > 5:
            return "bullish"
        if mspr < -5:
            return "bearish"

    # Ratio-based
    total = buys + sells
    if total == 0:
        return "neutral"

    buy_ratio = buys / total

    if buy_ratio >= 0.7 and net_shares > 0:
        return "bullish"
    if buy_ratio <= 0.3 and net_shares < 0:
        return "bearish"
    if buys > 0 and sells > 0:
        return "mixed"
    if buys > 0:
        return "bullish"
    if sells > 0:
        return "bearish"

    return "neutral"


def compute(
    ticker: str,
    transactions: list[dict],
    sentiment: dict | None = None,
) -> InsiderSnapshot:
    """Build an InsiderSnapshot from transaction list + optional sentiment."""
    buys = 0
    sells = 0
    buy_value = 0.0
    sell_value = 0.0
    net_shares = 0
    names: dict[str, float] = {}  # name → total abs value (for ranking)

    for txn in transactions:
        change = txn.get("change", 0)
        value = txn.get("value") or 0
        name = txn.get("name", "")

        if change > 0:
            buys += 1
            buy_value += value
        elif change < 0:
            sells += 1
            sell_value += value

        net_shares += change

        if name:
            names[name] = names.get(name, 0) + abs(value)

    # Rank insiders by total transaction value
    notable = sorted(names.keys(), key=lambda n: names[n], reverse=True)

    mspr = sentiment.get("mspr") if sentiment else None

    signal = _classify_signal(buys, sells, net_shares, mspr)

    return InsiderSnapshot(
        ticker=ticker,
        total_buys=buys,
        total_sells=sells,
        total_buy_value=buy_value,
        total_sell_value=sell_value,
        net_shares=net_shares,
        notable_names=notable[:5],
        recent_transactions=transactions[:5],
        mspr=mspr,
        signal=signal,
    )


def compute_all(
    tickers: list[str],
    raw_transactions: dict[str, list[dict]],
    raw_sentiment: dict[str, dict] | None = None,
) -> dict[str, InsiderSnapshot]:
    """
    Compute InsiderSnapshots for all tickers.

    Args:
        tickers: all equity tickers (ensures every ticker gets a snapshot)
        raw_transactions: output of tools.finnhub_insider.fetch_insider_transactions()
        raw_sentiment: output of tools.finnhub_insider.fetch_insider_sentiment()

    Returns:
        { ticker: InsiderSnapshot }
    """
    sentiment = raw_sentiment or {}
    return {
        ticker: compute(
            ticker,
            raw_transactions.get(ticker, []),
            sentiment.get(ticker),
        )
        for ticker in tickers
    }
