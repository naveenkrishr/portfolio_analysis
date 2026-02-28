"""
NewsSnapshot — recent headlines + keyword-based sentiment for one ticker.

Input:  list of article dicts from tools/yfinance_news.py
Output: NewsSnapshot with .summary() for LLM context

Sentiment approach: keyword matching on headline tokens.
  - No external API or ML model required.
  - Each positive keyword contributes +0.25 to headline score (clamped ±1).
  - Each negative keyword contributes -0.25.
  - NewsSnapshot.sentiment_score = mean across all headlines.
  - Label: bullish ≥ +0.1  |  bearish ≤ -0.1  |  neutral otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Keyword lists ──────────────────────────────────────────────────────────────

_POSITIVE_WORDS = {
    # Price / earnings action
    "surge", "surges", "surging", "soar", "soars", "soaring",
    "jump", "jumps", "jumping", "rise", "rises", "rising", "rally", "rallies",
    "gain", "gains", "high", "highs", "record", "records",
    # Earnings / fundamentals
    "beat", "beats", "beating", "exceeded", "exceeds", "exceed",
    "outperform", "outperforms", "strong", "strength", "profit",
    "revenue", "growth", "grew", "expand", "expands", "expansion",
    # Analyst / rating
    "upgrade", "upgrades", "upgraded", "buy", "overweight",
    "bullish", "positive", "optimistic", "upside",
    # Corporate actions
    "approved", "approves", "breakthrough", "partnership", "deal", "wins",
    "awarded", "contract", "dividend", "buyback",
}

_NEGATIVE_WORDS = {
    # Price action
    "drop", "drops", "dropping", "fall", "falls", "falling",
    "plunge", "plunges", "plunging", "crash", "crashes", "crashing",
    "decline", "declines", "declining", "slump", "slumps", "low", "lows",
    # Earnings / fundamentals
    "miss", "misses", "missed", "disappoints", "disappointing",
    "weak", "weakness", "loss", "losses", "deficit",
    "cut", "cuts", "cutting", "reduce", "reduces",
    # Analyst / rating
    "downgrade", "downgrades", "downgraded", "sell", "underperform",
    "underweight", "bearish", "negative", "pessimistic", "downside",
    # Legal / risk
    "lawsuit", "lawsuits", "investigation", "probe", "fine", "fined",
    "fraud", "recall", "warning", "concern", "concerns", "risk", "risks",
    "delay", "delays", "layoff", "layoffs",
}


# ── Sentiment helpers ──────────────────────────────────────────────────────────

def _score_headline(title: str) -> float:
    """
    Keyword-based sentiment score for a single headline.
    Returns a value in [-1.0, +1.0].
    """
    tokens = set(title.lower().replace(",", " ").replace(".", " ").split())
    score = 0.0
    for w in tokens:
        if w in _POSITIVE_WORDS:
            score += 0.25
        elif w in _NEGATIVE_WORDS:
            score -= 0.25
    return max(-1.0, min(1.0, score))


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class NewsSnapshot:
    ticker:          str
    headline_count:  int
    articles:        list[dict]   # {"title", "publisher", "age_hours", "url"}
    sentiment_score: float        # mean headline score, range [-1.0, +1.0]
    sentiment_label: str          # "bullish" | "neutral" | "bearish"

    def summary(self) -> str:
        """Multi-line text summary suitable for LLM context."""
        if self.headline_count == 0:
            return f"{self.ticker} news: no recent headlines found."

        score_str = f"{self.sentiment_score:+.2f}"
        lines = [
            f"{self.ticker} news ({self.headline_count} headlines, "
            f"sentiment: {self.sentiment_label.upper()} {score_str}):"
        ]
        for a in self.articles[:6]:   # show up to 6 headlines in LLM context
            age = a["age_hours"]
            age_str = f"{age:.0f}h" if age < 48 else f"{age/24:.0f}d"
            lines.append(f"  [{age_str}] \"{a['title']}\" ({a['publisher']})")
        return "\n".join(lines)


# ── Builders ───────────────────────────────────────────────────────────────────

def compute(ticker: str, articles: list[dict]) -> NewsSnapshot:
    """Build a NewsSnapshot from a list of article dicts."""
    if not articles:
        return NewsSnapshot(
            ticker=ticker,
            headline_count=0,
            articles=[],
            sentiment_score=0.0,
            sentiment_label="neutral",
        )

    scores = [_score_headline(a["title"]) for a in articles]
    avg = sum(scores) / len(scores)

    if avg >= 0.1:
        label = "bullish"
    elif avg <= -0.1:
        label = "bearish"
    else:
        label = "neutral"

    return NewsSnapshot(
        ticker=ticker,
        headline_count=len(articles),
        articles=articles,
        sentiment_score=round(avg, 3),
        sentiment_label=label,
    )


def compute_all(news_raw: dict[str, list[dict]]) -> dict[str, NewsSnapshot]:
    """
    Compute NewsSnapshots for all tickers.

    Args:
        news_raw: output of tools.yfinance_news.fetch_news()

    Returns:
        { ticker: NewsSnapshot }
    """
    return {ticker: compute(ticker, articles) for ticker, articles in news_raw.items()}
