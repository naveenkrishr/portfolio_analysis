"""
Agent 9 — LLM Portfolio Analysis

Uses the local Qwen2.5-14B model (via MLXChatModel) to analyze the portfolio
and produce prioritized recommendations + per-ticker analysis.

Step 4: enriched with live technical data from Agent 3 (SMA, RSI, MACD, Bollinger).
Step 5: enriched with fundamentals from Agent 4 (P/E, ROE, FCF, analyst ratings).
Step 6: enriched with news & sentiment from Agent 5 (recent headlines, bullish/bearish).
        Falls back gracefully if any data source is absent.
        Data warnings (cache fallbacks, partial failures) shown at prompt top.
"""
from __future__ import annotations

import time
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from state.graph_state import PortfolioState
from state.models import Holding, PortfolioAnalysis

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert financial portfolio analyst. You provide clear, actionable \
investment advice based on portfolio data. You are direct and honest — if a \
position has problems, say so. You focus on what the investor should DO, not \
just what the data says.\
"""

# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    holdings: list[Holding],
    total_value: float,
    market_data: dict | None = None,
    fundamentals: dict | None = None,
    news_data: dict | None = None,
    data_warnings: list[str] | None = None,
) -> str:
    today = date.today().strftime("%B %d, %Y")

    # Build allocation table
    invested   = sum(h.value for h in holdings if h.asset_type != "cash")
    cash_total = sum(h.value for h in holdings if h.asset_type == "cash")

    rows = []
    for h in holdings:
        pct = h.value / total_value * 100
        rows.append(
            f"  {h.ticker:<8} {h.name:<42} "
            f"{h.shares:>10,.2f} sh  "
            f"${h.price:>8,.2f}  "
            f"${h.value:>10,.0f}  "
            f"{pct:>5.1f}%  "
            f"[{h.account}]"
        )
    holdings_table = "\n".join(rows)

    # Data freshness warnings (cache fallbacks, partial failures)
    warnings_section = ""
    if data_warnings:
        lines = "\n".join(f"  - {w}" for w in data_warnings)
        warnings_section = (
            "\nDATA FRESHNESS NOTICE (some data may be from cache)\n"
            + lines
            + "\nAccount for potential staleness in your analysis.\n"
        )

    # MARKET DATA section from Agent 3 technicals
    market_section = ""
    if market_data:
        lines = []
        for ticker, snap in market_data.items():
            lines.append(f"  {snap.summary()}")
        market_section = (
            "\nMARKET DATA (live technicals — SMA, RSI, MACD, Bollinger)\n"
            + "\n".join(lines)
            + "\n"
        )

    # FUNDAMENTALS section from Agent 4
    fundamentals_section = ""
    if fundamentals:
        lines = []
        for ticker, snap in fundamentals.items():
            lines.append(f"  {snap.summary()}")
        fundamentals_section = (
            "\nFUNDAMENTALS (P/E, ROE, FCF, analyst ratings — use to assess valuation)\n"
            + "\n".join(lines)
            + "\n"
        )

    # NEWS section from Agent 5
    news_section = ""
    if news_data:
        lines = []
        for ticker, snap in news_data.items():
            lines.append(snap.summary())
        news_section = (
            "\nRECENT NEWS & SENTIMENT (last 7 days — factor into your recommendations)\n"
            + "\n".join(lines)
            + "\n"
        )

    return f"""\
Portfolio Analysis Request — {today}
{warnings_section}
PORTFOLIO OVERVIEW
  Total Value:   ${total_value:>12,.0f}
  Invested:      ${invested:>12,.0f}  ({invested/total_value*100:.1f}%)
  Cash & MM:     ${cash_total:>12,.0f}  ({cash_total/total_value*100:.1f}%)
  Positions:     {len([h for h in holdings if h.asset_type != 'cash'])} equity  +  {len([h for h in holdings if h.asset_type == 'cash'])} cash

HOLDINGS
  {"Ticker":<8} {"Name":<42} {"Shares":>14}  {"Price":>10}  {"Value":>12}  {"Alloc":>6}  Account
  {"-"*120}
{holdings_table}
{market_section}{fundamentals_section}{news_section}
Please analyze this portfolio and respond in EXACTLY this format:

## RECOMMENDED ACTIONS
List each action on its own line as: [PRIORITY] TICKER — action description
Priority levels: CRITICAL (act immediately) | HIGH (act within 2 weeks) | MEDIUM (act within a month) | LOW (informational)
Sort by priority (CRITICAL first). Be specific — name the action, not just "review this position".

## EXECUTIVE SUMMARY
2-3 sentences. Overall portfolio health, biggest strength, biggest concern.

## PER-TICKER ANALYSIS

### VOO — Vanguard S&P 500 ETF
**Recommendation:** HOLD / ADD / REDUCE / BUY / SELL
**Role in portfolio:** (what this position does for the portfolio)
**Assessment:** (2-3 sentences: quality of this holding, risks, outlook)
**Key risks:** (bullet list, max 3)

(repeat for each non-cash position)

## CASH POSITION
Comment on the ${cash_total:,.0f} in money market / cash ({cash_total/total_value*100:.1f}% of portfolio). \
Is it too high, appropriate, or should it be deployed?

## PORTFOLIO RISK ASSESSMENT
- Concentration risk:
- Diversification:
- Correlation risk:
- Overall rating: (CONSERVATIVE / BALANCED / AGGRESSIVE / OVER-CONCENTRATED)
"""

# ── Agent node ────────────────────────────────────────────────────────────────

def run(state: PortfolioState, llm) -> PortfolioState:
    """
    LangGraph node — runs LLM analysis over the holdings in state.
    llm is injected from main.py (MLXChatModel singleton).
    """
    holdings: list[Holding] = state["holdings"]
    total_value: float      = state["total_value"]
    market_data             = state.get("market_data")
    fundamentals            = state.get("fundamentals")
    news_data               = state.get("news_data")
    data_warnings           = state.get("data_warnings") or []

    prompt = _build_prompt(
        holdings, total_value,
        market_data=market_data,
        fundamentals=fundamentals,
        news_data=news_data,
        data_warnings=data_warnings,
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    print("\n" + "="*70)
    print("AGENT 9 — LLM Portfolio Analysis")
    print("="*70 + "\n")

    t0 = time.time()
    full_text = ""

    for chunk in llm.stream(messages):
        token = chunk.content
        print(token, end="", flush=True)
        full_text += token

    elapsed = time.time() - t0
    words = len(full_text.split())
    print(f"\n\n{'─'*70}")
    print(f"[{elapsed:.1f}s | ~{words/elapsed:.0f} words/s | {words} words]")
    print("─"*70)

    analysis = PortfolioAnalysis(
        executive_summary="",        # Step 2: parse from raw_llm_output
        recommended_actions=[],      # Step 2: parse from raw_llm_output
        ticker_analyses=[],          # Step 2: parse from raw_llm_output
        risk_summary="",             # Step 2: parse from raw_llm_output
        raw_llm_output=full_text,    # preserved for now
    )

    return {**state, "analysis": analysis}
