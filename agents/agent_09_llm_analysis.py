"""
Agent 9 — LLM Portfolio Analysis (two-phase)

Phase 1: Per-ticker analysis — one focused LLM call per equity position.
         Each call receives that ticker's data from agents 3-8 and produces
         a structured TickerAnalysis (recommendation, priority, action, assessment).
         Stocks and ETFs get different prompts with relevant questions.

Phase 2: Portfolio synthesis — one streamed LLM call that aggregates all
         per-ticker analyses + portfolio-level risk metrics into the final
         markdown report (same format agent_10 expects).

Data sources (agents 3-8):
  - Agent 3: technicals (SMA, RSI, MACD, Bollinger)
  - Agent 4: fundamentals (P/E, ROE, FCF, analyst ratings) — stocks only
  - Agent 4: ETF data (expense ratio, AUM, holdings, sectors) — ETFs only
  - Agent 5: news & sentiment (recent headlines, bullish/bearish)
  - Agent 6: earnings calendar (upcoming dates, EPS estimates) — stocks only
  - Agent 7: insider trading (Form 4 filings, insider sentiment) — stocks only
  - Agent 8: risk metrics (VaR, beta, Sharpe, correlation, concentration)
"""
from __future__ import annotations

import re
import time
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from state.graph_state import PortfolioState
from state.models import Holding, PortfolioAnalysis, TickerAnalysis

# ── System prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert financial portfolio analyst. You provide clear, actionable \
investment advice based on portfolio data. You are direct and honest — if a \
position has problems, say so. You focus on what the investor should DO, not \
just what the data says.\
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are an expert financial portfolio analyst synthesizing per-ticker analyses \
into a cohesive portfolio report. You have already analyzed each position \
individually — now produce the final report with portfolio-level insights, \
cross-position interactions, and overall strategy. Preserve the per-ticker \
recommendations faithfully. Be direct and actionable.\
"""

# ── Phase 1: Per-ticker prompt & parsing ─────────────────────────────────────

def _build_ticker_prompt(
    holding: Holding,
    total_value: float,
    market_snap=None,
    fundamental_snap=None,
    news_snap=None,
    earnings_snap=None,
    insider_snap=None,
    ticker_risk=None,
) -> str:
    """Build a focused prompt for analyzing a single stock position."""
    today = date.today().strftime("%B %d, %Y")
    pct = holding.value / total_value * 100 if total_value else 0

    # Cost basis info
    cost_info = ""
    if holding.total_cost is not None:
        gain = holding.value - holding.total_cost
        gain_pct = (gain / holding.total_cost * 100) if holding.total_cost else 0
        sign = "+" if gain >= 0 else ""
        cost_info = (
            f"  Cost basis:    ${holding.total_cost:>10,.0f}  (avg ${holding.avg_cost:,.2f}/sh)\n"
            f"  Gain/Loss:     {sign}${gain:>9,.0f}  ({sign}{gain_pct:.1f}%)\n"
        )

    # Gather data sections
    sections = []

    if market_snap:
        sections.append(f"TECHNICAL DATA\n  {market_snap.summary()}")
    if fundamental_snap:
        sections.append(f"FUNDAMENTALS\n  {fundamental_snap.summary()}")
    if news_snap:
        sections.append(f"RECENT NEWS & SENTIMENT\n{news_snap.summary()}")
    if earnings_snap:
        sections.append(f"UPCOMING EARNINGS\n  {earnings_snap.summary()}")
    if insider_snap:
        sections.append(f"INSIDER TRADING\n  {insider_snap.summary()}")
    if ticker_risk:
        sections.append(f"RISK METRICS\n  {ticker_risk.summary()}")

    data_block = "\n\n".join(sections) if sections else "No additional data available."

    return f"""\
Single Position Analysis — {today}

POSITION
  Ticker:      {holding.ticker}
  Name:        {holding.name}
  Type:        {holding.asset_type}
  Shares:      {holding.shares:,.2f}
  Price:       ${holding.price:,.2f}
  Value:       ${holding.value:,.0f}
  Allocation:  {pct:.1f}% of ${total_value:,.0f} portfolio
{cost_info}
{data_block}

Analyze this position considering:
- Is the allocation ({pct:.1f}%) appropriate for this type of holding?
- What do the technicals suggest about timing?
- Are fundamentals attractive at current valuation?
- Any news catalysts or risks to watch?
- Upcoming earnings impact?
- What are insiders signaling?
- Risk/reward profile?

Respond in EXACTLY this format (no extra text):

**Recommendation:** BUY / ADD / HOLD / REDUCE / SELL
**Priority:** CRITICAL / HIGH / MEDIUM / LOW
**Action:** one-line action item (be specific, e.g. "Reduce by 20% — overweight at 15%")
**Role in portfolio:** what this position does for the portfolio
**Assessment:** 2-3 sentences on quality, risks, and outlook
**Key risks:** risk1, risk2, risk3 (comma-separated, max 3)
"""


def _build_etf_prompt(
    holding: Holding,
    total_value: float,
    market_snap=None,
    etf_snap=None,
    news_snap=None,
    ticker_risk=None,
    all_holdings: list[Holding] | None = None,
) -> str:
    """Build a focused prompt for analyzing a single ETF position."""
    today = date.today().strftime("%B %d, %Y")
    pct = holding.value / total_value * 100 if total_value else 0

    # Cost basis info
    cost_info = ""
    if holding.total_cost is not None:
        gain = holding.value - holding.total_cost
        gain_pct = (gain / holding.total_cost * 100) if holding.total_cost else 0
        sign = "+" if gain >= 0 else ""
        cost_info = (
            f"  Cost basis:    ${holding.total_cost:>10,.0f}  (avg ${holding.avg_cost:,.2f}/sh)\n"
            f"  Gain/Loss:     {sign}${gain:>9,.0f}  ({sign}{gain_pct:.1f}%)\n"
        )

    # Gather data sections
    sections = []

    if market_snap:
        sections.append(f"TECHNICAL DATA\n  {market_snap.summary()}")
    if etf_snap:
        sections.append(f"ETF FUNDAMENTALS\n  {etf_snap.summary()}")
    if news_snap:
        sections.append(f"RECENT NEWS & SENTIMENT\n{news_snap.summary()}")
    if ticker_risk:
        sections.append(f"RISK METRICS\n  {ticker_risk.summary()}")

    # Overlap detection: check if individual stocks in portfolio overlap with ETF holdings
    overlap_section = ""
    if etf_snap and etf_snap.top_holdings and all_holdings:
        stock_tickers = {h.ticker for h in all_holdings if h.asset_type == "stock"}
        etf_holding_tickers = set(etf_snap.top_holdings)
        overlap = stock_tickers & etf_holding_tickers
        if overlap:
            overlap_section = f"\nHOLDINGS OVERLAP\n  You also hold these stocks individually: {', '.join(sorted(overlap))}\n  This creates duplicate exposure."

    data_block = "\n\n".join(sections) if sections else "No additional data available."

    return f"""\
ETF Position Analysis — {today}

POSITION
  Ticker:      {holding.ticker}
  Name:        {holding.name}
  Type:        ETF
  Shares:      {holding.shares:,.2f}
  Price:       ${holding.price:,.2f}
  Value:       ${holding.value:,.0f}
  Allocation:  {pct:.1f}% of ${total_value:,.0f} portfolio
{cost_info}
{data_block}
{overlap_section}

Analyze this ETF position considering:
- Is the expense ratio competitive for this category?
- Is the fund size (AUM) adequate? (>$1B = good liquidity, lower closure risk)
- Is the allocation ({pct:.1f}%) appropriate as a core or satellite holding?
- What do the technicals suggest about entry/exit timing?
- Any holdings overlap with other positions in the portfolio?
- Dividend/distribution yield vs alternatives?
- Risk metrics (beta, volatility)?
- Category fit — does this ETF serve the right role in the portfolio?

Respond in EXACTLY this format (no extra text):

**Recommendation:** BUY / ADD / HOLD / REDUCE / SELL
**Priority:** CRITICAL / HIGH / MEDIUM / LOW
**Action:** one-line action item (be specific, e.g. "Core holding — maintain current allocation")
**Role in portfolio:** what this ETF does for the portfolio (e.g. "Core large-cap exposure")
**Assessment:** 2-3 sentences on fund quality, cost efficiency, and fit
**Key risks:** risk1, risk2, risk3 (comma-separated, max 3)
"""


def _parse_ticker_response(ticker: str, raw: str) -> TickerAnalysis:
    """Parse the structured LLM output into a TickerAnalysis model."""
    def _extract(label: str, default: str = "") -> str:
        m = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+?)(?=\n\*\*|\Z)", raw, re.DOTALL)
        return m.group(1).strip() if m else default

    rec_raw = _extract("Recommendation", "HOLD").upper().split()[0]
    valid_recs = {"BUY", "ADD", "HOLD", "REDUCE", "SELL"}
    recommendation = rec_raw if rec_raw in valid_recs else "HOLD"

    pri_raw = _extract("Priority", "MEDIUM").upper().split()[0]
    valid_pris = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    priority = pri_raw if pri_raw in valid_pris else "MEDIUM"

    action = _extract("Action", "Review position")
    role = _extract("Role in portfolio", "")
    assessment = _extract("Assessment", "")
    risks = _extract("Key risks", "")

    # Combine role + assessment + risks into the full analysis paragraph
    parts = []
    if role:
        parts.append(f"Role: {role}")
    if assessment:
        parts.append(assessment)
    if risks:
        parts.append(f"Key risks: {risks}")
    analysis_text = " ".join(parts) if parts else raw[:500]

    return TickerAnalysis(
        ticker=ticker,
        recommendation=recommendation,
        priority=priority,
        action_summary=action,
        analysis=analysis_text,
    )


# ── Phase 2: Synthesis prompt ────────────────────────────────────────────────

def _build_synthesis_prompt(
    holdings: list[Holding],
    total_value: float,
    ticker_analyses: list[TickerAnalysis],
    risk_data=None,
    data_warnings: list[str] | None = None,
    etf_data: dict | None = None,
) -> str:
    """Build the synthesis prompt from per-ticker analyses + portfolio data."""
    today = date.today().strftime("%B %d, %Y")

    invested   = sum(h.value for h in holdings if h.asset_type != "cash")
    cash_total = sum(h.value for h in holdings if h.asset_type == "cash")
    etf_value  = sum(h.value for h in holdings if h.asset_type == "etf")
    stock_value = sum(h.value for h in holdings if h.asset_type == "stock")

    # Holdings table
    rows = []
    for h in holdings:
        pct = h.value / total_value * 100
        cost_str = f"${h.total_cost:>10,.0f}" if h.total_cost is not None else f"{'N/A':>11}"
        if h.total_cost is not None:
            gain = h.value - h.total_cost
            gain_pct = (gain / h.total_cost * 100) if h.total_cost else 0
            sign = "+" if gain >= 0 else ""
            gl_str = f"{sign}${gain:>9,.0f} ({sign}{gain_pct:.1f}%)"
        else:
            gl_str = f"{'N/A':>20}"
        rows.append(
            f"  {h.ticker:<8} {h.name:<42} "
            f"{h.shares:>10,.2f} sh  "
            f"${h.price:>8,.2f}  "
            f"${h.value:>10,.0f}  "
            f"{cost_str}  "
            f"{gl_str}  "
            f"{pct:>5.1f}%  "
            f"[{h.asset_type}]"
        )
    holdings_table = "\n".join(rows)

    # Data warnings
    warnings_section = ""
    if data_warnings:
        lines = "\n".join(f"  - {w}" for w in data_warnings)
        warnings_section = (
            "\nDATA FRESHNESS NOTICE (some data may be from cache)\n"
            + lines
            + "\nAccount for potential staleness in your analysis.\n"
        )

    # Per-ticker analysis summaries
    ticker_lines = []
    for ta in ticker_analyses:
        ticker_lines.append(
            f"  {ta.ticker}: {ta.recommendation} ({ta.priority}) — {ta.action_summary}\n"
            f"    {ta.analysis}"
        )
    ticker_block = "\n\n".join(ticker_lines)

    # Portfolio risk
    risk_section = ""
    if risk_data is not None:
        risk_section = (
            "\nQUANTITATIVE RISK ANALYSIS\n"
            + risk_data.summary()
            + "\n"
        )

    # ETF overlap analysis
    overlap_section = ""
    if etf_data:
        stock_tickers = {h.ticker for h in holdings if h.asset_type == "stock"}
        overlaps = []
        for etf_ticker, snap in etf_data.items():
            if hasattr(snap, "top_holdings") and snap.top_holdings:
                etf_holdings_set = set(snap.top_holdings)
                common = stock_tickers & etf_holdings_set
                if common:
                    overlaps.append(f"  {etf_ticker} shares holdings with: {', '.join(sorted(common))}")
        if overlaps:
            overlap_section = "\nHOLDINGS OVERLAP (ETFs vs individual stocks)\n" + "\n".join(overlaps) + "\n"

    # Asset type breakdown
    num_stocks = len([h for h in holdings if h.asset_type == "stock"])
    num_etfs = len([h for h in holdings if h.asset_type == "etf"])
    num_cash = len([h for h in holdings if h.asset_type == "cash"])

    return f"""\
Portfolio Synthesis Report — {today}
{warnings_section}
PORTFOLIO OVERVIEW
  Total Value:   ${total_value:>12,.0f}
  Invested:      ${invested:>12,.0f}  ({invested/total_value*100:.1f}%)
  Cash & MM:     ${cash_total:>12,.0f}  ({cash_total/total_value*100:.1f}%)
  Stocks:        ${stock_value:>12,.0f}  ({stock_value/total_value*100:.1f}%)
  ETFs:          ${etf_value:>12,.0f}  ({etf_value/total_value*100:.1f}%)
  Positions:     {num_stocks} stocks  +  {num_etfs} ETFs  +  {num_cash} cash

HOLDINGS
  {"Ticker":<8} {"Name":<42} {"Shares":>14}  {"Price":>10}  {"Value":>12}  {"Cost":>11}  {"Gain/Loss":>20}  {"Alloc":>6}  Type
  {"-"*160}
{holdings_table}

PER-TICKER ANALYSES (completed individually — incorporate these faithfully)
{ticker_block}
{risk_section}{overlap_section}
Using the per-ticker analyses above and the portfolio-level risk metrics, \
produce the final portfolio report in EXACTLY this format:

## RECOMMENDED ACTIONS
List each action on its own line as: [PRIORITY] TICKER — action description
Priority levels: CRITICAL (act immediately) | HIGH (act within 2 weeks) | MEDIUM (act within a month) | LOW (informational)
Sort by priority (CRITICAL first). Use the per-ticker recommendations above.

## EXECUTIVE SUMMARY
2-3 sentences. Overall portfolio health, biggest strength, biggest concern. \
Consider how stocks and ETFs work together, any holdings overlap, and overall \
diversification between individual stocks and index/ETF exposure.

## PER-TICKER ANALYSIS

### TICKER — Name
**Recommendation:** (from per-ticker analysis)
**Role in portfolio:** (from per-ticker analysis)
**Assessment:** (from per-ticker analysis — expand with portfolio context)
**Key risks:** (from per-ticker analysis)

(repeat for each non-cash position — both stocks and ETFs)

## CASH POSITION
Comment on the ${cash_total:,.0f} in money market / cash ({cash_total/total_value*100:.1f}% of portfolio). \
Is it too high, appropriate, or should it be deployed?

## PORTFOLIO RISK ASSESSMENT
- Concentration risk:
- Diversification (stocks vs ETFs vs cash):
- Holdings overlap (ETFs containing individually held stocks):
- Correlation risk:
- Overall rating: (CONSERVATIVE / BALANCED / AGGRESSIVE / OVER-CONCENTRATED)
"""


# ── Agent node ────────────────────────────────────────────────────────────────

def run(state: PortfolioState, llm) -> PortfolioState:
    """
    LangGraph node — two-phase LLM analysis.

    Phase 1: Per-ticker calls (sequential, non-streaming) → TickerAnalysis list
             Stocks get stock-specific prompts; ETFs get ETF-specific prompts.
    Phase 2: Synthesis call (streaming) → final markdown report
    """
    holdings: list[Holding] = state["holdings"]
    total_value: float      = state["total_value"]
    market_data             = state.get("market_data") or {}
    fundamentals            = state.get("fundamentals") or {}
    etf_data                = state.get("etf_data") or {}
    news_data               = state.get("news_data") or {}
    earnings_data           = state.get("earnings_data") or {}
    insider_data            = state.get("insider_data") or {}
    risk_data               = state.get("risk_data")
    data_warnings           = state.get("data_warnings") or []

    equity_holdings = [h for h in holdings if h.asset_type != "cash"]

    # ── Phase 1: Per-ticker analysis ──────────────────────────────────────────

    print("\n" + "=" * 70)
    print(f"AGENT 9 — Per-Ticker Analysis ({len(equity_holdings)} positions)")
    print("=" * 70)

    ticker_analyses: list[TickerAnalysis] = []
    phase1_t0 = time.time()

    for i, h in enumerate(equity_holdings):
        t0 = time.time()
        print(f"\n  [{i+1}/{len(equity_holdings)}] {h.ticker} ({h.asset_type})...", end="", flush=True)

        # Gather per-ticker risk data
        ticker_risk = None
        if risk_data is not None and hasattr(risk_data, "ticker_risks"):
            ticker_risk = risk_data.ticker_risks.get(h.ticker)

        # Build prompt based on asset type
        if h.asset_type == "etf":
            prompt = _build_etf_prompt(
                holding=h,
                total_value=total_value,
                market_snap=market_data.get(h.ticker),
                etf_snap=etf_data.get(h.ticker),
                news_snap=news_data.get(h.ticker),
                ticker_risk=ticker_risk,
                all_holdings=holdings,
            )
        else:
            prompt = _build_ticker_prompt(
                holding=h,
                total_value=total_value,
                market_snap=market_data.get(h.ticker),
                fundamental_snap=fundamentals.get(h.ticker),
                news_snap=news_data.get(h.ticker),
                earnings_snap=earnings_data.get(h.ticker),
                insider_snap=insider_data.get(h.ticker),
                ticker_risk=ticker_risk,
            )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            result = llm.invoke(messages)
            raw_text = result.content
            ta = _parse_ticker_response(h.ticker, raw_text)
        except Exception as e:
            print(f" [ERROR: {e}]", end="")
            ta = TickerAnalysis(
                ticker=h.ticker,
                recommendation="HOLD",
                priority="LOW",
                action_summary="Analysis unavailable — LLM error",
                analysis=f"LLM analysis failed for {h.ticker}: {str(e)[:100]}",
            )

        ticker_analyses.append(ta)
        elapsed = time.time() - t0
        print(f" {ta.recommendation} ({ta.priority}) — {ta.action_summary}  ({elapsed:.1f}s)")

    phase1_elapsed = time.time() - phase1_t0
    print(f"\n  Per-ticker phase: {phase1_elapsed:.1f}s total")

    # ── Phase 2: Portfolio synthesis (streamed) ───────────────────────────────

    print("\n" + "=" * 70)
    print("AGENT 9 — Portfolio Synthesis")
    print("=" * 70 + "\n")

    synthesis_prompt = _build_synthesis_prompt(
        holdings=holdings,
        total_value=total_value,
        ticker_analyses=ticker_analyses,
        risk_data=risk_data,
        data_warnings=data_warnings,
        etf_data=etf_data,
    )

    messages = [
        SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt),
    ]

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
    print(f"Total Agent 9: {time.time() - phase1_t0:.1f}s")
    print("─" * 70)

    analysis = PortfolioAnalysis(
        executive_summary="",
        recommended_actions=[],
        ticker_analyses=ticker_analyses,
        risk_summary="",
        raw_llm_output=full_text,
    )

    return {**state, "analysis": analysis}
