"""
Pipeline runner for Render.com deployment (Groq LLM).

Same structure as web_runner.py but uses ChatGroq instead of MLXChatModel.
Wraps the existing LangGraph pipeline (agents 3-9) with progress callbacks,
then generates the HTML report via agent_10._build_html() without sending email.
"""
from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Iterator, List, Optional

import yfinance as yf
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langgraph.graph import END, START, StateGraph

from agents import (
    agent_03_market_data,
    agent_04_fundamental,
    agent_05_news_sentiment,
    agent_06_earnings_events,
    agent_07_insider_institutional,
    agent_08_risk_analysis,
    agent_09_llm_analysis,
)
from state.graph_state import PortfolioState
from state.models import Holding

# ── Asset classification (same as spreadsheet_ingestion) ─────────────────────

_CASH_TICKERS = {"SPAXX", "FCASH", "CORE", "FDRXX", "FZFXX", "FDIC", "VMFXX"}
_KNOWN_ETFS = {"VOO", "QQQ", "QQMG", "SPY", "IVV", "VTI", "VEA", "VWO",
               "BND", "SCHB", "SCHD", "JEPI", "JEPQ", "QQQM"}


def _classify(ticker: str) -> str:
    if ticker in _CASH_TICKERS:
        return "cash"
    if ticker in _KNOWN_ETFS:
        return "etf"
    return "stock"


# ── Build holdings from form data ────────────────────────────────────────────

def build_holdings(
    rows: list[dict],
    progress_callback: Callable[[str], None],
) -> list[Holding]:
    """
    Convert form rows [{ticker, quantity, avg_price}, ...] into Holding objects.
    Consolidates duplicate tickers, fetches current prices via yfinance.
    """
    consolidated: dict[str, dict] = defaultdict(lambda: {"quantity": 0.0, "total_cost": 0.0})
    for row in rows:
        ticker = row["ticker"].strip().upper()
        qty = float(row["quantity"])
        avg = float(row["avg_price"])
        consolidated[ticker]["quantity"] += qty
        consolidated[ticker]["total_cost"] += avg * qty

    all_tickers = sorted(consolidated.keys())
    non_cash = [t for t in all_tickers if t not in _CASH_TICKERS]

    progress_callback(f"Fetching prices for {len(non_cash)} tickers...")
    prices: dict[str, tuple[float, str]] = {}
    if non_cash:
        objs = yf.Tickers(" ".join(non_cash))
        for ticker in non_cash:
            try:
                info = objs.tickers[ticker].info
                price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
                name = info.get("shortName", ticker)[:50]
                prices[ticker] = (float(price), name)
            except Exception:
                prices[ticker] = (0.0, ticker)

    holdings: list[Holding] = []
    for ticker, data in consolidated.items():
        shares = data["quantity"]
        total_cost = data["total_cost"]
        avg_cost = total_cost / shares if shares else 0.0

        price_info = prices.get(ticker)
        if price_info:
            price, name = price_info
        else:
            price = avg_cost
            name = ticker

        if price == 0.0 and avg_cost > 0:
            price = avg_cost

        holdings.append(Holding(
            ticker=ticker,
            name=name,
            shares=shares,
            price=price,
            value=shares * price,
            account="Manual",
            asset_type=_classify(ticker),
            avg_cost=avg_cost,
            total_cost=total_cost,
        ))

    holdings.sort(key=lambda h: (h.asset_type == "cash", -h.value))
    return holdings


# ── Agent labels ─────────────────────────────────────────────────────────────

AGENT_LABELS = {
    "agent_03": "Fetching market data & technicals",
    "agent_04": "Analyzing fundamentals",
    "agent_05": "Gathering news & sentiment",
    "agent_06": "Checking earnings calendar",
    "agent_07": "Analyzing insider activity",
    "agent_08": "Computing risk metrics",
    "agent_09": "Running LLM analysis",
}


# ── Streaming LLM proxy ─────────────────────────────────────────────────────

class StreamingLLMProxy(BaseChatModel):
    """
    Wraps ChatGroq to intercept stream() calls and push tokens to a callback
    so the Streamlit UI can display them live.
    """

    _real_llm: Any = None
    _token_callback: Any = None

    def __init__(self, real_llm, token_callback: Callable[[str], None], **kwargs):
        super().__init__(**kwargs)
        self._real_llm = real_llm
        self._token_callback = token_callback

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._real_llm._generate(messages, stop=stop, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for chunk in self._real_llm._stream(messages, stop=stop, **kwargs):
            text = chunk.message.content
            if text and self._token_callback:
                self._token_callback(text)
            yield chunk

    @property
    def _llm_type(self) -> str:
        return "streaming-proxy"


# ── Graph builder with progress wrappers ─────────────────────────────────────

def _wrap_agent(agent_fn, label: str, progress_cb: Callable[[str], None]):
    def wrapper(state):
        progress_cb(f"{label}...")
        result = agent_fn(state)
        progress_cb(f"{label} — done")
        return result
    return wrapper


def build_graph_with_progress(llm, progress_callback, token_callback):
    proxy_llm = StreamingLLMProxy(llm, token_callback)
    agent_09_node = partial(agent_09_llm_analysis.run, llm=proxy_llm)

    graph = StateGraph(PortfolioState)
    graph.add_node("agent_03", _wrap_agent(agent_03_market_data.run, AGENT_LABELS["agent_03"], progress_callback))
    graph.add_node("agent_04", _wrap_agent(agent_04_fundamental.run, AGENT_LABELS["agent_04"], progress_callback))
    graph.add_node("agent_05", _wrap_agent(agent_05_news_sentiment.run, AGENT_LABELS["agent_05"], progress_callback))
    graph.add_node("agent_06", _wrap_agent(agent_06_earnings_events.run, AGENT_LABELS["agent_06"], progress_callback))
    graph.add_node("agent_07", _wrap_agent(agent_07_insider_institutional.run, AGENT_LABELS["agent_07"], progress_callback))
    graph.add_node("agent_08", _wrap_agent(agent_08_risk_analysis.run, AGENT_LABELS["agent_08"], progress_callback))
    graph.add_node("agent_09", _wrap_agent(agent_09_node, AGENT_LABELS["agent_09"], progress_callback))

    graph.add_edge(START, "agent_03")
    graph.add_edge("agent_03", "agent_04")
    graph.add_edge("agent_04", "agent_05")
    graph.add_edge("agent_05", "agent_06")
    graph.add_edge("agent_06", "agent_07")
    graph.add_edge("agent_07", "agent_08")
    graph.add_edge("agent_08", "agent_09")
    graph.add_edge("agent_09", END)

    return graph.compile()


# ── Main entry point ─────────────────────────────────────────────────────────

def run_analysis(
    holdings: list[Holding],
    llm,
    progress_callback: Callable[[str], None],
    token_callback: Callable[[str], None],
) -> str:
    """
    Run the full analysis pipeline on a list of holdings.
    Returns: HTML report string.
    """
    total_value = sum(h.value for h in holdings)
    equity_tickers = [h.ticker for h in holdings if h.asset_type != "cash"]

    progress_callback(f"Portfolio: {len(holdings)} positions, ${total_value:,.0f}")

    app = build_graph_with_progress(llm, progress_callback, token_callback)

    initial_state: PortfolioState = {
        "holdings": holdings,
        "tickers": equity_tickers,
        "total_value": total_value,
        "data_warnings": [],
        "market_data": None,
        "fundamentals": None,
        "news_data": None,
        "earnings_data": None,
        "insider_data": None,
        "risk_data": None,
        "analysis": None,
        "recipient_email": None,
    }

    final_state = app.invoke(initial_state)

    progress_callback("Generating HTML report...")
    from agents.agent_10_report_delivery import _build_html
    html = _build_html(final_state)
    progress_callback("Report ready")

    return html
