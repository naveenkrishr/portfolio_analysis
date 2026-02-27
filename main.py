#!/usr/bin/env python3
"""
Portfolio Analysis — Runner
============================
Live portfolio (Robinhood MCP + Fidelity MCP)
  → Agent 3  (market data: price history + technicals via yfinance)
  → Agent 9  (LLM analysis — Qwen2.5-14B)
  → Agent 10 (HTML email via Gmail MCP)

Usage:
  ./venv/bin/python3 main.py                    # live portfolio + email
  ./venv/bin/python3 main.py --mock             # hardcoded mock data (offline/faster)
  ./venv/bin/python3 main.py --no-email         # skip email delivery
  ./venv/bin/python3 main.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
  ./venv/bin/python3 main.py --max-tokens 3000
"""
import argparse
import asyncio
import time
from functools import partial

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from mlx_wrapper import MLXChatModel
from state.graph_state import PortfolioState
from state.models import Holding
from agents import agent_03_market_data, agent_09_llm_analysis, agent_10_report_delivery

load_dotenv()

DEFAULT_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"


def build_graph(llm: MLXChatModel, send_email: bool = True):
    agent_09_node = partial(agent_09_llm_analysis.run, llm=llm)
    graph = StateGraph(PortfolioState)
    graph.add_node("agent_03", agent_03_market_data.run)
    graph.add_node("agent_09", agent_09_node)
    graph.add_edge(START, "agent_03")
    graph.add_edge("agent_03", "agent_09")
    if send_email:
        graph.add_node("agent_10", agent_10_report_delivery.run)
        graph.add_edge("agent_09", "agent_10")
        graph.add_edge("agent_10", END)
    else:
        graph.add_edge("agent_09", END)
    return graph.compile()


async def _fetch_live_holdings() -> list[Holding]:
    """Fetch real holdings from both brokers via MCP."""
    from agents import agent_01_portfolio_ingestion
    return await agent_01_portfolio_ingestion.run()


def _load_mock_holdings() -> list[Holding]:
    from mock_data import MOCK_HOLDINGS
    return MOCK_HOLDINGS


def main():
    parser = argparse.ArgumentParser(description="Portfolio Analysis")
    parser.add_argument("--mock",       action="store_true",
                        help="Use hardcoded mock data instead of live MCP data")
    parser.add_argument("--no-email",   action="store_true",
                        help="Skip HTML email delivery (Agent 10)")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\n=== Portfolio Analysis ===")
    print(f"Model: {args.model}  |  Source: {'MOCK' if args.mock else 'LIVE (Robinhood + Fidelity)'}  |  Email: {'off' if args.no_email else 'on'}")
    print("Loading model...", flush=True)

    t0 = time.time()
    llm = MLXChatModel(
        model_path=args.model,
        max_tokens=args.max_tokens,
        temperature=0.1,
    )
    llm._load()
    load_time = time.time() - t0

    mem = llm.memory_stats()
    print(f"Loaded in {load_time:.1f}s — active: {mem['active_gb']} GB | peak: {mem['peak_gb']} GB\n")

    # ── Fetch portfolio ──────────────────────────────────────────────────────
    if args.mock:
        holdings = _load_mock_holdings()
    else:
        t1 = time.time()
        holdings = asyncio.run(_fetch_live_holdings())
        print(f"Portfolio fetched in {time.time() - t1:.1f}s\n")

    total_value = sum(h.value for h in holdings)
    equity_tickers = [h.ticker for h in holdings if h.asset_type != "cash"]

    print(f"Portfolio: {len(holdings)} positions  |  Total value: ${total_value:,.0f}")
    print(f"Equity:    {', '.join(equity_tickers)}")
    cash = [h for h in holdings if h.asset_type == "cash"]
    if cash:
        print(f"Cash:      {', '.join(h.ticker for h in cash)}  (${sum(h.value for h in cash):,.0f})")
    print()

    # ── Run LangGraph ────────────────────────────────────────────────────────
    initial_state: PortfolioState = {
        "holdings": holdings,
        "tickers": equity_tickers,
        "total_value": total_value,
        "market_data": None,
        "analysis": None,
    }

    app = build_graph(llm, send_email=not args.no_email)
    app.invoke(initial_state)

    print(f"\nMemory after run — active: {llm.memory_stats()['active_gb']} GB\n")


if __name__ == "__main__":
    main()
