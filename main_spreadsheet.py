#!/usr/bin/env python3
"""
Spreadsheet Portfolio Analysis — Daily Report
===============================================
Reads all CSV/Excel files in input/, groups holdings by email,
runs agents 3-10 per email group, and sends each person their report.

Designed to run once a day (e.g. via cron).

Usage:
  ./venv/bin/python3 main_spreadsheet.py                    # process + email
  ./venv/bin/python3 main_spreadsheet.py --no-email         # process, skip email
  ./venv/bin/python3 main_spreadsheet.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
  ./venv/bin/python3 main_spreadsheet.py --max-tokens 3000

Folder layout:
  input/              ← place CSV/Excel files here (kept in place after processing)
"""
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from main import build_graph, DEFAULT_MODEL
from mlx_wrapper import MLXChatModel
from state.graph_state import PortfolioState
from agents.spreadsheet_ingestion import load as load_spreadsheet

INPUT_DIR  = Path("input")
EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _find_files() -> list[Path]:
    """Return spreadsheet files in input/ (not in subdirectories)."""
    if not INPUT_DIR.exists():
        return []
    return sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def _process_file(path: Path, app, send_email: bool):
    """Load spreadsheet, run pipeline per email group."""
    print(f"\n{'='*60}")
    print(f"Processing: {path.name}")
    print(f"{'='*60}")

    groups = load_spreadsheet(path)
    print(f"Found {len(groups)} email group(s): {', '.join(groups.keys())}\n")

    for email, holdings in groups.items():
        total_value = sum(h.value for h in holdings)
        equity_tickers = [h.ticker for h in holdings if h.asset_type != "cash"]

        print(f"\n--- {email} ---")
        print(f"Positions: {len(holdings)}  |  Total: ${total_value:,.0f}")
        print(f"Tickers:   {', '.join(equity_tickers)}")

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
            "recipient_email": email if send_email else None,
        }

        t0 = time.time()
        app.invoke(initial_state)
        print(f"Completed {email} in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Spreadsheet Portfolio Analysis — Daily Report")
    parser.add_argument("--no-email",   action="store_true",
                        help="Skip email delivery (Agent 10)")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    # ── Find files ────────────────────────────────────────────────────────────
    files = _find_files()
    if not files:
        print(f"No CSV/Excel files found in {INPUT_DIR.resolve()}/")
        return

    # ── Load model once ───────────────────────────────────────────────────────
    print(f"\n=== Spreadsheet Portfolio Analysis ===")
    print(f"Model: {args.model}  |  Email: {'off' if args.no_email else 'on'}")
    print(f"Files: {', '.join(f.name for f in files)}\n")
    print("Loading model...", flush=True)

    t0 = time.time()
    llm = MLXChatModel(
        model_path=args.model,
        max_tokens=args.max_tokens,
        temperature=0.1,
    )
    llm._load()
    mem = llm.memory_stats()
    print(f"Loaded in {time.time()-t0:.1f}s — active: {mem['active_gb']} GB | peak: {mem['peak_gb']} GB\n")

    # ── Build graph once ──────────────────────────────────────────────────────
    app = build_graph(llm, send_email=not args.no_email)

    # ── Process each file ─────────────────────────────────────────────────────
    for path in files:
        try:
            _process_file(path, app, send_email=not args.no_email)
        except Exception as e:
            print(f"\n[ERROR] Failed to process {path.name}: {e}")

    print(f"\nDone. Memory — active: {llm.memory_stats()['active_gb']} GB\n")


if __name__ == "__main__":
    main()
