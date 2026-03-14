"""
Spreadsheet Ingestion — replaces Agent 1 for the spreadsheet workflow.

Reads a CSV or Excel file with columns: ticker, quantity, avg_price, email
Groups by email, consolidates duplicate tickers, fetches current prices via yfinance.

Returns dict[email, list[Holding]].
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import yfinance as yf

from state.models import Holding

# ── Classification helpers (same as agent_01) ────────────────────────────────

_CASH_TICKERS = {"SPAXX", "FCASH", "CORE", "FDRXX", "FZFXX", "FDIC", "VMFXX"}
_KNOWN_ETFS = {"VOO", "QQQ", "QQMG", "SPY", "IVV", "VTI", "VEA", "VWO",
               "BND", "SCHB", "SCHD", "JEPI", "JEPQ", "QQQM"}


def _classify(ticker: str) -> str:
    if ticker in _CASH_TICKERS:
        return "cash"
    if ticker in _KNOWN_ETFS:
        return "etf"
    return "stock"


# ── Price fetch ──────────────────────────────────────────────────────────────

def _fetch_prices(tickers: list[str]) -> dict[str, tuple[float, str]]:
    """
    Batch-fetch current prices and short names via yfinance.
    Returns {TICKER: (price, short_name)}.
    """
    result: dict[str, tuple[float, str]] = {}
    if not tickers:
        return result

    objs = yf.Tickers(" ".join(tickers))
    for ticker in tickers:
        try:
            info = objs.tickers[ticker].info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
            name = info.get("shortName", ticker)[:50]
            result[ticker] = (float(price), name)
        except Exception:
            result[ticker] = (0.0, ticker)
    return result


# ── Main entry point ─────────────────────────────────────────────────────────

_REQUIRED_COLS = {"ticker", "quantity", "avg_price", "email"}


def load(path: str | Path) -> dict[str, list[Holding]]:
    """
    Read a CSV/Excel file and return holdings grouped by email.

    Expected columns: ticker, quantity, avg_price, email
    Returns: {email_address: [Holding, ...]}
    """
    path = Path(path)
    if path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Spreadsheet missing required columns: {', '.join(sorted(missing))}. "
            f"Found: {', '.join(df.columns)}"
        )

    # Clean data
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["avg_price"] = pd.to_numeric(df["avg_price"], errors="coerce").fillna(0)
    df = df[df["quantity"] > 0]

    if df.empty:
        raise ValueError("Spreadsheet has no valid rows (all quantity <= 0 or missing).")

    # Fetch current prices for all unique tickers
    all_tickers = sorted(df["ticker"].unique().tolist())
    non_cash = [t for t in all_tickers if t not in _CASH_TICKERS]
    print(f"  Fetching prices for {len(non_cash)} tickers via yfinance...", flush=True)
    prices = _fetch_prices(non_cash)

    # Group by email, consolidate duplicate tickers within each group
    grouped: dict[str, list[Holding]] = {}

    for email, group_df in df.groupby("email"):
        # Consolidate: sum quantity and total_cost per ticker
        consolidated: dict[str, dict] = defaultdict(lambda: {
            "quantity": 0.0, "total_cost": 0.0
        })
        for _, row in group_df.iterrows():
            ticker = row["ticker"]
            qty = row["quantity"]
            avg = row["avg_price"]
            consolidated[ticker]["quantity"] += qty
            consolidated[ticker]["total_cost"] += avg * qty

        holdings: list[Holding] = []
        for ticker, data in consolidated.items():
            shares = data["quantity"]
            total_cost = data["total_cost"]
            avg_cost = total_cost / shares if shares else 0.0

            price_info = prices.get(ticker)
            if price_info:
                price, name = price_info
            else:
                # Cash tickers or failed lookups — use avg_price as fallback
                price = avg_cost
                name = ticker

            if price == 0.0 and avg_cost > 0:
                print(f"  [WARN] No current price for {ticker}, using avg_cost as fallback")
                price = avg_cost

            value = shares * price

            holdings.append(Holding(
                ticker=ticker,
                name=name,
                shares=shares,
                price=price,
                value=value,
                account="Spreadsheet",
                asset_type=_classify(ticker),
                avg_cost=avg_cost,
                total_cost=total_cost,
            ))

        # Sort: equities by value desc, cash last
        holdings.sort(key=lambda h: (h.asset_type == "cash", -h.value))
        grouped[email] = holdings

    return grouped
