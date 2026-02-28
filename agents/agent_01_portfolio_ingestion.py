"""
Agent 1 — Portfolio Ingestion

Fetches live holdings from Robinhood MCP + Fidelity MCP, merges them
into a unified list of Holding objects, and deduplicates cross-broker positions.

Merge rules:
  - Same ticker in Fidelity Z24 + Z31         → merged into one Holding (summed)
  - Same ticker in Robinhood + Fidelity        → merged into one Holding (summed)
  - Cash tickers (SPAXX, FCASH, CORE, ...)     → asset_type = "cash"
  - Known ETF tickers                          → asset_type = "etf"
  - Everything else                            → asset_type = "stock"
"""
from __future__ import annotations

from collections import defaultdict

from cache.holdings_cache import load as cache_load, save as cache_save
from state.models import Holding
from tools import robinhood_client, fidelity_client

# ── Classification helpers ────────────────────────────────────────────────────

# Tickers that are always cash / money market
_CASH_TICKERS = {"SPAXX", "FCASH", "CORE", "FDRXX", "FZFXX", "FDIC", "VMFXX"}

# ETF tickers we know about — expanded as new positions appear
_KNOWN_ETFS = {"VOO", "QQQ", "QQMG", "SPY", "IVV", "VTI", "VEA", "VWO",
               "BND", "SCHB", "SCHD", "JEPI", "JEPQ", "QQQM"}


def _classify(ticker: str, rh_type: str | None = None) -> str:
    if ticker in _CASH_TICKERS:
        return "cash"
    if ticker in _KNOWN_ETFS:
        return "etf"
    if rh_type and rh_type.lower() == "etf":
        return "etf"
    return "stock"


# ── Robinhood parsing ─────────────────────────────────────────────────────────

def _parse_robinhood(raw: dict) -> list[Holding]:
    """
    Parse r.build_holdings() response.
    raw = { "TICKER": { price, quantity, equity, type, name, ... } }
    All numeric fields from Robinhood arrive as strings.
    """
    holdings = []
    for ticker, data in raw.items():
        if not isinstance(data, dict):
            continue
        try:
            shares = float(data.get("quantity", 0))
            price  = float(data.get("price", 0))
            value  = float(data.get("equity", 0)) or shares * price
            name   = data.get("name", ticker)[:50]
            rh_type = data.get("type", "")

            if shares <= 0:
                continue

            holdings.append(Holding(
                ticker=ticker.upper(),
                name=name,
                shares=shares,
                price=price,
                value=value,
                account="Robinhood",
                asset_type=_classify(ticker, rh_type),
            ))
        except (ValueError, TypeError):
            continue
    return holdings


# ── Fidelity parsing ──────────────────────────────────────────────────────────

def _parse_fidelity(raw: dict) -> list[Holding]:
    """
    Parse Fidelity MCP get_holdings() response.
    raw = { holdings: [{account_id, ticker, name, shares, price, market_value, ...}] }
    Numeric fields are already floats (parsed by the MCP server).
    """
    holdings = []
    for h in raw.get("holdings", []):
        try:
            ticker = h.get("ticker", "").upper()
            if not ticker:
                continue

            shares = float(h.get("shares", 0))
            price  = float(h.get("price", 0))
            value  = float(h.get("market_value", 0)) or shares * price
            name   = h.get("name", ticker)[:50]
            acct   = h.get("account_id", "Fidelity")

            if shares <= 0 and value <= 0:
                continue

            holdings.append(Holding(
                ticker=ticker,
                name=name,
                shares=shares,
                price=price,
                value=value,
                account=f"Fidelity-{acct}",
                asset_type=_classify(ticker),
            ))
        except (ValueError, TypeError):
            continue
    return holdings


# ── Merge ────────────────────────────────────────────────────────────────────

def _merge(rh: list[Holding], fid: list[Holding]) -> list[Holding]:
    """
    Merge Robinhood + Fidelity holdings, summing shares/values for duplicate tickers.
    - Multiple Fidelity accounts for the same ticker → one row
    - Cross-broker duplicates (e.g. QQMG) → one row
    """
    by_ticker: dict[str, list[Holding]] = defaultdict(list)
    for h in rh + fid:
        by_ticker[h.ticker].append(h)

    merged = []
    for ticker, entries in by_ticker.items():
        if len(entries) == 1:
            merged.append(entries[0])
            continue

        # Combine: sum shares and values, use the highest-confidence price
        total_shares = sum(e.shares for e in entries)
        total_value  = sum(e.value  for e in entries)

        # Price: prefer the entry with the most shares (most significant position)
        primary = max(entries, key=lambda e: e.shares)

        # Account label
        accounts = sorted({e.account for e in entries})
        account_label = " + ".join(accounts)

        merged.append(Holding(
            ticker=ticker,
            name=primary.name,
            shares=total_shares,
            price=primary.price,
            value=total_value,
            account=account_label,
            asset_type=primary.asset_type,
        ))

    # Sort: equities by value desc, cash last
    merged.sort(key=lambda h: (h.asset_type == "cash", -h.value))
    return merged


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _from_cache(broker: str) -> tuple[list[Holding], str] | None:
    """Load holdings from the on-disk snapshot for a broker."""
    result = cache_load(broker)
    if result is None:
        return None
    dicts, fetched_at = result
    holdings = []
    for d in dicts:
        try:
            holdings.append(Holding(**d))
        except Exception:
            pass
    return holdings, fetched_at


# ── Main entry point ─────────────────────────────────────────────────────────

async def run() -> tuple[list[Holding], list[str]]:
    """
    Fetch holdings from both brokers concurrently.

    Returns:
        (holdings, data_warnings)
        data_warnings is a list of human-readable strings describing any fallbacks
        or partial failures that the rest of the pipeline should know about.
    """
    import asyncio

    data_warnings: list[str] = []

    print("Fetching Robinhood holdings...", flush=True)
    print("Fetching Fidelity holdings (Playwright — may take ~10s)...", flush=True)

    rh_raw, fid_raw = await asyncio.gather(
        robinhood_client.get_holdings(),
        fidelity_client.get_holdings(),
        return_exceptions=True,
    )

    # ── Robinhood ──────────────────────────────────────────────────────────
    rh_holdings: list[Holding] = []
    if isinstance(rh_raw, Exception):
        print(f"  [WARN] Robinhood live fetch failed: {rh_raw}")
        cached = _from_cache("robinhood")
        if cached:
            rh_holdings, fetched_at = cached
            msg = f"Robinhood data from cache (last fetched: {fetched_at}) — live fetch failed."
            print(f"  [FALLBACK] {msg}")
            data_warnings.append(msg)
        else:
            data_warnings.append("Robinhood fetch failed and no cache available — positions excluded.")
    elif isinstance(rh_raw, dict):
        rh_holdings = _parse_robinhood(rh_raw)
        print(f"  Robinhood: {len(rh_holdings)} positions")
        cache_save("robinhood", [h.model_dump() for h in rh_holdings])
    else:
        print(f"  [WARN] Robinhood unexpected response: {type(rh_raw)} — {str(rh_raw)[:200]}")
        data_warnings.append(f"Robinhood returned unexpected response type: {type(rh_raw).__name__}.")

    # ── Fidelity ───────────────────────────────────────────────────────────
    fid_holdings: list[Holding] = []
    if isinstance(fid_raw, Exception):
        print(f"  [WARN] Fidelity live fetch failed: {fid_raw}")
        cached = _from_cache("fidelity")
        if cached:
            fid_holdings, fetched_at = cached
            msg = f"Fidelity data from cache (last fetched: {fetched_at}) — live fetch failed."
            print(f"  [FALLBACK] {msg}")
            data_warnings.append(msg)
        else:
            data_warnings.append("Fidelity fetch failed and no cache available — positions excluded.")
    elif isinstance(fid_raw, dict):
        if "error" in fid_raw and not fid_raw.get("holdings"):
            err = fid_raw["error"]
            print(f"  [WARN] Fidelity error: {err}")
            cached = _from_cache("fidelity")
            if cached:
                fid_holdings, fetched_at = cached
                msg = f"Fidelity data from cache (last fetched: {fetched_at}) — server returned error: {str(err)[:120]}"
                print(f"  [FALLBACK] {msg}")
                data_warnings.append(msg)
            else:
                data_warnings.append(f"Fidelity server error and no cache available: {str(err)[:120]}")
        else:
            fid_holdings = _parse_fidelity(fid_raw)
            print(f"  Fidelity: {len(fid_holdings)} positions")
            cache_save("fidelity", [h.model_dump() for h in fid_holdings])
    else:
        print(f"  [WARN] Fidelity unexpected response: {type(fid_raw)}")
        data_warnings.append(f"Fidelity returned unexpected response type: {type(fid_raw).__name__}.")

    all_holdings = _merge(rh_holdings, fid_holdings)

    if not all_holdings:
        raise RuntimeError(
            "No holdings fetched from either broker. "
            "Check MCP server logs or use --mock for offline testing."
        )

    return all_holdings, data_warnings
