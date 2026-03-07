"""
Finnhub insider transactions client — /stock/insider-transactions endpoint.

Fetches recent insider trades (Form 4 filings) per ticker.
Free tier: 60 API calls/min (shared across all Finnhub usage).
API key from FINNHUB_API_KEY env var.

Returns per ticker: list of transactions with name, share, change, value,
transaction type, and filing date.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import requests

_API_KEY = os.getenv("FINNHUB_API_KEY", "")
_BASE_URL = "https://finnhub.io/api/v1"


def fetch_insider_transactions(
    tickers: list[str],
    months_back: int = 6,
) -> dict[str, list[dict]]:
    """
    Fetch insider transactions from Finnhub for each ticker.

    Args:
        tickers: equity ticker symbols
        months_back: how many months of history to fetch

    Returns:
        dict[ticker, list[transaction]] where transaction = {
            name, share, change, transaction_type,
            filing_date, transaction_date, value
        }
        Sorted by filing_date (most recent first).
        Returns empty dict if no API key.
    """
    if not _API_KEY:
        return {}

    now = datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(days=months_back * 30)

    result: dict[str, list[dict]] = {}

    for ticker in tickers:
        try:
            resp = requests.get(
                f"{_BASE_URL}/stock/insider-transactions",
                params={
                    "symbol": ticker,
                    "token": _API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            transactions = []
            for txn in data.get("data", []):
                filing_date = txn.get("filingDate", "")
                if not filing_date:
                    continue

                # Filter to recent transactions
                try:
                    filed = datetime.strptime(filing_date, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    if filed < cutoff:
                        continue
                except ValueError:
                    continue

                name = (txn.get("name") or "").strip()
                change = txn.get("change", 0)
                # Finnhub uses transactionCode: P=Purchase, S=Sale, A=Award
                txn_code = txn.get("transactionCode") or ""

                # Only include open-market buys and sells (skip awards/grants)
                if txn_code not in ("P", "S"):
                    continue

                txn_type = "Purchase" if txn_code == "P" else "Sale"

                share = txn.get("share", 0)
                transaction_price = txn.get("transactionPrice")
                value = None
                if transaction_price and change:
                    value = abs(change * transaction_price)

                transactions.append({
                    "name": name,
                    "share": share,
                    "change": change,
                    "transaction_type": txn_type,
                    "filing_date": filing_date,
                    "transaction_date": txn.get("transactionDate", ""),
                    "transaction_price": transaction_price,
                    "value": value,
                })

            # Sort by filing date, most recent first
            transactions.sort(key=lambda t: t["filing_date"], reverse=True)

            if transactions:
                result[ticker] = transactions

        except Exception as e:
            print(f"  [warn] {ticker}: Finnhub insider transactions failed — {e}")

    return result


def fetch_insider_sentiment(
    tickers: list[str],
) -> dict[str, dict]:
    """
    Fetch aggregated insider sentiment from Finnhub /stock/insider-sentiment.

    Returns monthly aggregated insider buying/selling metrics (MSPR).
    Useful as a quick signal without parsing individual transactions.

    Returns:
        dict[ticker, {mspr, change, total_shares}] — latest month's data
    """
    if not _API_KEY:
        return {}

    now = datetime.now(tz=timezone.utc)
    date_from = (now - timedelta(days=180)).strftime("%Y-%m-%d")
    date_to = now.strftime("%Y-%m-%d")

    result: dict[str, dict] = {}

    for ticker in tickers:
        try:
            resp = requests.get(
                f"{_BASE_URL}/stock/insider-sentiment",
                params={
                    "symbol": ticker,
                    "from": date_from,
                    "to": date_to,
                    "token": _API_KEY,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            entries = data.get("data", [])
            if not entries:
                continue

            # Use the most recent month
            latest = entries[-1]
            result[ticker] = {
                "mspr": latest.get("mspr", 0),       # Monthly Share Purchase Ratio
                "change": latest.get("change", 0),    # Net share change
                "month": latest.get("month"),
                "year": latest.get("year"),
            }

        except Exception as e:
            print(f"  [warn] {ticker}: Finnhub insider sentiment failed — {e}")

    return result
