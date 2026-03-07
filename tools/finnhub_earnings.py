"""
Finnhub earnings calendar client — /calendar/earnings endpoint.

Single API call returns earnings dates for ALL companies in a date range.
We filter to only portfolio tickers.

Free tier: 60 API calls/min (shared across all Finnhub usage).
API key from FINNHUB_API_KEY env var.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import requests

_API_KEY = os.getenv("FINNHUB_API_KEY", "")
_BASE_URL = "https://finnhub.io/api/v1"


def fetch_earnings(
    tickers: list[str],
    weeks_ahead: int = 13,
) -> dict[str, list[dict]]:
    """
    Fetch upcoming earnings calendar from Finnhub, filtered to portfolio tickers.

    Single API call — returns all companies in the date range, then filters
    to only the tickers we care about.

    Args:
        tickers: equity ticker symbols to filter for
        weeks_ahead: how many weeks into the future to look

    Returns:
        dict[ticker, list[event]] where event = {
            date, hour, quarter, year,
            eps_estimate, revenue_estimate,
            eps_actual, revenue_actual
        }
        Sorted by date (soonest first).
        Returns empty dict if no API key.
    """
    if not _API_KEY:
        return {}

    now = datetime.now(tz=timezone.utc)
    date_from = now.strftime("%Y-%m-%d")
    date_to = (now + timedelta(weeks=weeks_ahead)).strftime("%Y-%m-%d")

    ticker_set = set(t.upper() for t in tickers)

    try:
        resp = requests.get(
            f"{_BASE_URL}/calendar/earnings",
            params={
                "from": date_from,
                "to": date_to,
                "token": _API_KEY,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        all_events = data.get("earningsCalendar", [])

        result: dict[str, list[dict]] = {}
        for ev in all_events:
            symbol = (ev.get("symbol") or "").upper()
            if symbol not in ticker_set:
                continue

            event = {
                "date": ev.get("date", ""),
                "hour": ev.get("hour", ""),          # "bmo", "amc", or ""
                "quarter": ev.get("quarter"),
                "year": ev.get("year"),
                "eps_estimate": ev.get("epsEstimate"),
                "revenue_estimate": ev.get("revenueEstimate"),
                "eps_actual": ev.get("epsActual"),
                "revenue_actual": ev.get("revenueActual"),
            }

            result.setdefault(symbol, []).append(event)

        # Sort each ticker's events by date
        for ticker in result:
            result[ticker].sort(key=lambda e: e["date"])

        return result

    except Exception as e:
        print(f"  [warn] Finnhub earnings calendar failed — {e}")
        return {}
