"""
Holdings snapshot cache — simple per-broker JSON files.

Used as fallback when a live MCP fetch fails.
Broker names: "robinhood", "fidelity"

Cache files: .cache/robinhood_snapshot.json, .cache/fidelity_snapshot.json
Each file: { "fetched_at": "2026-02-27T10:30:00", "holdings": [...] }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_CACHE_DIR = Path(".cache")


def _path(broker: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"{broker}_snapshot.json"


def save(broker: str, holdings_dicts: list[dict]) -> None:
    """Persist a successful fetch to disk."""
    data = {
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "holdings": holdings_dicts,
    }
    with open(_path(broker), "w") as f:
        json.dump(data, f, indent=2)


def load(broker: str) -> tuple[list[dict], str] | None:
    """
    Load the last cached snapshot for a broker.

    Returns:
        (holdings_dicts, fetched_at_str)  if cache exists
        None                               if no cache file
    """
    p = _path(broker)
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        return data["holdings"], data["fetched_at"]
    except Exception:
        return None
