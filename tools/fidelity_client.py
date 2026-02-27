"""
Fidelity MCP client (Playwright server).
Connects to the Fidelity MCP server via stdio subprocess.
No credentials here — server manages its own .env + session cookie.
"""
from __future__ import annotations

import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _server_params() -> StdioServerParameters:
    venv   = os.environ["FIDELITY_MCP_VENV"]
    server = os.environ["FIDELITY_MCP_SERVER"]
    return StdioServerParameters(command=venv, args=[server])


async def call_tool(tool_name: str, args: dict = {}) -> any:
    """Call any tool on the Fidelity MCP server. Returns parsed JSON."""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            if result.content:
                text = result.content[0].text
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, AttributeError):
                    return text
            return None


async def get_holdings(account_ids: list[str] | None = None) -> dict:
    """
    Returns:
      {
        holdings: [{account_id, ticker, name, shares, price, market_value, position_type}],
        accounts_queried: [...],
        total_positions: int,
        as_of: str,
      }
    Raises via error key if session expired or accounts not configured.
    """
    args = {}
    if account_ids:
        args["account_ids"] = account_ids
    return await call_tool("get_holdings", args) or {"holdings": [], "error": "no response"}
