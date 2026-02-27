"""
Robinhood MCP client.
Connects to the Robinhood MCP server via stdio subprocess.
No credentials here — server manages its own .env.
"""
from __future__ import annotations

import json
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _server_params() -> StdioServerParameters:
    venv   = os.environ["ROBINHOOD_MCP_VENV"]
    server = os.environ["ROBINHOOD_MCP_SERVER"]
    return StdioServerParameters(command=venv, args=[server])


async def call_tool(tool_name: str, args: dict = {}) -> any:
    """Call any tool on the Robinhood MCP server. Returns parsed JSON."""
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


def _parse(text: str) -> any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return text


async def get_holdings() -> dict:
    """
    Login + get holdings in a single MCP session so r.login() state persists.
    robin_stocks raises if build_holdings() is called before login() in each process.

    Returns r.build_holdings() dict:
      { "TICKER": { price, quantity, average_buy_price, equity, type, name, ... } }
    All numeric fields are strings — caller must cast.
    """
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Login first (server reads creds from its own .env)
            await session.call_tool("login", {})
            # Now fetch holdings with the live session
            result = await session.call_tool("get_holdings", {})
            if result.content:
                parsed = _parse(result.content[0].text)
                if isinstance(parsed, dict):
                    return parsed
                raise RuntimeError(f"Robinhood get_holdings error: {parsed}")
            return {}
