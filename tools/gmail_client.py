"""
Gmail MCP client.
Connects to the Gmail MCP server via stdio subprocess.
Credentials and OAuth token are managed by the server.
"""
from __future__ import annotations

import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _server_params() -> StdioServerParameters:
    venv   = os.environ["GMAIL_MCP_VENV"]
    server = os.environ["GMAIL_MCP_SERVER"]
    return StdioServerParameters(command=venv, args=[server])


async def send_email(subject: str, body: str, to: str | None = None) -> str:
    """
    Send an HTML email via the Gmail MCP server.

    Args:
        to: If provided, send to this address (single or comma-separated).
            If None, falls back to REPORT_RECIPIENT env var.
    Returns the server's response string(s).
    """
    raw = to if to else os.environ["REPORT_RECIPIENT"]
    recipients = [r.strip() for r in raw.split(",") if r.strip()]
    responses = []

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for recipient in recipients:
                result = await session.call_tool("send_email", {
                    "to":      recipient,
                    "subject": subject,
                    "body":    body,
                    "is_html": True,
                })
                if result.content:
                    responses.append(f"{recipient}: {result.content[0].text}")
                else:
                    responses.append(f"{recipient}: No response from Gmail MCP")

    return " | ".join(responses)
