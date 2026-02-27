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


async def send_email(subject: str, body: str) -> str:
    """
    Send an HTML email to REPORT_RECIPIENT via the Gmail MCP server.
    Returns the server's response string (e.g. "Email sent successfully. Message ID: ...").
    """
    recipient = os.environ["REPORT_RECIPIENT"]
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("send_email", {
                "to":      recipient,
                "subject": subject,
                "body":    body,
                "is_html": True,
            })
            if result.content:
                return result.content[0].text
            return "No response from Gmail MCP"
