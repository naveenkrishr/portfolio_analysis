"""
Agent 10 — HTML Report & Email Delivery

Takes the portfolio state (holdings + raw LLM analysis from Agent 9),
renders a styled HTML email, and sends it via the Gmail MCP server.

HTML sections (in order):
  1. Portfolio summary card  (from state data — exact numbers)
  2. Holdings table          (from state data)
  3. Recommended actions     (parsed from LLM markdown)
  4. Executive summary       (parsed from LLM markdown)
  5. Per-ticker analysis     (parsed from LLM markdown — colour-coded by recommendation)
  6. Cash position           (parsed from LLM markdown)
  7. Portfolio risk          (parsed from LLM markdown)
  8. Footer / disclaimer
"""
from __future__ import annotations

import asyncio
import re
import time
from datetime import date

from state.graph_state import PortfolioState
from state.models import Holding
from tools import gmail_client

# ── Colour maps ───────────────────────────────────────────────────────────────

_PRIORITY_COLOR = {
    "CRITICAL": "#e74c3c",
    "HIGH":     "#e67e22",
    "MEDIUM":   "#f39c12",
    "LOW":      "#7f8c8d",
}

_REC_COLOR = {
    "BUY":    "#27ae60",
    "ADD":    "#27ae60",
    "HOLD":   "#2980b9",
    "REDUCE": "#e67e22",
    "SELL":   "#e74c3c",
}

# ── Markdown section extractor ────────────────────────────────────────────────

def _section(text: str, heading: str) -> str:
    """Return content between ## heading and the next ## heading (or EOF)."""
    m = re.search(
        rf"## {re.escape(heading)}\n(.*?)(?=\n## |\Z)",
        text, re.DOTALL
    )
    return m.group(1).strip() if m else ""


# ── HTML block builders ───────────────────────────────────────────────────────

def _header(total_value: float, n_equity: int, today: str) -> str:
    return f"""\
<h2 style="color:#1a1a2e;border-bottom:3px solid #4a90e2;padding-bottom:10px;margin-top:0;">
  Portfolio Analysis &mdash; {today}
</h2>
<div style="background:#f0f7ff;border-left:4px solid #4a90e2;padding:15px;border-radius:6px;margin:0 0 24px 0;">
  <p style="font-size:22px;font-weight:bold;margin:0 0 4px 0;">${total_value:,.0f}</p>
  <p style="color:#666;margin:0;">{n_equity} equity position{"s" if n_equity != 1 else ""} &nbsp;|&nbsp; {today}</p>
</div>"""


def _holdings_table(holdings: list[Holding], total_value: float) -> str:
    rows = ""
    for i, h in enumerate(holdings):
        bg  = "#ffffff" if i % 2 == 0 else "#f9f9f9"
        pct = h.value / total_value * 100
        rows += f"""
  <tr style="background:{bg};border-bottom:1px solid #eee;">
    <td style="padding:10px;font-weight:bold;">{h.ticker}</td>
    <td style="padding:10px;">{h.name}</td>
    <td style="padding:10px;text-align:right;">{h.shares:,.2f}</td>
    <td style="padding:10px;text-align:right;">${h.price:,.2f}</td>
    <td style="padding:10px;text-align:right;">${h.value:,.0f}</td>
    <td style="padding:10px;text-align:right;">{pct:.1f}%</td>
    <td style="padding:10px;color:#888;font-size:12px;">{h.account}</td>
  </tr>"""
    return f"""\
<h3 style="color:#1a1a2e;">Holdings</h3>
<table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:24px;">
  <thead>
    <tr style="background:#1a1a2e;color:white;">
      <th style="padding:10px;text-align:left;">Symbol</th>
      <th style="padding:10px;text-align:left;">Name</th>
      <th style="padding:10px;text-align:right;">Shares</th>
      <th style="padding:10px;text-align:right;">Price</th>
      <th style="padding:10px;text-align:right;">Value</th>
      <th style="padding:10px;text-align:right;">Alloc</th>
      <th style="padding:10px;text-align:left;">Account</th>
    </tr>
  </thead>
  <tbody>{rows}
  </tbody>
</table>"""


def _actions_table(raw: str) -> str:
    rows = ""
    for line in raw.splitlines():
        m = re.match(r"\[(CRITICAL|HIGH|MEDIUM|LOW)\]\s+(\w+)\s+[—–-]\s+(.+)", line.strip())
        if not m:
            continue
        priority, ticker, action = m.group(1), m.group(2), m.group(3)
        color = _PRIORITY_COLOR.get(priority, "#7f8c8d")
        badge = (
            f'<span style="background:{color};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:12px;font-weight:bold;">{priority}</span>'
        )
        rows += f"""
  <tr style="border-bottom:1px solid #eee;">
    <td style="padding:8px;">{badge}</td>
    <td style="padding:8px;font-weight:bold;">{ticker}</td>
    <td style="padding:8px;">{action}</td>
  </tr>"""
    if not rows:
        return ""
    return f"""\
<h3 style="color:#1a1a2e;">Recommended Actions</h3>
<table style="width:100%;border-collapse:collapse;font-size:14px;margin-bottom:24px;">
  <thead>
    <tr style="background:#1a1a2e;color:white;">
      <th style="padding:8px;text-align:left;">Priority</th>
      <th style="padding:8px;text-align:left;">Ticker</th>
      <th style="padding:8px;text-align:left;">Action</th>
    </tr>
  </thead>
  <tbody>{rows}
  </tbody>
</table>"""


def _executive_summary(raw: str) -> str:
    paras = [p.strip() for p in raw.strip().splitlines() if p.strip()]
    body  = "".join(f"<p style='margin:4px 0;'>{p}</p>" for p in paras)
    return f"""\
<div style="background:#f0f7ff;border-left:4px solid #4a90e2;padding:15px;border-radius:6px;margin:0 0 24px 0;">
  <h3 style="margin:0 0 10px 0;color:#4a90e2;">Executive Summary</h3>
  {body}
</div>"""


def _ticker_cards(raw: str) -> str:
    blocks = re.split(r"(?=### )", raw.strip())
    cards  = ""

    for block in blocks:
        if not block.strip():
            continue
        hm = re.match(r"### (\w+)\s*[—–-]\s*(.+)", block)
        if not hm:
            continue
        ticker = hm.group(1)
        name   = hm.group(2).strip()

        def _field(label: str) -> str:
            fm = re.search(rf"\*\*{re.escape(label)}:\*\*\s*(.+)", block)
            return fm.group(1).strip() if fm else ""

        rec    = _field("Recommendation")
        role   = _field("Role in portfolio")
        assess = _field("Assessment")
        risks  = _field("Key risks")

        rec_key = rec.upper().split()[0] if rec else "HOLD"
        color   = _REC_COLOR.get(rec_key, "#7f8c8d")

        risks_html = ""
        if risks:
            items = [r.strip().lstrip("-•").strip() for r in re.split(r"[,;]", risks) if r.strip()]
            risks_html = (
                "<p style='margin:4px 0;'><strong>Key risks:</strong></p>"
                "<ul style='margin:4px 0 0 0;padding-left:18px;'>"
                + "".join(f"<li>{r}</li>" for r in items)
                + "</ul>"
            )

        cards += f"""\
<div style="border-left:4px solid {color};padding:15px;border-radius:6px;margin:0 0 16px 0;background:#fafafa;">
  <h3 style="margin:0 0 8px 0;color:{color};">{ticker} &mdash; {name}</h3>
  <p style="margin:4px 0;"><strong>Recommendation:</strong>
    <span style="color:{color};font-weight:bold;">{rec}</span></p>
  {"<p style='margin:4px 0;'><strong>Role:</strong> " + role + "</p>" if role else ""}
  {"<p style='margin:4px 0;'><strong>Assessment:</strong> " + assess + "</p>" if assess else ""}
  {risks_html}
</div>"""

    return f"<h3 style='color:#1a1a2e;'>Per-Ticker Analysis</h3>{cards}" if cards else ""


def _cash_block(raw: str) -> str:
    return f"""\
<div style="background:#f0fff4;border-left:4px solid #27ae60;padding:15px;border-radius:6px;margin:0 0 24px 0;">
  <h3 style="margin:0 0 8px 0;color:#27ae60;">Cash Position</h3>
  <p style="margin:0;">{raw.strip()}</p>
</div>"""


def _risk_block(raw: str) -> str:
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    items = "".join(
        f"<li style='margin:4px 0;'>{l.lstrip('-•').strip()}</li>"
        for l in lines
    )
    return f"""\
<div style="background:#f9f9f9;border-radius:6px;padding:15px;margin:0 0 24px 0;">
  <h3 style="margin:0 0 10px 0;color:#1a1a2e;">Portfolio Risk Assessment</h3>
  <ul style="margin:0;padding-left:18px;">{items}</ul>
</div>"""


def _warnings_banner(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li style='margin:4px 0;'>{w}</li>" for w in warnings)
    return f"""\
<div style="background:#fff8e1;border-left:4px solid #ffc107;padding:15px;border-radius:6px;margin:0 0 24px 0;">
  <strong style="color:#856404;">&#9888; Data Freshness Notice</strong>
  <ul style="margin:8px 0 0 0;padding-left:18px;color:#555;">{items}</ul>
  <p style="margin:8px 0 0 0;font-size:12px;color:#888;">Some positions may reflect cached data. Verify figures before acting.</p>
</div>"""


def _footer(today: str) -> str:
    return f"""\
<div style="margin-top:30px;padding:12px;background:#f9f9f9;border-radius:6px;font-size:12px;color:#999;">
  <strong>Disclaimer:</strong> This analysis is for informational purposes only and does not
  constitute financial advice. Always consult a licensed financial advisor before making
  investment decisions.<br><br>
  Generated by Portfolio Analysis Agent &middot; {today}
</div>"""


# ── HTML assembler ────────────────────────────────────────────────────────────

def _build_html(state: PortfolioState) -> str:
    holdings:     list[Holding] = state["holdings"]
    total_value:  float         = state["total_value"]
    raw:          str           = state["analysis"].raw_llm_output or ""
    warnings:     list[str]     = state.get("data_warnings") or []
    today = date.today().strftime("%B %d, %Y")

    equity = [h for h in holdings if h.asset_type != "cash"]

    s_actions    = _section(raw, "RECOMMENDED ACTIONS")
    s_summary    = _section(raw, "EXECUTIVE SUMMARY")
    s_per_ticker = _section(raw, "PER-TICKER ANALYSIS")
    s_cash       = _section(raw, "CASH POSITION")
    s_risk       = _section(raw, "PORTFOLIO RISK ASSESSMENT")

    body = "".join([
        _header(total_value, len(equity), today),
        _warnings_banner(warnings),
        _holdings_table(holdings, total_value),
        _actions_table(s_actions)      if s_actions    else "",
        _executive_summary(s_summary)  if s_summary    else "",
        _ticker_cards(s_per_ticker)    if s_per_ticker else "",
        _cash_block(s_cash)            if s_cash       else "",
        _risk_block(s_risk)            if s_risk       else "",
        _footer(today),
    ])

    return (
        '<!DOCTYPE html><html>'
        '<body style="font-family:Arial,sans-serif;max-width:720px;'
        'margin:auto;padding:20px;color:#333;">'
        + body
        + "</body></html>"
    )


# ── LangGraph node ────────────────────────────────────────────────────────────

def run(state: PortfolioState) -> PortfolioState:
    """
    LangGraph node — builds HTML report from state and sends via Gmail MCP.
    Runs the async Gmail client in a new event loop (graph.invoke is sync).
    """
    if not state.get("analysis") or not state["analysis"].raw_llm_output:
        print("\n[Agent 10] No analysis in state — skipping email.")
        return state

    today   = date.today().strftime("%B %d, %Y")
    subject = f"Portfolio Analysis \u2014 {today}"

    print("\n[Agent 10] Building HTML report...", flush=True)
    html = _build_html(state)

    print("[Agent 10] Sending via Gmail MCP...", flush=True)
    t0     = time.time()
    result = asyncio.run(gmail_client.send_email(subject=subject, body=html))
    print(f"[Agent 10] {result}  ({time.time()-t0:.1f}s)")

    return state
