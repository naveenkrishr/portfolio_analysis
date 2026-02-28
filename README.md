# Portfolio Analysis

**Mac Only** — Requires Apple Silicon for the MLX local LLM.

## Overview

An AI-powered multi-agent system that fetches a live investment portfolio, runs multi-dimensional analysis, and emails a formatted HTML report — all locally, no cloud LLM.

**Data sources:**
- **Robinhood MCP** — live holdings
- **Fidelity MCP** — live holdings via Playwright browser automation
- **yfinance** — price history, technicals, fundamentals, news headlines
- **Qwen2.5-14B on MLX** — local LLM for analysis and recommendations
- **Gmail MCP** — HTML report delivery

## Pipeline

```
Agent 1: Portfolio Ingestion  ←  Robinhood MCP + Fidelity MCP
                                  (falls back to cached snapshot if broker fails)
         │
         ├── Agent 3: Market Data   ←  yfinance: OHLCV + SMA/RSI/MACD/Bollinger
         │
         ├── Agent 4: Fundamentals  ←  yfinance: P/E, ROE, FCF, analyst ratings
         │
         ├── Agent 5: News          ←  yfinance: recent headlines + keyword sentiment
         │
         └── Agent 9: LLM Analysis  ←  Qwen2.5-14B (MLX) — synthesises all data
                  │
                  └── Agent 10: Report Delivery  ←  HTML email via Gmail MCP
```

## What the LLM receives per ticker

| Data | Source | Fields |
|---|---|---|
| Holdings | Broker MCPs | shares, price, value, allocation % |
| Technicals | Agent 3 | SMA50/200, RSI, MACD, Bollinger %B, trend label |
| Fundamentals | Agent 4 | P/E (ttm/fwd), P/B, EV/EBITDA, ROE, net margin, revenue growth, D/E, FCF, analyst consensus + target |
| News | Agent 5 | Last 7 days of headlines, sentiment score (bullish/neutral/bearish) |

## Requirements

- macOS (Apple Silicon M1 or newer)
- Python 3.11+
- 16 GB RAM (Qwen2.5-14B-4bit uses ~8.5 GB)
- Robinhood, Fidelity, and Gmail MCP servers configured separately

## Setup

```bash
git clone https://github.com/naveenkrishr/portfolio_analysis.git
cd portfolio_analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in MCP server paths and email recipient
```

**.env required keys:**
```bash
ROBINHOOD_MCP_VENV=/path/to/robinhood-agent/venv/bin/python3
ROBINHOOD_MCP_SERVER=/path/to/robinhood-agent/server.py

FIDELITY_MCP_VENV=/path/to/fidelity-playwright-agent/venv/bin/python3
FIDELITY_MCP_SERVER=/path/to/fidelity-playwright-agent/server.py

GMAIL_MCP_VENV=/path/to/gmail-agent/venv/bin/python3
GMAIL_MCP_SERVER=/path/to/gmail-agent/server.py

REPORT_RECIPIENT=your@email.com
```

## Usage

```bash
# Full pipeline — live portfolio + analysis + email
./venv/bin/python3 main.py

# Offline test — hardcoded mock data, skip email
./venv/bin/python3 main.py --mock --no-email

# Skip email only
./venv/bin/python3 main.py --no-email

# Lighter model (faster, less RAM)
./venv/bin/python3 main.py --model mlx-community/Qwen2.5-7B-Instruct-4bit

# Increase output length
./venv/bin/python3 main.py --max-tokens 6000
```

## Broker Fallback

If a broker MCP fails, the pipeline automatically falls back to the last successful snapshot stored in `.cache/`. The LLM prompt and HTML report both display a **Data Freshness Notice** listing which sources used cached data and when they were last fetched.

## Project Structure

```
portfolio_analysis/
├── agents/
│   ├── agent_01_portfolio_ingestion.py   # Robinhood + Fidelity merge, cache fallback
│   ├── agent_03_market_data.py           # yfinance OHLCV + technical indicators
│   ├── agent_04_fundamental.py           # yfinance fundamentals + analyst ratings
│   ├── agent_05_news_sentiment.py        # yfinance headlines + keyword sentiment
│   ├── agent_09_llm_analysis.py          # Qwen2.5-14B prompt + streaming
│   └── agent_10_report_delivery.py       # HTML report + Gmail MCP send
│
├── tools/
│   ├── robinhood_client.py               # MCP client
│   ├── fidelity_client.py                # MCP client (Playwright)
│   ├── gmail_client.py                   # MCP client
│   ├── yfinance_client.py                # Batch OHLCV download
│   ├── yfinance_fundamentals.py          # .info per ticker
│   └── yfinance_news.py                  # .news per ticker (yfinance 1.x format)
│
├── analysis/
│   ├── technicals.py                     # SMA, RSI, MACD, Bollinger, TechnicalSnapshot
│   ├── fundamentals.py                   # FundamentalSnapshot + .summary()
│   └── news_sentiment.py                 # NewsSnapshot + keyword sentiment
│
├── cache/
│   └── holdings_cache.py                 # Per-broker JSON snapshot (save/load)
│
├── state/
│   ├── models.py                         # Holding, PortfolioAnalysis (Pydantic)
│   └── graph_state.py                    # PortfolioState TypedDict
│
├── main.py                               # Entry point, LangGraph orchestration
├── mlx_wrapper.py                        # LangChain BaseChatModel wrapping mlx-lm
├── mock_data.py                          # Offline test fixtures
├── test_agent_03.py                      # Standalone market data test
├── test_agent_04.py                      # Standalone fundamentals test
└── test_agent_05.py                      # Standalone news sentiment test
```

## Standalone Tests

Each data agent can be tested independently without the LLM or MCP servers:

```bash
./venv/bin/python3 test_agent_03.py   # technicals for VOO, SNOW, QQMG, NVDA, TSLA
./venv/bin/python3 test_agent_04.py   # fundamentals + analyst ratings
./venv/bin/python3 test_agent_05.py   # recent headlines + sentiment scores
```

## MLX Models

| Model | VRAM | Speed | Notes |
|---|---|---|---|
| `mlx-community/Qwen2.5-14B-Instruct-4bit` | ~8.5 GB | ~35 words/s | Default |
| `mlx-community/Qwen2.5-7B-Instruct-4bit`  | ~4.5 GB | ~60 words/s | Lighter |
| `mlx-community/Qwen2.5-32B-Instruct-4bit` | ~18 GB  | ~15 words/s | Requires 32 GB RAM |

## Troubleshooting

**Fidelity session expired:**
Re-run `login.py` in the fidelity MCP server directory (~30 day TTL on Playwright session).

**First run slow:**
The MLX model (~8.5 GB) downloads on first use. Subsequent runs load from cache.

**Out of memory:**
Switch to the 7B model: `--model mlx-community/Qwen2.5-7B-Instruct-4bit`

**MCP connection error:**
Verify paths in `.env` and that each MCP server's venv has its dependencies installed.

---

**Tested on:** macOS 15 (Sequoia), Apple M4, 16 GB RAM
