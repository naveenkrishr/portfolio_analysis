# Portfolio Analysis 📊

**⚠️ Mac Only** — This project is designed exclusively for macOS machines due to MLX framework requirements.

## Overview

Portfolio Analysis is an AI-powered multi-agent system that analyzes live investment portfolios and generates HTML email reports. It integrates with:

- **Robinhood** — Fetch live account holdings
- **Fidelity** — Fetch account positions (via Playwright automation)  
- **yfinance** — Retrieve historical price data and technical indicators
- **Qwen2.5-14B** (MLX) — LLM-powered portfolio analysis
- **Gmail** — Send formatted HTML reports

## Architecture

The system uses **LangGraph** to orchestrate three main agents:

```
┌─────────────────────────────────────────────────────┐
│ Agent 1: Portfolio Ingestion (Robinhood + Fidelity) │
└─────────────────┬───────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────┐
│ Agent 3: Market Data (yfinance + Technicals)        │
└─────────────────┬───────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────┐
│ Agent 9: LLM Analysis (Qwen2.5 on MLX)              │
└─────────────────┬───────────────────────────────────┘
                  │
                  ↓
┌─────────────────────────────────────────────────────┐
│ Agent 10: Report Delivery (HTML Email via Gmail)    │
└─────────────────────────────────────────────────────┘
```

## Requirements

**System:**
- macOS (Big Sur or newer recommended)
- Python 3.11+
- Homebrew (for M1/M2/M3 Macs)

**Accounts/Access:**
- Robinhood account credentials
- Fidelity account credentials
- Gmail account (with app-specific password)

**Hardware (Recommended):**
- Apple Silicon (M1, M2, M3+) or Intel Mac
- 16GB+ RAM (for Qwen2.5-14B model)
- ~10GB disk space (for MLX model cache)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/naveenkrishr/portfolio_analysis.git
cd portfolio_analysis
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create/edit `.env` file in the project root:

```bash
# MCP Server Paths (for Robinhood, Fidelity, Gmail agents)
ROBINHOOD_MCP_VENV=/path/to/robinhood-agent/venv/bin/python3
ROBINHOOD_MCP_SERVER=/path/to/robinhood-agent/server.py

FIDELITY_MCP_VENV=/path/to/fidelity-playwright-agent/venv/bin/python3
FIDELITY_MCP_SERVER=/path/to/fidelity-playwright-agent/server.py

GMAIL_MCP_VENV=/path/to/gmail-agent/venv/bin/python3
GMAIL_MCP_SERVER=/path/to/gmail-agent/server.py

# Report Delivery
REPORT_RECIPIENT=your-email@gmail.com

# MLX Configuration
MLX_MODEL_PATH=mlx-community/Qwen2.5-14B-Instruct-4bit
MLX_MAX_TOKENS=4096
```

## Usage

### Run Full Pipeline (Live Portfolio + Analysis + Email)
```bash
python main.py
```

### Run with Mock Data (Offline/Faster Testing)
```bash
python main.py --mock
```

### Skip Email Delivery
```bash
python main.py --no-email
```

### Use Different Model
```bash
python main.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

### Customize Token Limit
```bash
python main.py --max-tokens 2048
```

### Full Example
```bash
python main.py --mock --no-email --max-tokens 3000
```

## Project Structure

```
portfolio_analysis/
├── agents/                 # LangGraph agent nodes
│   ├── agent_01_portfolio_ingestion.py
│   ├── agent_03_market_data.py
│   ├── agent_09_llm_analysis.py
│   └── agent_10_report_delivery.py
│
├── tools/                  # MCP client wrappers & API integrations
│   ├── robinhood_client.py
│   ├── fidelity_client.py
│   ├── gmail_client.py
│   └── yfinance_client.py
│
├── analysis/               # Technical analysis utilities
│   └── technicals.py
│
├── state/                  # Data models & graph state
│   ├── models.py          # Holding, PortfolioSummary, etc.
│   └── graph_state.py     # PortfolioState for LangGraph
│
├── main.py                # Entry point & graph orchestration
├── mlx_wrapper.py         # MLX LLM wrapper for Qwen-style models
├── mock_data.py           # Test data for offline development
│
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (git-ignored)
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Key Components

### Agents

1. **Agent 1: Portfolio Ingestion**
   - Fetches live holdings from Robinhood and Fidelity

2. **Agent 3: Market Data**
   - Retrieves price history (yfinance)
   - Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)

3. **Agent 9: LLM Analysis**
   - Analyzes portfolio using Qwen2.5-14B
   - Generates investment insights
   - Formats analysis in HTML

4. **Agent 10: Report Delivery**
   - Sends HTML-formatted email report via Gmail MCP

### MLX Wrapper

Custom wrapper (`mlx_wrapper.py`) for running Qwen2.5 LLMs efficiently on Apple Silicon:
- Lazy loads model + tokenizer
- Handles chat templates and generation
- Supports streaming output

### Technical Analysis

Module (`analysis/technicals.py`):
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Simple/Exponential Moving Averages

## Configuration Details

### MLX Models Available

- `mlx-community/Qwen2.5-14B-Instruct-4bit` (Default - ~8GB VRAM)
- `mlx-community/Qwen2.5-7B-Instruct-4bit` (~4GB VRAM - lighter)
- `mlx-community/Qwen2.5-32B-Instruct-4bit` (~16GB VRAM - most capable)

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ROBINHOOD_MCP_VENV` | Yes | — | Path to robinhood-agent venv |
| `ROBINHOOD_MCP_SERVER` | Yes | — | Path to robinhood-agent server.py |
| `FIDELITY_MCP_VENV` | Yes | — | Path to fidelity-agent venv |
| `FIDELITY_MCP_SERVER` | Yes | — | Path to fidelity-agent server.py |
| `GMAIL_MCP_VENV` | Yes | — | Path to gmail-agent venv |
| `GMAIL_MCP_SERVER` | Yes | — | Path to gmail-agent server.py |
| `REPORT_RECIPIENT` | Yes | — | Email address for report delivery |
| `MLX_MODEL_PATH` | No | `mlx-community/Qwen2.5-14B-Instruct-4bit` | Qwen model to use |
| `MLX_MAX_TOKENS` | No | `4096` | Max tokens for LLM generation |

## Security

⚠️ **Important:**
- `.env` file contains sensitive paths and credentials — **never commit to git**
- `.gitignore` prevents accidental credential leaks
- Credentials are managed by separate MCP servers (not stored here)
- Gmail uses app-specific passwords (not main account password)

## Testing

Run unit tests:
```bash
python test_yfinance.py
python test_technicals.py
python test_agent_03.py
```

## Development

### Adding New Agents
1. Create agent module in `agents/agent_XX_name.py`
2. Implement `run(state: PortfolioState) → PortfolioState` function
3. Add to graph in `main.py`

### Extending Technical Analysis
Add functions to `analysis/technicals.py`:
```python
def calculate_custom_indicator(prices: list[float]) -> list[float]:
    # Your analysis here
    return values
```

### Modifying LLM Behavior
Edit prompt in `agents/agent_09_llm_analysis.py` or switch model in `.env`

## Troubleshooting

### "MLX not installed" / Mac Intel Only
```
MLX is Apple Silicon only. For Intel Macs, use CPU-based LLMs:
- Use ollama + open-source models
- Switch to cloud-based LLM APIs (OpenAI, Claude, etc.)
```

### "Model download taking forever"
```
First model download (~5-8GB) happens on first run.
Set MLX_CACHE_DIR environment variable to control download location:
export MLX_CACHE_DIR=/custom/path
```

### "Insufficient GPU Memory"
```
Switch to smaller model:
python main.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

### "MCP Server connection failed"
```
Ensure all MCP server paths in .env are correct and servers are running:
- Check ROBINHOOD_MCP_SERVER, FIDELITY_MCP_SERVER, GMAIL_MCP_SERVER paths
- Verify venv paths exist and are activated
```

## Dependencies

- **langgraph** (≥0.2.0) — Agent orchestration framework
- **langchain-core** (≥0.3.0) — LLM/chain abstractions
- **mcp** (≥1.0.0) — Model Context Protocol client
- **mlx** (≥0.22.0) — Apple Silicon ML framework
- **mlx-lm** (≥0.21.0) — LLM utilities for MLX
- **pydantic** (≥2.6.0) — Data validation
- **python-dotenv** — Environment variable loading

## License

[Add your license here]

## Author

Naveen Krishnan  
[Add contact info]

## Contributing

[Add contribution guidelines]

---

**Last Updated:** February 27, 2026  
**Tested On:** macOS 13+ on Apple Silicon