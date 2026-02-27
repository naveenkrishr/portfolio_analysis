"""
Hardcoded mock portfolio for Step 1 LLM prompt tuning.
Replace this with Agent 1 (Portfolio Ingestion) in Step 3.

Holdings reflect known accounts:
  Robinhood   — SNOW, QQMG (7 sh)
  Fidelity Z24 — VOO, QQMG (1111 sh), SPAXX
  Fidelity Z31 — VOO, FCASH
"""
from state.models import Holding

MOCK_HOLDINGS: list[Holding] = [
    # ETFs
    Holding(
        ticker="VOO",
        name="Vanguard S&P 500 ETF",
        shares=298.0,
        price=541.80,
        value=161_456.0,
        account="Fidelity",
        asset_type="etf",
    ),
    Holding(
        ticker="QQMG",
        name="Invesco NASDAQ 100 ETF",
        shares=1118.0,
        price=28.40,
        value=31_751.0,
        account="Fidelity+Robinhood",
        asset_type="etf",
    ),
    # Individual stock
    Holding(
        ticker="SNOW",
        name="Snowflake Inc",
        shares=25.0,
        price=148.50,
        value=3_713.0,
        account="Robinhood",
        asset_type="stock",
    ),
    # Cash / money market
    Holding(
        ticker="SPAXX",
        name="Fidelity Government Money Market Fund",
        shares=8_200.0,
        price=1.0,
        value=8_200.0,
        account="Fidelity",
        asset_type="cash",
    ),
    Holding(
        ticker="FCASH",
        name="Fidelity Cash",
        shares=1_800.0,
        price=1.0,
        value=1_800.0,
        account="Fidelity",
        asset_type="cash",
    ),
]
