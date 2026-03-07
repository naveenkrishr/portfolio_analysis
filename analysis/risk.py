"""
RiskSnapshot — portfolio-level and per-ticker risk metrics.

Pure computation from price history + holdings weights.
No API calls — uses DataFrames from yfinance_client.

Metrics computed:
  Per-ticker:   beta (vs SPY), annualized volatility, max drawdown, Sharpe ratio
  Portfolio:    portfolio VaR (95%), CVaR (95%), Sharpe, concentration (HHI),
                correlation matrix, overall risk rating
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ── Per-ticker risk ───────────────────────────────────────────────────────────

@dataclass
class TickerRisk:
    ticker: str
    beta: Optional[float]               # vs SPY
    annualized_vol: Optional[float]      # annualized std dev of daily returns
    max_drawdown: Optional[float]        # worst peak-to-trough (negative %)
    sharpe: Optional[float]             # annualized Sharpe (rf=0 for simplicity)

    def summary(self) -> str:
        parts = [f"{self.ticker}:"]
        if self.beta is not None:
            parts.append(f"beta={self.beta:.2f}")
        if self.annualized_vol is not None:
            parts.append(f"vol={self.annualized_vol*100:.1f}%")
        if self.max_drawdown is not None:
            parts.append(f"max_dd={self.max_drawdown*100:.1f}%")
        if self.sharpe is not None:
            parts.append(f"Sharpe={self.sharpe:.2f}")
        return " | ".join(parts)


# ── Portfolio-level risk ──────────────────────────────────────────────────────

@dataclass
class PortfolioRiskSnapshot:
    # Per-ticker
    ticker_risks: dict[str, TickerRisk] = field(default_factory=dict)

    # Portfolio-level
    portfolio_var_95: Optional[float] = None      # 1-day VaR at 95% (as negative %)
    portfolio_cvar_95: Optional[float] = None     # Conditional VaR (Expected Shortfall)
    portfolio_vol: Optional[float] = None         # annualized portfolio volatility
    portfolio_sharpe: Optional[float] = None      # portfolio Sharpe ratio
    portfolio_beta: Optional[float] = None        # weighted portfolio beta

    # Concentration
    hhi: Optional[float] = None                   # Herfindahl-Hirschman Index (0-10000)
    top3_concentration: Optional[float] = None    # % of portfolio in top 3 positions
    concentration_rating: str = "unknown"         # WELL-DIVERSIFIED / MODERATE / CONCENTRATED / OVER-CONCENTRATED

    # Correlation highlights
    high_correlations: list[str] = field(default_factory=list)   # pairs with corr > 0.8
    low_correlations: list[str] = field(default_factory=list)    # pairs with corr < 0.2

    # Overall
    risk_rating: str = "unknown"  # CONSERVATIVE / BALANCED / AGGRESSIVE / OVER-CONCENTRATED

    def summary(self) -> str:
        """Text summary suitable for LLM context."""
        parts = ["PORTFOLIO RISK METRICS:"]

        if self.portfolio_var_95 is not None:
            parts.append(f"  1-day VaR(95%): {self.portfolio_var_95*100:.2f}%")
        if self.portfolio_cvar_95 is not None:
            parts.append(f"  1-day CVaR(95%): {self.portfolio_cvar_95*100:.2f}%")
        if self.portfolio_vol is not None:
            parts.append(f"  Annualized volatility: {self.portfolio_vol*100:.1f}%")
        if self.portfolio_sharpe is not None:
            parts.append(f"  Portfolio Sharpe: {self.portfolio_sharpe:.2f}")
        if self.portfolio_beta is not None:
            parts.append(f"  Portfolio beta: {self.portfolio_beta:.2f}")

        # Concentration
        parts.append(f"  Concentration (HHI): {self.hhi:.0f} — {self.concentration_rating}")
        if self.top3_concentration is not None:
            parts.append(f"  Top 3 positions: {self.top3_concentration*100:.1f}% of portfolio")

        # Correlation
        if self.high_correlations:
            parts.append(f"  Highly correlated pairs (>0.8): {'; '.join(self.high_correlations)}")
        if self.low_correlations:
            parts.append(f"  Diversifying pairs (<0.2): {'; '.join(self.low_correlations)}")

        parts.append(f"  Overall risk rating: {self.risk_rating}")

        # Per-ticker
        parts.append("\n  PER-TICKER RISK:")
        for tr in self.ticker_risks.values():
            parts.append(f"    {tr.summary()}")

        return "\n".join(parts)


# ── Computation ───────────────────────────────────────────────────────────────

def _daily_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from Close prices."""
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()


def _beta(ticker_returns: pd.Series, benchmark_returns: pd.Series) -> Optional[float]:
    """Compute beta vs benchmark using OLS."""
    aligned = pd.concat([ticker_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None
    aligned.columns = ["ticker", "benchmark"]
    cov = aligned["ticker"].cov(aligned["benchmark"])
    var = aligned["benchmark"].var()
    if var == 0:
        return None
    return float(cov / var)


def _max_drawdown(close: pd.Series) -> Optional[float]:
    """Compute maximum drawdown from a price series."""
    if len(close) < 2:
        return None
    peak = close.cummax()
    drawdown = (close - peak) / peak
    return float(drawdown.min())


def _sharpe(returns: pd.Series, rf_annual: float = 0.0) -> Optional[float]:
    """Annualized Sharpe ratio from daily returns."""
    if len(returns) < 30 or returns.std() == 0:
        return None
    rf_daily = rf_annual / 252
    excess = returns.mean() - rf_daily
    return float(excess / returns.std() * np.sqrt(252))


def _concentration_rating(hhi: float) -> str:
    """Rate concentration based on HHI."""
    if hhi < 1000:
        return "WELL-DIVERSIFIED"
    if hhi < 1800:
        return "MODERATE"
    if hhi < 3000:
        return "CONCENTRATED"
    return "OVER-CONCENTRATED"


def _overall_risk_rating(
    portfolio_beta: float | None,
    portfolio_vol: float | None,
    concentration: str,
) -> str:
    """Classify overall portfolio risk."""
    if concentration == "OVER-CONCENTRATED":
        return "OVER-CONCENTRATED"

    if portfolio_beta is not None:
        if portfolio_beta > 1.3:
            return "AGGRESSIVE"
        if portfolio_beta < 0.7:
            return "CONSERVATIVE"

    if portfolio_vol is not None:
        if portfolio_vol > 0.25:
            return "AGGRESSIVE"
        if portfolio_vol < 0.12:
            return "CONSERVATIVE"

    return "BALANCED"


def compute(
    price_history: dict[str, pd.DataFrame],
    weights: dict[str, float],
    benchmark_history: pd.DataFrame | None = None,
) -> PortfolioRiskSnapshot:
    """
    Compute portfolio risk metrics.

    Args:
        price_history: {ticker: DataFrame} with Close column (from yfinance)
        weights: {ticker: weight} where weight is fraction of portfolio (0-1)
        benchmark_history: DataFrame for SPY (or other benchmark) with Close column

    Returns:
        PortfolioRiskSnapshot with per-ticker and portfolio-level metrics
    """
    # Compute daily returns for each ticker
    ticker_returns: dict[str, pd.Series] = {}
    for ticker, df in price_history.items():
        if ticker in weights and not df.empty:
            ticker_returns[ticker] = _daily_returns(df)

    # Benchmark returns
    bench_ret = None
    if benchmark_history is not None and not benchmark_history.empty:
        bench_ret = _daily_returns(benchmark_history)

    # Per-ticker risk
    ticker_risks: dict[str, TickerRisk] = {}
    for ticker, ret in ticker_returns.items():
        df = price_history[ticker]
        b = _beta(ret, bench_ret) if bench_ret is not None else None
        ticker_risks[ticker] = TickerRisk(
            ticker=ticker,
            beta=b,
            annualized_vol=float(ret.std() * np.sqrt(252)) if len(ret) > 1 else None,
            max_drawdown=_max_drawdown(df["Close"]),
            sharpe=_sharpe(ret),
        )

    # Build portfolio return series (weighted sum)
    if ticker_returns:
        returns_df = pd.DataFrame(ticker_returns)
        # Align all return series
        returns_df = returns_df.dropna()

        if not returns_df.empty and len(returns_df) > 30:
            # Portfolio weights vector (only tickers with data)
            w = np.array([weights.get(t, 0) for t in returns_df.columns])
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum  # renormalize to available tickers

            portfolio_ret = returns_df.values @ w

            # VaR and CVaR (historical simulation)
            var_95 = float(np.percentile(portfolio_ret, 5))
            cvar_95 = float(portfolio_ret[portfolio_ret <= var_95].mean()) if (portfolio_ret <= var_95).any() else var_95

            port_vol = float(np.std(portfolio_ret) * np.sqrt(252))
            port_sharpe = _sharpe(pd.Series(portfolio_ret))

            # Portfolio beta (weighted average)
            port_beta = None
            if any(tr.beta is not None for tr in ticker_risks.values()):
                betas = []
                beta_weights = []
                for t in returns_df.columns:
                    if t in ticker_risks and ticker_risks[t].beta is not None:
                        betas.append(ticker_risks[t].beta)
                        beta_weights.append(weights.get(t, 0))
                if betas:
                    bw = np.array(beta_weights)
                    if bw.sum() > 0:
                        bw = bw / bw.sum()
                        port_beta = float(np.dot(betas, bw))
        else:
            var_95 = None
            cvar_95 = None
            port_vol = None
            port_sharpe = None
            port_beta = None

        # Correlation matrix highlights
        high_corr = []
        low_corr = []
        if len(returns_df.columns) > 1:
            corr_matrix = returns_df.corr()
            tickers_list = list(returns_df.columns)
            for i in range(len(tickers_list)):
                for j in range(i + 1, len(tickers_list)):
                    c = corr_matrix.iloc[i, j]
                    pair = f"{tickers_list[i]}/{tickers_list[j]} ({c:.2f})"
                    if c > 0.8:
                        high_corr.append(pair)
                    elif c < 0.2:
                        low_corr.append(pair)
    else:
        var_95 = None
        cvar_95 = None
        port_vol = None
        port_sharpe = None
        port_beta = None
        high_corr = []
        low_corr = []

    # Concentration metrics (use all holdings weights, not just those with price data)
    all_weights = list(weights.values())
    hhi = sum(w * w * 10000 for w in all_weights) if all_weights else 0
    sorted_weights = sorted(all_weights, reverse=True)
    top3 = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)

    conc_rating = _concentration_rating(hhi)
    risk_rating = _overall_risk_rating(port_beta, port_vol, conc_rating)

    return PortfolioRiskSnapshot(
        ticker_risks=ticker_risks,
        portfolio_var_95=var_95,
        portfolio_cvar_95=cvar_95,
        portfolio_vol=port_vol,
        portfolio_sharpe=port_sharpe,
        portfolio_beta=port_beta,
        hhi=hhi,
        top3_concentration=top3,
        concentration_rating=conc_rating,
        high_correlations=high_corr,
        low_correlations=low_corr,
        risk_rating=risk_rating,
    )
