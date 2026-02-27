"""
Technical indicator calculations — pure pandas/numpy, no TA-Lib dependency.

Indicators computed per ticker:
  - SMA 50 / SMA 200
  - RSI 14
  - MACD (12, 26, 9) — line, signal, histogram
  - Bollinger Bands (20, 2σ) — upper, mid, lower, %B

Input:  dict[ticker, DataFrame] from yfinance_client.fetch_price_history()
Output: dict[ticker, TechnicalSnapshot] dataclass
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TechnicalSnapshot:
    ticker: str

    # Price context
    close: float
    prev_close: float            # 1 trading day ago
    week_ago_close: float        # ~5 bars
    month_ago_close: float       # ~21 bars
    year_ago_close: float        # ~252 bars

    # Moving averages
    sma_50: Optional[float]
    sma_200: Optional[float]

    # RSI (14-period)
    rsi: Optional[float]

    # MACD (12, 26, 9)
    macd_line: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]

    # Bollinger Bands (20, 2σ)
    bb_upper: Optional[float]
    bb_mid: Optional[float]
    bb_lower: Optional[float]
    bb_pct_b: Optional[float]    # position within bands: 0=at lower, 1=at upper

    # Derived signals (human-readable)
    trend: str                   # "uptrend" | "downtrend" | "sideways"
    rsi_signal: str              # "overbought" | "oversold" | "neutral"
    macd_signal_str: str         # "bullish" | "bearish" | "neutral"

    def pct_change(self, past_close: float) -> float:
        if past_close == 0:
            return 0.0
        return (self.close - past_close) / past_close * 100

    @property
    def change_1d(self) -> float:
        return self.pct_change(self.prev_close)

    @property
    def change_1w(self) -> float:
        return self.pct_change(self.week_ago_close)

    @property
    def change_1m(self) -> float:
        return self.pct_change(self.month_ago_close)

    @property
    def change_1y(self) -> float:
        return self.pct_change(self.year_ago_close)

    def summary(self) -> str:
        """One-paragraph text summary suitable for LLM context."""
        lines = [
            f"{self.ticker} closed at ${self.close:.2f}.",
            f"Performance: 1d {self.change_1d:+.1f}%  1w {self.change_1w:+.1f}%  "
            f"1m {self.change_1m:+.1f}%  1y {self.change_1y:+.1f}%.",
        ]
        if self.sma_50 and self.sma_200:
            rel50  = (self.close - self.sma_50)  / self.sma_50  * 100
            rel200 = (self.close - self.sma_200) / self.sma_200 * 100
            lines.append(
                f"Moving averages: SMA50=${self.sma_50:.2f} ({rel50:+.1f}%), "
                f"SMA200=${self.sma_200:.2f} ({rel200:+.1f}%). Trend: {self.trend}."
            )
        if self.rsi is not None:
            lines.append(f"RSI(14)={self.rsi:.1f} ({self.rsi_signal}).")
        if self.macd_line is not None:
            lines.append(
                f"MACD: line={self.macd_line:.3f}, signal={self.macd_signal:.3f}, "
                f"hist={self.macd_hist:.3f} ({self.macd_signal_str})."
            )
        if self.bb_pct_b is not None:
            lines.append(
                f"Bollinger Bands: upper=${self.bb_upper:.2f}, mid=${self.bb_mid:.2f}, "
                f"lower=${self.bb_lower:.2f}, %B={self.bb_pct_b:.2f}."
            )
        return " ".join(lines)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _sma(series: pd.Series, window: int) -> Optional[float]:
    if len(series) < window:
        return None
    return float(series.rolling(window).mean().iloc[-1])


def _rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if not np.isnan(val) else None


def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    if len(close) < slow + signal:
        return None, None, None
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    sig_line   = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - sig_line
    return float(macd_line.iloc[-1]), float(sig_line.iloc[-1]), float(histogram.iloc[-1])


def _bollinger(close: pd.Series, window=20, num_std=2):
    if len(close) < window:
        return None, None, None, None
    rolling = close.rolling(window)
    mid     = rolling.mean()
    std     = rolling.std()
    upper   = mid + num_std * std
    lower   = mid - num_std * std
    c, u, m, l = (
        float(close.iloc[-1]),
        float(upper.iloc[-1]),
        float(mid.iloc[-1]),
        float(lower.iloc[-1]),
    )
    pct_b = (c - l) / (u - l) if (u - l) != 0 else 0.5
    return u, m, l, pct_b


def _safe_idx(series: pd.Series, n: int) -> float:
    """Return series.iloc[-n] if available, else series.iloc[0]."""
    if len(series) >= n:
        return float(series.iloc[-n])
    return float(series.iloc[0])


# ── Main entry point ──────────────────────────────────────────────────────────

def compute(history: dict[str, pd.DataFrame]) -> dict[str, TechnicalSnapshot]:
    """
    Compute technical indicators for each ticker.

    Args:
        history: output of yfinance_client.fetch_price_history()

    Returns:
        { ticker: TechnicalSnapshot }
    """
    result: dict[str, TechnicalSnapshot] = {}

    for ticker, df in history.items():
        close = df["Close"].astype(float)

        sma50  = _sma(close, 50)
        sma200 = _sma(close, 200)
        rsi    = _rsi(close)
        macd_l, macd_s, macd_h = _macd(close)
        bb_u, bb_m, bb_l, bb_pct = _bollinger(close)

        # Trend: price vs SMA50 vs SMA200
        if sma50 and sma200:
            c = float(close.iloc[-1])
            if c > sma50 > sma200:
                trend = "uptrend"
            elif c < sma50 < sma200:
                trend = "downtrend"
            else:
                trend = "sideways"
        else:
            trend = "sideways"

        # RSI signal
        if rsi is None:
            rsi_signal = "neutral"
        elif rsi >= 70:
            rsi_signal = "overbought"
        elif rsi <= 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"

        # MACD signal
        if macd_l is None:
            macd_signal_str = "neutral"
        elif macd_l > macd_s:
            macd_signal_str = "bullish"
        else:
            macd_signal_str = "bearish"

        result[ticker] = TechnicalSnapshot(
            ticker=ticker,
            close=float(close.iloc[-1]),
            prev_close=_safe_idx(close, 2),
            week_ago_close=_safe_idx(close, 6),
            month_ago_close=_safe_idx(close, 22),
            year_ago_close=_safe_idx(close, 252),
            sma_50=sma50,
            sma_200=sma200,
            rsi=rsi,
            macd_line=macd_l,
            macd_signal=macd_s,
            macd_hist=macd_h,
            bb_upper=bb_u,
            bb_mid=bb_m,
            bb_lower=bb_l,
            bb_pct_b=bb_pct,
            trend=trend,
            rsi_signal=rsi_signal,
            macd_signal_str=macd_signal_str,
        )

    return result
