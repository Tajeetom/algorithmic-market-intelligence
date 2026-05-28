"""
Streaming Feature Extraction Engine

Real-time statistical feature computation over streaming market data:
RSI, EMA, MACD, VWAP, rolling z-score, momentum decay, realized
volatility, volume acceleration, and order flow imbalance.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class FeatureVector:
    """Computed feature set for a single observation window."""

    symbol: str
    timestamp: float = 0.0

    # Price features
    log_return: float = 0.0
    rolling_mean: float = 0.0
    rolling_std: float = 0.0
    rolling_zscore: float = 0.0

    # Trend indicators
    rsi: float = 50.0
    ema_short: float = 0.0
    ema_long: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0

    # Volume features
    vwap: float = 0.0
    volume_delta: float = 0.0
    volume_acceleration: float = 0.0

    # Volatility & momentum
    realized_volatility: float = 0.0
    momentum: float = 0.0
    momentum_decay: float = 0.0

    # Microstructure
    spread_compression: float = 0.0
    liquidity_imbalance: float = 0.0

    def to_array(self) -> List[float]:
        """Convert to flat numeric array for model input."""
        return [
            self.log_return, self.rolling_mean, self.rolling_std,
            self.rolling_zscore, self.rsi, self.ema_short, self.ema_long,
            self.macd, self.macd_signal, self.macd_histogram,
            self.vwap, self.volume_delta, self.volume_acceleration,
            self.realized_volatility, self.momentum, self.momentum_decay,
            self.spread_compression, self.liquidity_imbalance,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "log_return", "rolling_mean", "rolling_std", "rolling_zscore",
            "rsi", "ema_short", "ema_long", "macd", "macd_signal",
            "macd_histogram", "vwap", "volume_delta", "volume_acceleration",
            "realized_volatility", "momentum", "momentum_decay",
            "spread_compression", "liquidity_imbalance",
        ]


class FeatureExtractor:
    """
    Stateful streaming feature extractor.

    Maintains internal ring buffers per symbol and computes
    features incrementally on each new tick.
    """

    def __init__(
        self,
        window: int = 50,
        rsi_period: int = 14,
        ema_short: int = 12,
        ema_long: int = 26,
        macd_signal_period: int = 9,
        zscore_window: int = 20,
        vwap_window: int = 20,
    ):
        self.window = window
        self.rsi_period = rsi_period
        self.ema_short_period = ema_short
        self.ema_long_period = ema_long
        self.macd_signal_period = macd_signal_period
        self.zscore_window = zscore_window
        self.vwap_window = vwap_window

        self._prices: Dict[str, Deque[float]] = {}
        self._volumes: Dict[str, Deque[float]] = {}
        self._liquidities: Dict[str, Deque[float]] = {}
        self._timestamps: Dict[str, Deque[float]] = {}

        # EMA state
        self._ema_short_val: Dict[str, float] = {}
        self._ema_long_val: Dict[str, float] = {}
        self._macd_signal_val: Dict[str, float] = {}

        # RSI state
        self._avg_gain: Dict[str, float] = {}
        self._avg_loss: Dict[str, float] = {}

    def _ensure_buffers(self, symbol: str) -> None:
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self.window)
            self._volumes[symbol] = deque(maxlen=self.window)
            self._liquidities[symbol] = deque(maxlen=self.window)
            self._timestamps[symbol] = deque(maxlen=self.window)

    def update(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0,
        liquidity: float = 0.0,
        timestamp: float = 0.0,
    ) -> FeatureVector:
        """Ingest a tick and compute the latest feature vector."""
        self._ensure_buffers(symbol)

        self._prices[symbol].append(price)
        self._volumes[symbol].append(volume)
        self._liquidities[symbol].append(liquidity)
        self._timestamps[symbol].append(timestamp)

        prices = list(self._prices[symbol])
        volumes = list(self._volumes[symbol])
        n = len(prices)

        fv = FeatureVector(symbol=symbol, timestamp=timestamp)

        if n < 2:
            fv.ema_short = price
            fv.ema_long = price
            self._ema_short_val[symbol] = price
            self._ema_long_val[symbol] = price
            return fv

        # ── Log return ───────────────────────────────────────────────
        fv.log_return = math.log(prices[-1] / prices[-2]) if prices[-2] > 0 else 0.0

        # ── Rolling stats ────────────────────────────────────────────
        win = prices[-self.zscore_window:] if n >= self.zscore_window else prices
        fv.rolling_mean = sum(win) / len(win)
        variance = sum((p - fv.rolling_mean) ** 2 for p in win) / len(win)
        fv.rolling_std = math.sqrt(variance) if variance > 0 else 1e-10
        fv.rolling_zscore = (prices[-1] - fv.rolling_mean) / fv.rolling_std

        # ── RSI ──────────────────────────────────────────────────────
        fv.rsi = self._compute_rsi(symbol, prices)

        # ── EMA ──────────────────────────────────────────────────────
        fv.ema_short = self._update_ema(
            symbol, "_ema_short_val", price, self.ema_short_period
        )
        fv.ema_long = self._update_ema(
            symbol, "_ema_long_val", price, self.ema_long_period
        )

        # ── MACD ─────────────────────────────────────────────────────
        fv.macd = fv.ema_short - fv.ema_long
        fv.macd_signal = self._update_ema(
            symbol, "_macd_signal_val", fv.macd, self.macd_signal_period
        )
        fv.macd_histogram = fv.macd - fv.macd_signal

        # ── VWAP ─────────────────────────────────────────────────────
        vw_prices = prices[-self.vwap_window:]
        vw_volumes = volumes[-self.vwap_window:]
        total_vol = sum(vw_volumes)
        fv.vwap = (
            sum(p * v for p, v in zip(vw_prices, vw_volumes)) / total_vol
            if total_vol > 0
            else prices[-1]
        )

        # ── Volume features ──────────────────────────────────────────
        if n >= 2:
            fv.volume_delta = volumes[-1] - volumes[-2]
        if n >= 3:
            prev_delta = volumes[-2] - volumes[-3]
            fv.volume_acceleration = fv.volume_delta - prev_delta

        # ── Realized volatility ──────────────────────────────────────
        returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(max(1, n - self.zscore_window), n)
            if prices[i - 1] > 0
        ]
        if returns:
            mean_r = sum(returns) / len(returns)
            fv.realized_volatility = math.sqrt(
                sum((r - mean_r) ** 2 for r in returns) / len(returns)
            )

        # ── Momentum ─────────────────────────────────────────────────
        lookback = min(10, n - 1)
        if lookback > 0 and prices[-(lookback + 1)] > 0:
            fv.momentum = (prices[-1] / prices[-(lookback + 1)]) - 1.0
            fv.momentum_decay = fv.momentum * (0.95**lookback)

        # ── Liquidity imbalance ──────────────────────────────────────
        liqs = list(self._liquidities[symbol])
        if len(liqs) >= 2:
            avg_liq = sum(liqs) / len(liqs)
            fv.liquidity_imbalance = (
                (liqs[-1] - avg_liq) / avg_liq if avg_liq > 0 else 0.0
            )

        return fv

    def _compute_rsi(self, symbol: str, prices: List[float]) -> float:
        n = len(prices)
        if n < self.rsi_period + 1:
            return 50.0

        change = prices[-1] - prices[-2]
        gain = max(change, 0.0)
        loss = abs(min(change, 0.0))

        if symbol not in self._avg_gain:
            # Initial RSI computation
            gains = []
            losses = []
            for i in range(1, min(self.rsi_period + 1, n)):
                d = prices[-(i)] - prices[-(i + 1)] if i + 1 <= n else 0.0
                gains.append(max(d, 0.0))
                losses.append(abs(min(d, 0.0)))
            self._avg_gain[symbol] = sum(gains) / len(gains) if gains else 0.0
            self._avg_loss[symbol] = sum(losses) / len(losses) if losses else 0.0
        else:
            p = self.rsi_period
            self._avg_gain[symbol] = (self._avg_gain[symbol] * (p - 1) + gain) / p
            self._avg_loss[symbol] = (self._avg_loss[symbol] * (p - 1) + loss) / p

        ag = self._avg_gain[symbol]
        al = self._avg_loss[symbol]
        if al == 0:
            return 100.0
        rs = ag / al
        return 100.0 - (100.0 / (1.0 + rs))

    def _update_ema(
        self, symbol: str, attr: str, value: float, period: int
    ) -> float:
        store = getattr(self, attr)
        if symbol not in store:
            store[symbol] = value
            return value
        k = 2.0 / (period + 1.0)
        new_val = value * k + store[symbol] * (1.0 - k)
        store[symbol] = new_val
        return new_val
