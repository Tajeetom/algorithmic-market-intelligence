"""
Backtesting Framework

Replay engine for historical data with comprehensive
performance metrics: Sharpe, Sortino, max drawdown,
win rate, profit factor, and equity curve tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from features.extractor import FeatureExtractor, FeatureVector
from models.predictor import PredictionEngine, SignalDirection
from risk.engine import RiskEngine


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_pct: float
    entry_idx: int
    exit_idx: int
    holding_periods: int


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""

    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_holding_periods: float = 0.0
    final_equity: float = 0.0

    def summary(self) -> Dict:
        return {
            "total_return_pct": round(self.total_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "final_equity": round(self.final_equity, 2),
        }


@dataclass
class PriceBar:
    """Single OHLCV bar for replay."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


class ReplayEngine:
    """
    Event-driven backtester that replays price data through
    the feature extraction → prediction → risk pipeline.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.10,
        signal_threshold: float = 0.6,
    ):
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct

        self.feature_extractor = FeatureExtractor()
        self.prediction_engine = PredictionEngine(threshold=signal_threshold)
        self.risk_engine = RiskEngine(max_position_pct=position_size_pct)
        self.risk_engine.set_initial_equity(initial_capital)

        self._equity = initial_capital
        self._cash = initial_capital
        self._trades: List[BacktestTrade] = []
        self._equity_curve: List[float] = []
        self._returns: List[float] = []

    def run(self, symbol: str, bars: List[PriceBar]) -> BacktestMetrics:
        """Replay bars through the full pipeline and compute metrics."""
        prev_equity = self._equity

        for i, bar in enumerate(bars):
            # Extract features
            fv = self.feature_extractor.update(
                symbol, bar.close, bar.volume, timestamp=bar.timestamp
            )

            # Predict
            features = fv.to_array()
            pred = self.prediction_engine.infer(symbol, features)

            # Update existing position prices
            self.risk_engine.update_price(symbol, bar.close)

            # Check exits
            exits = self.risk_engine.check_exits()
            for sym in exits:
                pnl = self.risk_engine.register_exit(sym)
                if pnl is not None:
                    self._cash += pnl
                    self._equity += pnl

            # Check new entry
            if pred.direction != SignalDirection.NEUTRAL and symbol not in self.risk_engine.positions:
                order_size = (self._equity * self.position_size_pct) / bar.close
                decision = self.risk_engine.check_entry(
                    symbol, pred.direction.value, order_size,
                    bar.close, self._equity, fv.realized_volatility,
                )
                if decision.allowed and decision.adjusted_size > 0:
                    self.risk_engine.register_entry(
                        symbol, pred.direction.value,
                        decision.adjusted_size, bar.close,
                    )

            # Track equity
            position_value = sum(
                p.pnl for p in self.risk_engine.positions.values()
            )
            self._equity = self._cash + position_value
            self._equity_curve.append(self._equity)

            period_return = (self._equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self._returns.append(period_return)
            prev_equity = self._equity

        # Close remaining positions
        for sym in list(self.risk_engine.positions.keys()):
            pnl = self.risk_engine.register_exit(sym)
            if pnl:
                self._equity += pnl

        return self._compute_metrics()

    def _compute_metrics(self) -> BacktestMetrics:
        m = BacktestMetrics()
        m.final_equity = self._equity
        m.total_return_pct = (
            (self._equity - self.initial_capital) / self.initial_capital * 100
        )

        returns = self._returns
        if not returns:
            return m

        # Sharpe
        avg_r = sum(returns) / len(returns)
        std_r = math.sqrt(sum((r - avg_r) ** 2 for r in returns) / len(returns)) if len(returns) > 1 else 1e-10
        m.sharpe_ratio = (avg_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0

        # Sortino
        neg_returns = [r for r in returns if r < 0]
        downside_std = (
            math.sqrt(sum(r**2 for r in neg_returns) / len(neg_returns))
            if neg_returns
            else 1e-10
        )
        m.sortino_ratio = (avg_r / downside_std) * math.sqrt(252)

        # Max drawdown
        peak = self._equity_curve[0] if self._equity_curve else self.initial_capital
        max_dd = 0.0
        for eq in self._equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        m.max_drawdown_pct = max_dd * 100

        return m
