"""
Risk Management Engine

Position sizing, exposure management, volatility throttling,
drawdown protection, and stop-loss enforcement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionState:
    symbol: str
    side: str  # "long" | "short"
    qty: float
    entry_price: float
    current_price: float = 0.0
    opened_at: float = field(default_factory=time.time)

    @property
    def notional(self) -> float:
        return self.qty * self.current_price

    @property
    def pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.qty
        return (self.entry_price - self.current_price) * self.qty

    @property
    def pnl_pct(self) -> float:
        cost = self.entry_price * self.qty
        return (self.pnl / cost * 100.0) if cost > 0 else 0.0


@dataclass
class RiskDecision:
    allowed: bool
    reason: str
    adjusted_size: float = 0.0
    risk_score: float = 0.0


class RiskEngine:
    """
    Pre-trade and in-flight risk checks.

    Enforces position limits, portfolio exposure caps,
    volatility throttling, drawdown circuit breakers,
    and dynamic position sizing.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_drawdown_pct: float = 0.15,
        max_open_positions: int = 5,
        volatility_threshold: float = 0.05,
        daily_loss_limit_pct: float = 0.05,
        stop_loss_pct: float = -5.0,
        take_profit_pct: float = 10.0,
    ):
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_open_positions = max_open_positions
        self.volatility_threshold = volatility_threshold
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self._positions: Dict[str, PositionState] = {}
        self._peak_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._initial_equity: float = 0.0

    def set_initial_equity(self, equity: float) -> None:
        self._initial_equity = equity
        self._peak_equity = equity

    @property
    def positions(self) -> Dict[str, PositionState]:
        return self._positions

    def check_entry(
        self,
        symbol: str,
        side: str,
        proposed_size: float,
        price: float,
        equity: float,
        volatility: float = 0.0,
    ) -> RiskDecision:
        """Pre-trade risk check before opening a new position."""

        # Position count limit
        if len(self._positions) >= self.max_open_positions:
            return RiskDecision(
                allowed=False,
                reason=f"Max open positions reached ({self.max_open_positions}).",
            )

        # Duplicate check
        if symbol in self._positions:
            return RiskDecision(
                allowed=False,
                reason=f"Already holding position in {symbol}.",
            )

        # Position size cap
        max_notional = equity * self.max_position_pct
        proposed_notional = proposed_size * price
        if proposed_notional > max_notional:
            adjusted = max_notional / price if price > 0 else 0.0
            logger.info(
                "Position capped: %s size %.4f -> %.4f (max %.1f%% of equity).",
                symbol, proposed_size, adjusted, self.max_position_pct * 100,
            )
            proposed_size = adjusted

        # Volatility throttle
        if volatility > self.volatility_threshold:
            dampen = max(0.3, 1.0 - (volatility / self.volatility_threshold - 1.0))
            proposed_size *= dampen
            logger.info(
                "Volatility throttle: %s size dampened by %.0f%% (vol=%.4f).",
                symbol, (1 - dampen) * 100, volatility,
            )

        # Drawdown circuit breaker
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
        if drawdown >= self.max_drawdown_pct:
            return RiskDecision(
                allowed=False,
                reason=f"Drawdown limit hit ({drawdown:.1%} >= {self.max_drawdown_pct:.1%}).",
                risk_score=drawdown,
            )

        # Daily loss limit
        if self._initial_equity > 0:
            daily_loss = -self._daily_pnl / self._initial_equity
            if daily_loss >= self.daily_loss_limit_pct:
                return RiskDecision(
                    allowed=False,
                    reason=f"Daily loss limit hit ({daily_loss:.1%}).",
                    risk_score=daily_loss,
                )

        return RiskDecision(
            allowed=True,
            reason="Approved.",
            adjusted_size=proposed_size,
            risk_score=drawdown,
        )

    def register_entry(
        self, symbol: str, side: str, qty: float, price: float
    ) -> None:
        self._positions[symbol] = PositionState(
            symbol=symbol, side=side, qty=qty,
            entry_price=price, current_price=price,
        )

    def update_price(self, symbol: str, price: float) -> None:
        if symbol in self._positions:
            self._positions[symbol].current_price = price

    def check_exits(self) -> List[str]:
        """Return symbols that should be closed (stop-loss / take-profit)."""
        exits: List[str] = []
        for sym, pos in self._positions.items():
            if pos.pnl_pct <= self.stop_loss_pct:
                logger.info("Stop-loss triggered for %s (%.2f%%).", sym, pos.pnl_pct)
                exits.append(sym)
            elif pos.pnl_pct >= self.take_profit_pct:
                logger.info("Take-profit triggered for %s (%.2f%%).", sym, pos.pnl_pct)
                exits.append(sym)
        return exits

    def register_exit(self, symbol: str) -> Optional[float]:
        pos = self._positions.pop(symbol, None)
        if pos:
            self._daily_pnl += pos.pnl
            return pos.pnl
        return None

    def portfolio_summary(self, equity: float) -> Dict:
        total_exposure = sum(p.notional for p in self._positions.values())
        return {
            "open_positions": len(self._positions),
            "total_exposure_usd": total_exposure,
            "exposure_pct": total_exposure / equity * 100 if equity > 0 else 0,
            "peak_equity": self._peak_equity,
            "drawdown_pct": (
                (self._peak_equity - equity) / self._peak_equity * 100
                if self._peak_equity > 0
                else 0
            ),
            "daily_pnl": self._daily_pnl,
        }
