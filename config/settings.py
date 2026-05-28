"""
Platform Configuration — Pydantic Settings

Typed, validated environment-variable loading for all subsystems:
ingestion, feature extraction, inference, execution, risk, and observability.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class PlatformSettings(BaseSettings):
    """Central configuration for the streaming prediction platform."""

    # ── General ──────────────────────────────────────────────────────────
    simulation_mode: bool = Field(default=True)
    initial_capital: float = Field(default=10000.0, ge=0)
    poll_interval: float = Field(default=2.0, ge=0.5)
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="data/platform.log")

    # ── Ingestion ────────────────────────────────────────────────────────
    solana_rpc_url: str = Field(default="https://api.mainnet-beta.solana.com")
    solana_rpc_api_key: str = Field(default="")
    dex_api_key: str = Field(default="")
    jupiter_api_key: str = Field(default="")
    cmc_api_key: str = Field(default="")
    dex_queries: List[str] = Field(
        default=["solana", "raydium", "jupiter", "bonk", "wif", "pyth"]
    )
    market_refresh_seconds: float = Field(default=30.0, ge=10)

    # ── Feature Engineering ──────────────────────────────────────────────
    feature_window: int = Field(default=50, ge=10)
    rsi_period: int = Field(default=14, ge=2)
    ema_short: int = Field(default=12, ge=2)
    ema_long: int = Field(default=26, ge=5)
    macd_signal: int = Field(default=9, ge=2)
    vwap_window: int = Field(default=20, ge=5)
    zscore_window: int = Field(default=20, ge=5)

    # ── Inference ────────────────────────────────────────────────────────
    model_path: str = Field(default="models/artifacts/model.onnx")
    inference_timeout_ms: int = Field(default=200, ge=10)
    signal_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # ── Execution ────────────────────────────────────────────────────────
    jupiter_live_enabled: bool = Field(default=False)
    order_size_sol: float = Field(default=0.02, ge=0)
    buy_slippage_bps: int = Field(default=500, ge=0)
    sell_profit_threshold_pct: float = Field(default=10.0)
    sell_stop_loss_pct: float = Field(default=-5.0)

    # ── Risk ─────────────────────────────────────────────────────────────
    max_position_pct: float = Field(default=0.10, ge=0, le=1.0)
    max_drawdown_pct: float = Field(default=0.15, ge=0, le=1.0)
    max_open_positions: int = Field(default=5, ge=1)
    volatility_throttle_threshold: float = Field(default=0.05, ge=0)
    daily_loss_limit_pct: float = Field(default=0.05, ge=0, le=1.0)

    # ── Storage ──────────────────────────────────────────────────────────
    db_path: str = Field(default="data/platform.db")
    checkpoint_dir: str = Field(default="data/checkpoints")

    # ── Observability ────────────────────────────────────────────────────
    metrics_export_path: str = Field(default="data/metrics.json")
    metrics_port: int = Field(default=9090, ge=1024)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log_level: {v}")
        return v

    @property
    def data_dir(self) -> Path:
        d = Path("data")
        d.mkdir(exist_ok=True)
        return d

    model_config = {
        "env_prefix": "AMP_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> PlatformSettings:
    return PlatformSettings()
