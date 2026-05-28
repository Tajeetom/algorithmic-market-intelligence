"""
Test Suite — Algorithmic Market Intelligence Platform

Covers feature extraction, prediction engine, risk engine,
backtesting metrics, storage, and observability.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.extractor import FeatureExtractor, FeatureVector
from models.predictor import EnsemblePredictor, PredictionEngine, SignalDirection
from risk.engine import RiskEngine, RiskDecision
from observability.metrics import MetricsRegistry
from storage.database import PlatformStore
from ingestion.market_feed import CircuitBreaker, TickBuffer, MarketTick


# ── Feature Extraction ───────────────────────────────────────────────────


class TestFeatureExtractor:

    def test_single_tick_returns_vector(self) -> None:
        ext = FeatureExtractor()
        fv = ext.update("TEST", 100.0, 1000.0)
        assert fv.symbol == "TEST"
        assert isinstance(fv.to_array(), list)
        assert len(fv.to_array()) == 18

    def test_rsi_bounded(self) -> None:
        ext = FeatureExtractor(rsi_period=5)
        for i in range(30):
            fv = ext.update("TEST", 100 + i * 0.5, 1000.0)
        assert 0 <= fv.rsi <= 100

    def test_ema_tracks_price(self) -> None:
        ext = FeatureExtractor(window=20)
        for i in range(50):
            fv = ext.update("TEST", 100 + i, 1000.0)
        assert fv.ema_short > 100
        assert fv.ema_long > 100
        assert fv.ema_short > fv.ema_long  # trending up

    def test_macd_computation(self) -> None:
        ext = FeatureExtractor()
        for i in range(50):
            fv = ext.update("TEST", 100 + math.sin(i * 0.3) * 5, 1000.0)
        assert fv.macd != 0 or fv.macd_signal != 0

    def test_zscore_centered(self) -> None:
        ext = FeatureExtractor()
        for i in range(50):
            fv = ext.update("TEST", 100.0, 1000.0)
        assert abs(fv.rolling_zscore) < 0.01

    def test_volume_features(self) -> None:
        ext = FeatureExtractor()
        volumes = [100, 200, 150, 300, 250, 400]
        for i, v in enumerate(volumes):
            fv = ext.update("TEST", 100.0, float(v))
        assert fv.volume_delta != 0

    def test_feature_names_match_array(self) -> None:
        assert len(FeatureVector.feature_names()) == 18
        fv = FeatureVector(symbol="X")
        assert len(fv.to_array()) == len(FeatureVector.feature_names())


# ── Prediction Engine ────────────────────────────────────────────────────


class TestPredictionEngine:

    def test_ensemble_returns_result(self) -> None:
        engine = PredictionEngine(threshold=0.3)
        features = [0.01] * 18
        result = engine.infer("TEST", features)
        assert result.symbol == "TEST"
        assert result.latency_ms >= 0
        assert result.direction in SignalDirection

    def test_neutral_below_threshold(self) -> None:
        engine = PredictionEngine(threshold=0.99)
        features = [0.0] * 18
        result = engine.infer("TEST", features)
        assert result.direction == SignalDirection.NEUTRAL

    def test_short_features_handled(self) -> None:
        engine = PredictionEngine()
        result = engine.infer("TEST", [0.1, 0.2])
        assert result.confidence == 0.0


# ── Risk Engine ──────────────────────────────────────────────────────────


class TestRiskEngine:

    def test_entry_approved(self) -> None:
        risk = RiskEngine(max_position_pct=0.1, max_open_positions=5)
        risk.set_initial_equity(10000)
        decision = risk.check_entry("BTC", "long", 1.0, 100.0, 10000)
        assert decision.allowed

    def test_duplicate_rejected(self) -> None:
        risk = RiskEngine()
        risk.set_initial_equity(10000)
        risk.register_entry("BTC", "long", 1.0, 100.0)
        decision = risk.check_entry("BTC", "long", 1.0, 100.0, 10000)
        assert not decision.allowed

    def test_max_positions_enforced(self) -> None:
        risk = RiskEngine(max_open_positions=2)
        risk.set_initial_equity(10000)
        risk.register_entry("A", "long", 1.0, 10.0)
        risk.register_entry("B", "long", 1.0, 10.0)
        decision = risk.check_entry("C", "long", 1.0, 10.0, 10000)
        assert not decision.allowed

    def test_position_sizing_cap(self) -> None:
        risk = RiskEngine(max_position_pct=0.05)
        risk.set_initial_equity(1000)
        decision = risk.check_entry("X", "long", 100.0, 10.0, 1000)
        assert decision.allowed
        assert decision.adjusted_size < 100.0

    def test_stop_loss_exit(self) -> None:
        risk = RiskEngine(stop_loss_pct=-5.0)
        risk.register_entry("X", "long", 10.0, 100.0)
        risk.update_price("X", 90.0)  # -10%
        exits = risk.check_exits()
        assert "X" in exits

    def test_take_profit_exit(self) -> None:
        risk = RiskEngine(take_profit_pct=10.0)
        risk.register_entry("X", "long", 10.0, 100.0)
        risk.update_price("X", 115.0)  # +15%
        exits = risk.check_exits()
        assert "X" in exits

    def test_portfolio_summary(self) -> None:
        risk = RiskEngine()
        risk.set_initial_equity(10000)
        risk.register_entry("A", "long", 10.0, 50.0)
        risk.update_price("A", 55.0)
        summary = risk.portfolio_summary(10000)
        assert summary["open_positions"] == 1
        assert summary["total_exposure_usd"] > 0


# ── Circuit Breaker ──────────────────────────────────────────────────────


class TestCircuitBreaker:

    def test_closed_by_default(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        assert not cb.is_open

    def test_opens_after_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open

    def test_resets_on_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open


# ── Tick Buffer ──────────────────────────────────────────────────────────


class TestTickBuffer:

    def test_ring_buffer(self) -> None:
        buf = TickBuffer(maxlen=5)
        for i in range(10):
            buf.append(MarketTick(symbol="X", price_usd=float(i), volume_usd=0, liquidity_usd=0, change_pct=0))
        assert len(buf) == 5
        assert buf.prices == [5.0, 6.0, 7.0, 8.0, 9.0]


# ── Observability ────────────────────────────────────────────────────────


class TestMetrics:

    def test_counter(self) -> None:
        reg = MetricsRegistry()
        reg.inc("test_counter", 5)
        reg.inc("test_counter", 3)
        assert reg.get_counter("test_counter") == 8

    def test_gauge(self) -> None:
        reg = MetricsRegistry()
        reg.set_gauge("cpu", 75.5)
        assert reg.get_gauge("cpu") == 75.5

    def test_histogram(self) -> None:
        reg = MetricsRegistry()
        for v in [1, 2, 3, 4, 5]:
            reg.observe("latency", float(v))
        stats = reg.histogram_stats("latency")
        assert stats["count"] == 5
        assert stats["mean"] == 3.0

    def test_export(self, tmp_path: Path) -> None:
        reg = MetricsRegistry(export_path=str(tmp_path / "m.json"))
        reg.inc("x")
        reg.export()
        data = json.loads((tmp_path / "m.json").read_text())
        assert "counters" in data


# ── Storage ──────────────────────────────────────────────────────────────


class TestStorage:

    def test_tick_roundtrip(self, tmp_path: Path) -> None:
        store = PlatformStore(str(tmp_path / "test.db"))
        store.store_tick("BTC", 50000.0, 1e6)
        ticks = store.get_ticks("BTC")
        assert len(ticks) == 1
        assert ticks[0]["price_usd"] == 50000.0
        store.close()

    def test_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        store = PlatformStore(str(tmp_path / "test.db"))
        store.save_checkpoint("state", {"tick": 42, "status": "ok"})
        loaded = store.load_checkpoint("state")
        assert loaded["tick"] == 42
        store.close()

    def test_missing_checkpoint(self, tmp_path: Path) -> None:
        store = PlatformStore(str(tmp_path / "test.db"))
        assert store.load_checkpoint("nope") is None
        store.close()
