"""
Prediction Engine — Multi-Model Inference

Supports LightGBM, XGBoost, and ONNX runtime for low-latency
signal generation from streaming feature vectors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class PredictionResult:
    """Output of the inference pipeline."""

    symbol: str
    direction: SignalDirection
    confidence: float
    latency_ms: float
    model_id: str
    features_used: int
    timestamp: float = 0.0


class BasePredictor:
    """Abstract base for all prediction models."""

    model_id: str = "base"

    def predict(self, features: List[float]) -> tuple[float, float]:
        """Return (signal_score, confidence) in [-1, 1] and [0, 1]."""
        raise NotImplementedError


class EnsemblePredictor(BasePredictor):
    """
    Ensemble of statistical heuristics for demonstration.

    In production, replace with trained LightGBM/XGBoost/ONNX models
    loaded from the training pipeline artifacts.
    """

    model_id = "ensemble_v1"

    def predict(self, features: List[float]) -> tuple[float, float]:
        if len(features) < 18:
            return 0.0, 0.0

        # Feature indices (from FeatureVector.to_array order)
        log_return = features[0]
        zscore = features[3]
        rsi = features[4]
        macd_hist = features[9]
        volume_accel = features[12]
        realized_vol = features[13]
        momentum = features[14]

        # Multi-factor signal
        trend_signal = (
            0.25 * (1.0 if macd_hist > 0 else -1.0)
            + 0.20 * max(-1.0, min(1.0, momentum * 10.0))
            + 0.15 * max(-1.0, min(1.0, -zscore / 2.0))
        )

        # RSI mean-reversion overlay
        if rsi > 70:
            trend_signal -= 0.2
        elif rsi < 30:
            trend_signal += 0.2

        # Volume confirmation
        if volume_accel > 0 and trend_signal > 0:
            trend_signal *= 1.15
        elif volume_accel < 0 and trend_signal < 0:
            trend_signal *= 1.15

        # Volatility dampening
        vol_penalty = min(realized_vol * 5.0, 0.3)
        trend_signal *= (1.0 - vol_penalty)

        signal = max(-1.0, min(1.0, trend_signal))
        confidence = min(abs(signal), 1.0)

        return signal, confidence


class OnnxPredictor(BasePredictor):
    """ONNX Runtime predictor for production-trained models."""

    model_id = "onnx_v1"

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._session = None
        self._load_model()

    def _load_model(self) -> None:
        if not Path(self._model_path).exists():
            logger.info(
                "ONNX model not found at %s — using fallback.", self._model_path
            )
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(self._model_path)
            logger.info("ONNX model loaded from %s", self._model_path)
        except ImportError:
            logger.warning("onnxruntime not installed — ONNX inference disabled.")
        except Exception:
            logger.exception("Failed to load ONNX model.")

    def predict(self, features: List[float]) -> tuple[float, float]:
        if self._session is None:
            return 0.0, 0.0

        import numpy as np

        input_name = self._session.get_inputs()[0].name
        x = np.array([features], dtype=np.float32)
        outputs = self._session.run(None, {input_name: x})
        prob = float(outputs[0][0][1]) if len(outputs[0][0]) > 1 else float(outputs[0][0][0])
        signal = prob * 2.0 - 1.0  # map [0,1] → [-1,1]
        return signal, abs(signal)


class PredictionEngine:
    """
    Orchestrates inference across one or more models,
    tracks latency, and generates trade signals.
    """

    def __init__(
        self,
        threshold: float = 0.6,
        model_path: str = "models/artifacts/model.onnx",
    ):
        self.threshold = threshold
        self._predictors: List[BasePredictor] = []

        # Try ONNX first, fall back to ensemble
        onnx = OnnxPredictor(model_path)
        if onnx._session is not None:
            self._predictors.append(onnx)
        self._predictors.append(EnsemblePredictor())

    def infer(self, symbol: str, features: List[float]) -> PredictionResult:
        """Run inference and return a prediction result."""
        t0 = time.perf_counter()

        best_signal, best_conf, best_model = 0.0, 0.0, "none"
        for predictor in self._predictors:
            signal, conf = predictor.predict(features)
            if conf > best_conf:
                best_signal, best_conf, best_model = signal, conf, predictor.model_id

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if best_conf >= self.threshold:
            direction = SignalDirection.LONG if best_signal > 0 else SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        return PredictionResult(
            symbol=symbol,
            direction=direction,
            confidence=best_conf,
            latency_ms=latency_ms,
            model_id=best_model,
            features_used=len(features),
            timestamp=time.time(),
        )
