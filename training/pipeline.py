"""
Offline Training Pipeline

Dataset construction from stored ticks, model training with
XGBoost/LightGBM, hyperparameter tuning, evaluation, and
ONNX export for production inference.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from features.extractor import FeatureExtractor, FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters for the training pipeline."""

    model_type: str = "xgboost"  # xgboost | lightgbm
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    test_split: float = 0.2
    random_seed: int = 42
    target_horizon: int = 5  # periods ahead for label
    label_threshold: float = 0.001  # min return for positive label


def build_dataset(
    prices: List[float],
    volumes: List[float],
    config: TrainingConfig,
    symbol: str = "ASSET",
) -> Tuple[List[List[float]], List[int]]:
    """
    Build feature matrix and labels from raw price/volume series.

    Labels: 1 if forward return > threshold, 0 otherwise.
    Uses temporal ordering (no future leakage).
    """
    extractor = FeatureExtractor()
    features: List[List[float]] = []
    labels: List[int] = []
    horizon = config.target_horizon

    for i in range(len(prices)):
        fv = extractor.update(symbol, prices[i], volumes[i] if i < len(volumes) else 0.0)

        if i + horizon < len(prices) and i >= 50:
            future_return = (prices[i + horizon] - prices[i]) / prices[i]
            label = 1 if future_return > config.label_threshold else 0
            features.append(fv.to_array())
            labels.append(label)

    return features, labels


def train_xgboost(
    X: List[List[float]],
    y: List[int],
    config: TrainingConfig,
) -> Tuple[Any, Dict[str, float]]:
    """Train an XGBoost classifier and return (model, metrics)."""
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        import numpy as np
    except ImportError as exc:
        logger.error("Training dependencies missing: %s", exc)
        raise

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y)

    split_idx = int(len(X_arr) * (1 - config.test_split))
    X_train, X_test = X_arr[:split_idx], X_arr[split_idx:]
    y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_seed,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else 0.0,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_time_sec": round(train_time, 2),
        "feature_count": X_arr.shape[1],
    }

    logger.info("Training complete: %s", metrics)
    return model, metrics


def export_onnx(
    model: Any,
    feature_count: int,
    output_path: str = "models/artifacts/model.onnx",
) -> str:
    """Export trained model to ONNX format for production serving."""
    try:
        import numpy as np
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.warning("skl2onnx not installed — skipping ONNX export.")
        return ""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    initial_type = [("float_input", FloatTensorType([None, feature_count]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(out, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info("ONNX model exported to %s", out)
    return str(out)


def save_metrics(metrics: Dict, path: str = "training/metrics.json") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
