"""
Observability — Runtime Metrics & Structured Logging

Tracks signals generated, inference latency, API response times,
stream drop rates, portfolio exposure, and throughput.
Exports to JSON for Grafana / dashboard consumption.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsRegistry:
    """
    In-process metrics registry with counter, gauge, and histogram support.
    """

    def __init__(self, export_path: str = "data/metrics.json", window: int = 1000):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Deque[float]] = {}
        self._samples: Deque[MetricSample] = deque(maxlen=window)
        self._export_path = Path(export_path)
        self._start_time = time.time()

    def inc(self, name: str, value: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        if name not in self._histograms:
            self._histograms[name] = deque(maxlen=500)
        self._histograms[name].append(value)
        self._samples.append(MetricSample(name=name, value=value))

    def get_counter(self, name: str) -> float:
        return self._counters.get(name, 0)

    def get_gauge(self, name: str) -> float:
        return self._gauges.get(name, 0)

    def histogram_stats(self, name: str) -> Dict[str, float]:
        values = list(self._histograms.get(name, []))
        if not values:
            return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
        values.sort()
        n = len(values)
        return {
            "count": n,
            "mean": sum(values) / n,
            "p50": values[int(n * 0.5)],
            "p95": values[int(n * 0.95)] if n > 20 else values[-1],
            "p99": values[int(n * 0.99)] if n > 100 else values[-1],
            "max": values[-1],
        }

    def summary(self) -> Dict:
        return {
            "uptime_sec": round(time.time() - self._start_time, 1),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.histogram_stats(name)
                for name in self._histograms
            },
        }

    def export(self) -> None:
        self._export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._export_path, "w") as f:
            json.dump(self.summary(), f, indent=2)
        logger.debug("Metrics exported to %s", self._export_path)


# ── Convenience singleton ────────────────────────────────────────────────

_registry: MetricsRegistry | None = None


def get_metrics(export_path: str = "data/metrics.json") -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry(export_path=export_path)
    return _registry
