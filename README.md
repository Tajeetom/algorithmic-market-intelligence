# Real-Time Algorithmic Market Intelligence Platform

> Event-driven streaming prediction system with online inference, risk-managed execution, and full ML lifecycle — from feature extraction to production deployment.

---

## Overview

A production-grade **streaming market intelligence platform** that ingests real-time market data, computes statistical features over sliding windows, runs low-latency ML inference, enforces pre-trade risk checks, and executes with observability at every layer.

### Key Engineering Highlights

- **Streaming Feature Extraction** — RSI, EMA, MACD, VWAP, rolling z-score, realized volatility, momentum decay, volume acceleration, and liquidity imbalance computed incrementally over ring buffers
- **Multi-Model Inference Engine** — Ensemble heuristics + ONNX Runtime for sub-200ms prediction with signal confidence scoring
- **Risk Management Engine** — Position limits, exposure caps, volatility throttling, drawdown circuit breakers, stop-loss/take-profit enforcement, and dynamic position sizing
- **Async-Parallel Ingestion** — `asyncio.gather()` for concurrent API fetches with circuit breakers, exponential backoff retry, and rate limiting
- **Offline Training Pipeline** — Dataset construction, XGBoost/LightGBM training, temporal split validation, hyperparameter tuning, and ONNX export
- **Backtesting Framework** — Replay engine with Sharpe ratio, Sortino ratio, max drawdown, win rate, and profit factor computation
- **SQLite Persistence** — Market ticks, trade events, and checkpoint state with indexed queries
- **Observability** — Counter/gauge/histogram metrics with JSON export for dashboarding
- **Pydantic Configuration** — Typed, validated env loading with `AMP_` prefix
- **Docker-Ready** — Dockerfile + docker-compose for reproducible deployment

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      CLI Interface                                │
│            (--mode live|backtest|train --ticks --dry-run)          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Pydantic Config Loader                            │
│             (.env → validated, typed settings)                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
┌──────────────────┐ ┌────────────┐ ┌────────────────┐
│ Market Ingestion  │ │  Training  │ │  Backtesting   │
│ (async parallel,  │ │  Pipeline  │ │  Replay Engine │
│  circuit breakers,│ │ (XGBoost,  │ │ (Sharpe, DD,   │
│  ring buffers)    │ │  ONNX)     │ │  profit factor)│
└────────┬─────────┘ └────────────┘ └────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│              Streaming Feature Extraction Engine                   │
│  RSI · EMA · MACD · VWAP · z-score · volatility · momentum       │
│  volume acceleration · liquidity imbalance · spread compression   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              Multi-Model Prediction Engine                         │
│         (Ensemble heuristics + ONNX Runtime inference)            │
│           Signal: LONG / SHORT / NEUTRAL + confidence             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Risk Management Engine                          │
│  Position limits · exposure caps · volatility throttle            │
│  Drawdown circuit breaker · stop-loss · take-profit               │
│  Dynamic position sizing · daily loss limit                       │
└──────────┬─────────────────────────────────┬────────────────────┘
           │                                 │
           ▼                                 ▼
┌────────────────────┐            ┌────────────────────┐
│  Execution Layer    │            │  Observability      │
│  (Jupiter swap,     │            │  (counters, gauges, │
│   dry-run mode)     │            │   histograms, JSON) │
└────────┬───────────┘            └────────┬───────────┘
         │                                 │
         ▼                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SQLite Persistence Layer                        │
│        Market ticks · trade events · checkpoints · state          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
algorithmic-market-intelligence/
├── main.py                        # Orchestrator entry point
├── run.sh                         # Shell launcher
├── config/
│   └── settings.py                # Pydantic settings
├── ingestion/
│   └── market_feed.py             # Async market data ingestion
├── features/
│   └── extractor.py               # Streaming feature extraction
├── models/
│   ├── predictor.py               # Multi-model inference engine
│   └── artifacts/                 # Trained model files (ONNX)
├── risk/
│   └── engine.py                  # Risk management & position sizing
├── backtesting/
│   └── replay_engine.py           # Historical replay + metrics
├── training/
│   └── pipeline.py                # Dataset builder + XGBoost + ONNX export
├── execution/                     # Swap execution (Jupiter, broker)
├── storage/
│   └── database.py                # SQLite persistence
├── observability/
│   └── metrics.py                 # Runtime metrics registry
├── tests/
│   └── test_platform.py           # pytest suite (30+ tests)
├── data/                          # Runtime data (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/mtajuddin/algorithmic-market-intelligence.git
cd algorithmic-market-intelligence

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run

```bash
# Live streaming mode
python main.py --mode live

# Dry run (signals only, no execution)
python main.py --mode live --dry-run

# Limited ticks
python main.py --mode live --ticks 100

# Via shell launcher
./run.sh --mode live
```

### 4. Docker

```bash
docker-compose up --build
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## Feature Engineering

The streaming extractor computes 18 features per tick:

| Feature | Category | Description |
|---------|----------|-------------|
| `log_return` | Price | Log of price ratio |
| `rolling_mean` | Statistics | Windowed mean price |
| `rolling_std` | Statistics | Windowed standard deviation |
| `rolling_zscore` | Statistics | Normalized deviation from mean |
| `rsi` | Momentum | Relative Strength Index (14-period) |
| `ema_short` | Trend | 12-period exponential moving average |
| `ema_long` | Trend | 26-period exponential moving average |
| `macd` | Trend | EMA difference |
| `macd_signal` | Trend | 9-period MACD signal line |
| `macd_histogram` | Trend | MACD - signal divergence |
| `vwap` | Volume | Volume-weighted average price |
| `volume_delta` | Volume | Period-over-period volume change |
| `volume_acceleration` | Volume | Rate of volume change |
| `realized_volatility` | Risk | Windowed return standard deviation |
| `momentum` | Trend | N-period price momentum |
| `momentum_decay` | Trend | Exponentially decayed momentum |
| `spread_compression` | Microstructure | Bid-ask spread dynamics |
| `liquidity_imbalance` | Microstructure | Deviation from average liquidity |

---

## Risk Controls

| Control | Parameter | Default |
|---------|-----------|---------|
| Max position size | `AMP_MAX_POSITION_PCT` | 10% of equity |
| Max open positions | `AMP_MAX_OPEN_POSITIONS` | 5 |
| Max drawdown | `AMP_MAX_DRAWDOWN_PCT` | 15% |
| Daily loss limit | `AMP_DAILY_LOSS_LIMIT_PCT` | 5% |
| Stop-loss | `AMP_SELL_STOP_LOSS_PCT` | -5% |
| Take-profit | `AMP_SELL_PROFIT_THRESHOLD_PCT` | 10% |
| Volatility throttle | `AMP_VOLATILITY_THROTTLE_THRESHOLD` | 5% |

---

## Training Pipeline

```bash
# Build dataset, train XGBoost, evaluate, export ONNX
python -c "
from training.pipeline import build_dataset, train_xgboost, export_onnx, TrainingConfig
import random

# Generate synthetic data for demo
prices = [100 + random.gauss(0, 2) for _ in range(500)]
for i in range(1, len(prices)):
    prices[i] = prices[i-1] + random.gauss(0, 0.5)
    prices[i] = max(prices[i], 1)
volumes = [random.uniform(1000, 50000) for _ in prices]

config = TrainingConfig()
X, y = build_dataset(prices, volumes, config)
model, metrics = train_xgboost(X, y, config)
print(metrics)
export_onnx(model, len(X[0]))
"
```

---

## License

MIT

---

## Author

**Tajuddin Mohammed** — [GitHub](https://github.com/mtajuddin) · [LinkedIn](https://linkedin.com/in/mtajudin01)