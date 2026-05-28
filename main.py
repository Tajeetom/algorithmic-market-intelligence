"""
Real-Time Algorithmic Market Intelligence Platform — Main Orchestrator

Event-driven streaming pipeline integrating:
  Ingestion → Feature Extraction → Inference → Risk → Execution → Observability
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import List

from config.settings import get_settings
from features.extractor import FeatureExtractor
from ingestion.market_feed import MarketIngestion, MarketTick
from models.predictor import PredictionEngine, SignalDirection
from observability.metrics import get_metrics
from risk.engine import RiskEngine
from storage.database import PlatformStore


def _setup_logging(level: str, log_file: str) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(Path(log_file))
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)


logger = logging.getLogger("platform")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="amp",
        description=(
            "Real-Time Algorithmic Market Intelligence Platform — "
            "Streaming prediction with risk-managed execution."
        ),
    )
    parser.add_argument("--mode", choices=["live", "backtest", "train"], default="live")
    parser.add_argument("--ticks", type=int, default=0, help="Max ticks (0=unlimited).")
    parser.add_argument("--interval", type=float, default=0, help="Override poll interval.")
    parser.add_argument("--dry-run", action="store_true", help="No execution, signals only.")
    parser.add_argument("--export-metrics", type=str, default="", help="Metrics export path.")
    return parser.parse_args()


def _render_dashboard(
    tick: int,
    ticks_data: List[MarketTick],
    risk: RiskEngine,
    metrics_reg,
    settings,
    sol_price: float,
) -> None:
    """Render a terminal dashboard view."""
    print("\033[2J\033[H", end="")
    print("=" * 78)
    print(" REAL-TIME MARKET INTELLIGENCE PLATFORM ".center(78))
    print("=" * 78)
    print(f"Tick: {tick}  |  SOL/USD: ${sol_price:,.2f}  |  Interval: {settings.poll_interval}s")

    summary = risk.portfolio_summary(settings.initial_capital)
    print(
        f"Positions: {summary['open_positions']}  |  "
        f"Exposure: ${summary['total_exposure_usd']:,.2f} ({summary['exposure_pct']:.1f}%)  |  "
        f"Drawdown: {summary['drawdown_pct']:.2f}%"
    )
    print("-" * 78)

    # Signals
    signals = metrics_reg.get_counter("signals_generated")
    inf_stats = metrics_reg.histogram_stats("inference_latency_ms")
    print(
        f"Signals: {int(signals)}  |  "
        f"Inference p50={inf_stats['p50']:.1f}ms p95={inf_stats['p95']:.1f}ms"
    )
    print("-" * 78)

    # Top ticks
    print("Market Feed (top movers):")
    for t in ticks_data[:5]:
        print(
            f"  {t.symbol:<8} ${t.price_usd:>10.6f}  "
            f"chg={t.change_pct:+6.2f}%  vol=${t.volume_usd:>10,.0f}  "
            f"liq=${t.liquidity_usd:>10,.0f}  {t.source}/{t.chain}"
        )
    if not ticks_data:
        print("  (awaiting data...)")

    # Positions
    print("-" * 78)
    print("Open Positions:")
    if not risk.positions:
        print("  (none)")
    for sym, pos in risk.positions.items():
        print(
            f"  {pos.side.upper():<5} {sym:<8} qty={pos.qty:.4f}  "
            f"entry=${pos.entry_price:.6f}  now=${pos.current_price:.6f}  "
            f"pnl=${pos.pnl:+.4f} ({pos.pnl_pct:+.2f}%)"
        )
    print("=" * 78)
    print("Ctrl+C to stop")


async def run_live(args: argparse.Namespace) -> None:
    """Main streaming loop."""
    settings = get_settings()
    _setup_logging(settings.log_level, settings.log_file)

    logger.info("=" * 60)
    logger.info("Platform starting — mode=live")
    logger.info("=" * 60)

    ingestion = MarketIngestion(settings.dex_queries, settings.cmc_api_key)
    extractor = FeatureExtractor(
        window=settings.feature_window,
        rsi_period=settings.rsi_period,
        ema_short=settings.ema_short,
        ema_long=settings.ema_long,
    )
    predictor = PredictionEngine(
        threshold=settings.signal_threshold,
        model_path=settings.model_path,
    )
    risk = RiskEngine(
        max_position_pct=settings.max_position_pct,
        max_drawdown_pct=settings.max_drawdown_pct,
        max_open_positions=settings.max_open_positions,
        volatility_threshold=settings.volatility_throttle_threshold,
        daily_loss_limit_pct=settings.daily_loss_limit_pct,
        stop_loss_pct=settings.sell_stop_loss_pct,
        take_profit_pct=settings.sell_profit_threshold_pct,
    )
    risk.set_initial_equity(settings.initial_capital)

    store = PlatformStore(settings.db_path)
    metrics_reg = get_metrics(settings.metrics_export_path)

    interval = args.interval if args.interval > 0 else settings.poll_interval
    sol_price = 0.0

    try:
        tick = 0
        while True:
            tick += 1
            t0 = time.time()

            # Fetch market data (async parallel)
            ticks = await ingestion.fetch_dexscreener(limit=20)
            metrics_reg.inc("ingestion_cycles")
            metrics_reg.observe("api_response_latency_ms", (time.time() - t0) * 1000)

            # SOL price
            new_sol = await ingestion.fetch_sol_price()
            if new_sol:
                sol_price = new_sol

            # Feature extraction + inference for each tick
            for market_tick in ticks[:10]:
                fv = extractor.update(
                    market_tick.symbol,
                    market_tick.price_usd,
                    market_tick.volume_usd,
                    market_tick.liquidity_usd,
                    market_tick.timestamp,
                )

                pred = predictor.infer(market_tick.symbol, fv.to_array())
                metrics_reg.observe("inference_latency_ms", pred.latency_ms)

                if pred.direction != SignalDirection.NEUTRAL:
                    metrics_reg.inc("signals_generated")
                    logger.info(
                        "Signal: %s %s conf=%.2f model=%s",
                        pred.direction.value,
                        market_tick.symbol,
                        pred.confidence,
                        pred.model_id,
                    )

                # Update existing position prices
                risk.update_price(market_tick.symbol, market_tick.price_usd)

                # Store tick
                store.store_tick(
                    market_tick.symbol, market_tick.price_usd,
                    market_tick.volume_usd, market_tick.liquidity_usd,
                    market_tick.change_pct, market_tick.source, market_tick.chain,
                )

            # Risk checks
            exits = risk.check_exits()
            for sym in exits:
                pnl = risk.register_exit(sym)
                if pnl is not None:
                    store.store_trade(sym, "EXIT", pnl=pnl, status="auto")
                    metrics_reg.inc("exits")

            # Render
            _render_dashboard(tick, ticks, risk, metrics_reg, settings, sol_price)

            # Export metrics periodically
            if tick % 10 == 0:
                metrics_reg.export()

            if args.ticks > 0 and tick >= args.ticks:
                break

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        pass
    finally:
        metrics_reg.export()
        store.close()
        logger.info("Platform shutdown. Metrics exported.")


def main() -> None:
    args = _parse_args()

    if args.mode == "live":
        asyncio.run(run_live(args))
    elif args.mode == "backtest":
        print("Run backtesting via: python -m backtesting.replay_engine")
    elif args.mode == "train":
        print("Run training via: python -m training.pipeline")
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
