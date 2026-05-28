"""
Market Data Ingestion Engine

Async-parallel ingestion from multiple market data sources
with retry/circuit-breaker resilience and rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

DEXSCREENER_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
DEXSCREENER_TOKEN_URL = "https://api.dexscreener.com/tokens/v1/solana/{mint}"
JUPITER_QUOTE_URL = "https://api.jup.ag/swap/v1/quote"
WRAPPED_SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SUPPORTED_CHAINS = {"solana", "ethereum", "bsc", "base", "arbitrum", "polygon"}
EXCLUDED_SYMBOLS = {"SOL", "USDC", "USDT", "USDS", "USDH", "WSOL"}


@dataclass
class MarketTick:
    """Single market data observation."""

    symbol: str
    price_usd: float
    volume_usd: float
    liquidity_usd: float
    change_pct: float
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    chain: str = ""
    pair_address: str = ""
    mint: str = ""
    dex_id: str = ""
    bid: float = 0.0
    ask: float = 0.0


class CircuitBreaker:
    """Simple circuit breaker for API resilience."""

    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 60.0):
        self._failures = 0
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._last_failure: float = 0.0
        self._state = "closed"  # closed | open | half-open

    @property
    def is_open(self) -> bool:
        if self._state == "open":
            if time.time() - self._last_failure > self._reset_timeout:
                self._state = "half-open"
                return False
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure = time.time()
        if self._failures >= self._threshold:
            self._state = "open"
            logger.warning("Circuit breaker OPEN after %d failures.", self._failures)


class TickBuffer:
    """Time-aware ring buffer for streaming market data."""

    def __init__(self, maxlen: int = 500):
        self._buffer: Deque[MarketTick] = deque(maxlen=maxlen)

    def append(self, tick: MarketTick) -> None:
        self._buffer.append(tick)

    def latest(self, n: int = 1) -> List[MarketTick]:
        return list(self._buffer)[-n:]

    def window(self, seconds: float) -> List[MarketTick]:
        cutoff = time.time() - seconds
        return [t for t in self._buffer if t.timestamp >= cutoff]

    @property
    def prices(self) -> List[float]:
        return [t.price_usd for t in self._buffer]

    @property
    def volumes(self) -> List[float]:
        return [t.volume_usd for t in self._buffer]

    def __len__(self) -> int:
        return len(self._buffer)


class MarketIngestion:
    """Async-parallel market data ingestion with circuit breakers."""

    def __init__(self, queries: List[str], cmc_api_key: str = ""):
        self.queries = queries
        self.cmc_api_key = cmc_api_key
        self._breakers: Dict[str, CircuitBreaker] = {
            "dexscreener": CircuitBreaker(),
            "jupiter": CircuitBreaker(),
            "cmc": CircuitBreaker(),
        }
        self._buffers: Dict[str, TickBuffer] = {}

    def get_buffer(self, symbol: str) -> TickBuffer:
        if symbol not in self._buffers:
            self._buffers[symbol] = TickBuffer()
        return self._buffers[symbol]

    async def _fetch_with_retry(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        max_retries: int = 3,
        breaker_key: str = "",
    ) -> Optional[dict]:
        breaker = self._breakers.get(breaker_key)
        if breaker and breaker.is_open:
            logger.debug("Circuit breaker open for %s — skipping.", breaker_key)
            return None

        for attempt in range(1, max_retries + 1):
            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status != 200:
                        logger.warning("%s returned HTTP %d (attempt %d).", url, resp.status, attempt)
                        if attempt < max_retries:
                            await asyncio.sleep(2**attempt * 0.5)
                        continue
                    data = await resp.json()
                    if breaker:
                        breaker.record_success()
                    return data
            except Exception as exc:
                logger.warning("Fetch %s failed (attempt %d): %s", url, attempt, exc)
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt * 0.5)

        if breaker:
            breaker.record_failure()
        return None

    async def fetch_dexscreener(self, limit: int = 20) -> List[MarketTick]:
        ticks: List[MarketTick] = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            tasks = [
                self._fetch_with_retry(
                    session, DEXSCREENER_SEARCH_URL,
                    params={"q": q}, breaker_key="dexscreener"
                )
                for q in self.queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            for pair in result.get("pairs", [])[:50]:
                try:
                    chain = str(pair.get("chainId", "")).lower()
                    if chain and chain not in SUPPORTED_CHAINS:
                        continue
                    base = pair.get("baseToken", {})
                    symbol = str(base.get("symbol", "?")).upper()
                    if symbol in EXCLUDED_SYMBOLS:
                        continue
                    price = float(pair.get("priceUsd", 0) or 0)
                    if price <= 0:
                        continue

                    tick = MarketTick(
                        symbol=symbol,
                        price_usd=price,
                        volume_usd=float(pair.get("volume", {}).get("h24", 0) or 0),
                        liquidity_usd=float(pair.get("liquidity", {}).get("usd", 0) or 0),
                        change_pct=float(pair.get("priceChange", {}).get("h24", 0) or 0),
                        source="dexscreener",
                        chain=chain,
                        pair_address=str(pair.get("pairAddress", "")),
                        mint=str(base.get("address", "")),
                        dex_id=str(pair.get("dexId", "")),
                    )
                    ticks.append(tick)
                    self.get_buffer(symbol).append(tick)
                except Exception:
                    continue

        ticks.sort(key=lambda t: t.change_pct, reverse=True)
        return ticks[:limit]

    async def fetch_sol_price(self) -> Optional[float]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            data = await self._fetch_with_retry(
                session, JUPITER_QUOTE_URL,
                params={
                    "inputMint": WRAPPED_SOL_MINT,
                    "outputMint": USDC_MINT,
                    "amount": 1_000_000_000,
                    "slippageBps": 100,
                    "swapMode": "ExactIn",
                },
                breaker_key="jupiter",
            )
        if not data:
            return None
        out = data.get("outAmount")
        return float(out) / 1_000_000.0 if out else None

    async def fetch_token_price(self, mint: str) -> Optional[float]:
        url = DEXSCREENER_TOKEN_URL.format(mint=mint)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            data = await self._fetch_with_retry(
                session, url, breaker_key="dexscreener"
            )
        if not data or not isinstance(data, list) or not data:
            return None
        price = data[0].get("priceUsd")
        return float(price) if price else None
