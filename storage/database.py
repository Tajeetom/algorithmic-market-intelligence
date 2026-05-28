"""
Storage Layer — SQLite Persistence

Lightweight persistence for market ticks, trade events,
checkpoint state, and feature snapshots.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PlatformStore:
    """SQLite-backed storage for platform state and market data."""

    def __init__(self, db_path: str = "data/platform.db"):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price_usd REAL NOT NULL,
                volume_usd REAL DEFAULT 0,
                liquidity_usd REAL DEFAULT 0,
                change_pct REAL DEFAULT 0,
                source TEXT DEFAULT '',
                chain TEXT DEFAULT '',
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                side TEXT DEFAULT '',
                qty REAL DEFAULT 0,
                price REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                status TEXT DEFAULT '',
                details TEXT DEFAULT '',
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts
                ON ticks(symbol, timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_ts
                ON trades(timestamp);
            """
        )
        self._conn.commit()

    def store_tick(
        self,
        symbol: str,
        price: float,
        volume: float = 0,
        liquidity: float = 0,
        change_pct: float = 0,
        source: str = "",
        chain: str = "",
        timestamp: float | None = None,
    ) -> None:
        ts = timestamp or time.time()
        self._conn.execute(
            "INSERT INTO ticks (symbol, price_usd, volume_usd, liquidity_usd, "
            "change_pct, source, chain, timestamp) VALUES (?,?,?,?,?,?,?,?)",
            (symbol, price, volume, liquidity, change_pct, source, chain, ts),
        )
        self._conn.commit()

    def get_ticks(
        self, symbol: str, limit: int = 100, since: float = 0
    ) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM ticks WHERE symbol = ? AND timestamp >= ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (symbol, since, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def store_trade(
        self,
        symbol: str,
        action: str,
        side: str = "",
        qty: float = 0,
        price: float = 0,
        pnl: float = 0,
        status: str = "",
        details: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT INTO trades (symbol, action, side, qty, price, pnl, "
            "status, details, timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            (symbol, action, side, qty, price, pnl, status, details, time.time()),
        )
        self._conn.commit()

    def save_checkpoint(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (key, value, updated_at) "
            "VALUES (?,?,?)",
            (key, payload, time.time()),
        )
        self._conn.commit()

    def load_checkpoint(self, key: str) -> Optional[Any]:
        row = self._conn.execute(
            "SELECT value FROM checkpoints WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row["value"]) if row else None

    def close(self) -> None:
        self._conn.close()
