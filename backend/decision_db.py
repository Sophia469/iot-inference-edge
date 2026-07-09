"""SQLite persistence layer for Decision Engine artefacts.

Rationale: the ML pipeline (training records + decision history) is tabular
and benefits from SQL. Runtime IoT telemetry stays in MongoDB (document-oriented).
"""
import aiosqlite
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import time
import uuid

DB_PATH = Path(__file__).parent / "artefacts" / "decisions.db"
DB_PATH.parent.mkdir(exist_ok=True)


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        # Enable Write-Ahead Log for concurrent readers/writers (fixes intermittent 'database is locked' under load)
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                engine TEXT NOT NULL,
                route TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT,
                probabilities TEXT,
                context TEXT NOT NULL,
                latency_us REAL,
                created_at REAL NOT NULL
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_decisions_created ON decisions(created_at DESC)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_decisions_engine ON decisions(engine)")
        await db.commit()


async def insert_decision(engine: str, route: str, confidence: float, reason: str,
                          probabilities: Dict[str, float], context: Dict[str, Any],
                          latency_us: float) -> str:
    _id = str(uuid.uuid4())
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO decisions (id, engine, route, confidence, reason, probabilities, context, latency_us, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_id, engine, route, confidence, reason,
             json.dumps(probabilities), json.dumps(context), latency_us, time.time()),
        )
        await db.commit()
    return _id


async def recent_decisions(limit: int = 50, engine: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT id, engine, route, confidence, reason, probabilities, context, latency_us, created_at FROM decisions"
    params: tuple = ()
    if engine:
        query += " WHERE engine = ?"
        params = (engine,)
    query += " ORDER BY created_at DESC LIMIT ?"
    params = params + (limit,)
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(query, params)
        rows = await cur.fetchall()
    result = []
    for r in rows:
        result.append({
            "id": r[0], "engine": r[1], "route": r[2], "confidence": r[3],
            "reason": r[4], "probabilities": json.loads(r[5]) if r[5] else {},
            "context": json.loads(r[6]), "latency_us": r[7], "created_at": r[8],
        })
    return result


async def route_distribution(engine: Optional[str] = None, minutes: int = 30) -> Dict[str, int]:
    cutoff = time.time() - minutes * 60
    query = "SELECT route, COUNT(*) FROM decisions WHERE created_at >= ?"
    params: tuple = (cutoff,)
    if engine:
        query += " AND engine = ?"
        params = params + (engine,)
    query += " GROUP BY route"
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(query, params)
        rows = await cur.fetchall()
    dist = {"edge": 0, "cloud": 0, "hybrid": 0}
    for route, count in rows:
        dist[route] = count
    return dist
