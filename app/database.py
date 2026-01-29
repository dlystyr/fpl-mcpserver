"""PostgreSQL database connection using asyncpg."""

import asyncpg
from config import get_settings

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    """Initialize database connection pool."""
    global _pool
    settings = get_settings()
    _pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)


async def close_db() -> None:
    """Close database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def get_pool() -> asyncpg.Pool:
    """Get database pool."""
    if _pool is None:
        await init_db()
    return _pool


async def fetch_all(query: str, *args) -> list[dict]:
    """Execute query and return all rows as dicts."""
    pool = await get_pool()
    rows = await pool.fetch(query, *args)
    return [dict(r) for r in rows]


async def fetch_one(query: str, *args) -> dict | None:
    """Execute query and return one row as dict."""
    pool = await get_pool()
    row = await pool.fetchrow(query, *args)
    return dict(row) if row else None


async def execute(query: str, *args) -> str:
    """Execute a query."""
    pool = await get_pool()
    return await pool.execute(query, *args)
