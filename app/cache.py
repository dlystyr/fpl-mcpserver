"""Redis cache with fallback for providers without RedisJSON."""

import json
import logging
import redis.asyncio as redis
from config import get_settings

logger = logging.getLogger(__name__)
_client: redis.Redis | None = None
_has_json_module: bool = False


async def init_cache() -> bool:
    """Initialize Redis connection and detect capabilities."""
    global _client, _has_json_module
    settings = get_settings()

    try:
        _client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            username=settings.redis_user,
            password=settings.redis_password,
            ssl=settings.redis_ssl,
            decode_responses=True,
        )
        await _client.ping()

        # Detect if RedisJSON module is available
        try:
            await _client.execute_command("JSON.SET", "_test_json", "$", '"test"')
            await _client.delete("_test_json")
            _has_json_module = True
            logger.info("RedisJSON module detected")
        except redis.ResponseError:
            _has_json_module = False
            logger.info("RedisJSON not available, using standard Redis with JSON serialization")

        mode = "external" if settings.is_external_redis else "local"
        ssl_status = " (SSL)" if settings.redis_ssl else ""
        logger.info(f"Connected to Redis ({mode}) at {settings.redis_host}:{settings.redis_port}{ssl_status}")
        return True
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        _client = None
        return False


async def close_cache() -> None:
    """Close Redis connection."""
    global _client
    if _client:
        await _client.close()
        _client = None


async def cache_get(key: str):
    """Get JSON value from cache."""
    if not _client:
        return None
    try:
        if _has_json_module:
            result = await _client.json().get(key, "$")
            # RedisJSON with "$" path returns results wrapped in a list
            if isinstance(result, list) and len(result) == 1:
                return result[0]
            return result
        else:
            # Standard Redis: deserialize JSON string
            result = await _client.get(key)
            if result is None:
                return None
            return json.loads(result)
    except Exception as e:
        logger.debug(f"Cache get error for {key}: {e}")
        return None


def _json_serializer(obj):
    """Custom JSON serializer for types not supported by default."""
    from datetime import datetime, date
    from decimal import Decimal
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)


async def cache_set(key: str, value, ttl: int = 3600) -> bool:
    """Set JSON value in cache."""
    if not _client:
        return False
    try:
        # Pre-serialize to handle Decimal, datetime, etc.
        json_str = json.dumps(value, default=_json_serializer)
        if _has_json_module:
            # RedisJSON expects a JSON string for the value
            await _client.execute_command("JSON.SET", key, "$", json_str)
            await _client.expire(key, ttl)
        else:
            # Standard Redis
            await _client.set(key, json_str, ex=ttl)
        return True
    except Exception as e:
        logger.warning(f"Cache set error: {e}")
        return False


async def cache_delete(key: str) -> bool:
    """Delete key from cache."""
    if not _client:
        return False
    try:
        await _client.delete(key)
        return True
    except Exception:
        return False


async def cache_get_player(player_id: int) -> dict | None:
    return await cache_get(f"fpl:player:{player_id}")


async def cache_get_all_players() -> list[dict] | None:
    return await cache_get("fpl:players:all")


async def cache_get_team(team_id: int) -> dict | None:
    return await cache_get(f"fpl:team:{team_id}")


async def cache_get_all_teams() -> list[dict] | None:
    return await cache_get("fpl:teams:all")


async def cache_get_fixtures(gw: int | None = None) -> list[dict] | None:
    key = f"fpl:fixtures:gw:{gw}" if gw else "fpl:fixtures:all"
    return await cache_get(key)


async def populate_cache_from_db() -> dict:
    """Populate Redis cache from PostgreSQL."""
    from database import fetch_all

    stats = {"players": 0, "teams": 0, "fixtures": 0}

    # Cache teams
    teams = await fetch_all("SELECT * FROM teams")
    await cache_set("fpl:teams:all", teams)
    for t in teams:
        await cache_set(f"fpl:team:{t['id']}", t)
    stats["teams"] = len(teams)

    # Cache players with team info
    players = await fetch_all("""
        SELECT p.*, t.name as team_name, t.short_name as team
        FROM players p JOIN teams t ON p.team_id = t.id
    """)
    await cache_set("fpl:players:all", players)
    for p in players:
        await cache_set(f"fpl:player:{p['id']}", p)
    stats["players"] = len(players)

    # Cache fixtures
    fixtures = await fetch_all("SELECT * FROM fixtures ORDER BY event, kickoff_time")
    await cache_set("fpl:fixtures:all", fixtures)

    # Group by gameweek
    gw_map = {}
    for f in fixtures:
        gw = f.get("event")
        if gw:
            gw_map.setdefault(gw, []).append(f)
    for gw, gw_fixtures in gw_map.items():
        await cache_set(f"fpl:fixtures:gw:{gw}", gw_fixtures)
    stats["fixtures"] = len(fixtures)

    logger.info(f"Cache populated: {stats}")
    return stats
