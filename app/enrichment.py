"""
External data enrichment module for FPL MCP Server.

Fetches additional player statistics from:
- Understat: xG per shot, shot data, situation breakdowns
- FBref: Shots, SOT, progressive passes/carries, defensive actions

Data is cached in Redis (24h TTL) to survive server restarts.
Falls back to in-memory cache if Redis unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

# Cache TTL - 24 hours since external data only changes after matches
CACHE_TTL_SECONDS = 3600 * 24

# Redis client (lazy initialized)
_redis_client = None
_redis_available: bool | None = None

# Fallback in-memory cache if Redis unavailable
_memory_cache: dict[str, tuple[float, Any]] = {}


async def _get_redis():
    """Get or create Redis client."""
    global _redis_client, _redis_available

    if _redis_available is False:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis.asyncio as redis

        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        user = os.getenv("REDIS_USER")
        use_ssl = os.getenv("REDIS_SSL", "").lower() in ("true", "1", "yes")

        _redis_client = redis.Redis(
            host=host,
            port=port,
            username=user,
            password=password,
            ssl=use_ssl,
            decode_responses=True,
        )
        await _redis_client.ping()
        _redis_available = True
        logger.info(f"Enrichment cache connected to Redis at {host}:{port}")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable for enrichment cache, using memory: {e}")
        _redis_available = False
        return None


async def _get_cached(key: str) -> Any | None:
    """Get cached value from Redis (using RedisJSON) or memory fallback."""
    redis_key = f"fpl:enrichment:{key}"

    # Try Redis with RedisJSON
    redis_client = await _get_redis()
    if redis_client:
        try:
            # Use RedisJSON JSON.GET command
            result = await redis_client.execute_command("JSON.GET", redis_key, "$")
            if result:
                # JSON.GET with "$" returns a JSON array string, parse and unwrap
                parsed = json.loads(result)
                if isinstance(parsed, list) and len(parsed) == 1:
                    return parsed[0]
                return parsed
        except Exception as e:
            logger.debug(f"Redis JSON.GET error: {e}")

    # Fallback to memory cache
    if key in _memory_cache:
        expiry, data = _memory_cache[key]
        if time.time() < expiry:
            return data
        del _memory_cache[key]

    return None


async def _set_cached(key: str, data: Any) -> None:
    """Set cached value in Redis (using RedisJSON) or memory fallback."""
    redis_key = f"fpl:enrichment:{key}"

    # Try Redis with RedisJSON
    redis_client = await _get_redis()
    if redis_client:
        try:
            # Use RedisJSON JSON.SET command
            json_str = json.dumps(data, default=str)
            await redis_client.execute_command("JSON.SET", redis_key, "$", json_str)
            # Set TTL separately (JSON.SET doesn't support EX)
            await redis_client.expire(redis_key, CACHE_TTL_SECONDS)
            return
        except Exception as e:
            logger.debug(f"Redis JSON.SET error: {e}")

    # Fallback to memory cache
    _memory_cache[key] = (time.time() + CACHE_TTL_SECONDS, data)


def _normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    # Remove accents and special characters
    name = name.lower().strip()
    # Common substitutions
    replacements = {
        "á": "a", "à": "a", "ä": "a", "â": "a", "ã": "a",
        "é": "e", "è": "e", "ë": "e", "ê": "e",
        "í": "i", "ì": "i", "ï": "i", "î": "i",
        "ó": "o", "ò": "o", "ö": "o", "ô": "o", "õ": "o", "ø": "o",
        "ú": "u", "ù": "u", "ü": "u", "û": "u",
        "ñ": "n", "ç": "c", "ß": "ss",
        "-": " ", "'": "", "'": "", ".": "",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    # Remove extra spaces
    name = re.sub(r"\s+", " ", name)
    return name


def _name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names (0-1)."""
    n1 = _normalize_name(name1)
    n2 = _normalize_name(name2)

    # Exact match
    if n1 == n2:
        return 1.0

    # Check if one contains the other (handles "Bruno" vs "Bruno Fernandes")
    if n1 in n2 or n2 in n1:
        return 0.9

    # Check last name match (most reliable for footballers)
    parts1 = n1.split()
    parts2 = n2.split()
    if parts1 and parts2:
        if parts1[-1] == parts2[-1]:  # Last names match
            return 0.85

    # Fuzzy match
    return SequenceMatcher(None, n1, n2).ratio()


def _format_understat_stats(raw: dict[str, Any]) -> dict[str, Any]:
    """Format raw Understat data into useful stats."""
    games = max(1, int(float(raw.get("games", 1) or 1)))
    shots = float(raw.get("shots", 0) or 0)
    xg = float(raw.get("xG", 0) or 0)
    goals = int(float(raw.get("goals", 0) or 0))
    xa = float(raw.get("xA", 0) or 0)
    assists = int(float(raw.get("assists", 0) or 0))
    key_passes = int(float(raw.get("key_passes", 0) or 0))
    npg = int(float(raw.get("npg", 0) or 0))  # Non-penalty goals
    npxg = float(raw.get("npxG", 0) or 0)  # Non-penalty xG
    xgchain = float(raw.get("xGChain", 0) or 0)  # xG chain involvement
    xgbuildup = float(raw.get("xGBuildup", 0) or 0)  # xG buildup involvement

    return {
        "source": "understat",
        "understat_id": raw.get("id"),
        "team": raw.get("team_title"),
        "games": games,
        "shots": shots,
        "shots_per_90": round(shots / games, 2) if games else 0,
        "xg": round(xg, 2),
        "xg_per_90": round(xg / games, 2) if games else 0,
        "xg_per_shot": round(xg / shots, 3) if shots > 0 else 0,
        "goals": goals,
        "goals_minus_xg": round(goals - xg, 2),
        "xa": round(xa, 2),
        "xa_per_90": round(xa / games, 2) if games else 0,
        "assists": assists,
        "assists_minus_xa": round(assists - xa, 2),
        "key_passes": key_passes,
        "key_passes_per_90": round(key_passes / games, 2) if games else 0,
        "npg": npg,
        "npxg": round(npxg, 2),
        "npxg_per_90": round(npxg / games, 2) if games else 0,
        "xg_chain": round(xgchain, 2),
        "xg_buildup": round(xgbuildup, 2),
        "conversion_rate": round(goals / shots * 100, 1) if shots > 0 else 0,
    }


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if val is None:
        return default
    try:
        import pandas as pd
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default
    except ImportError:
        try:
            return float(val)
        except (ValueError, TypeError):
            return default


def _format_fbref_stats(raw: dict[str, Any]) -> dict[str, Any]:
    """Format raw FBref data into useful stats."""
    standard = raw.get("standard", {})
    shooting = raw.get("shooting", {})
    passing = raw.get("passing", {})

    # Extract key stats with safe defaults
    games = max(1, int(_safe_float(standard.get("MP", 1))))
    minutes = int(_safe_float(standard.get("Min", 0)))
    nineties = minutes / 90 if minutes > 0 else 1

    # Shooting stats
    shots = _safe_float(shooting.get("Sh", 0))
    sot = _safe_float(shooting.get("SoT", 0))
    sh_per_90 = _safe_float(shooting.get("Sh/90", 0))
    sot_per_90 = _safe_float(shooting.get("SoT/90", 0))
    sot_pct = _safe_float(shooting.get("SoT%", 0))
    g_per_sh = _safe_float(shooting.get("G/Sh", 0))
    g_per_sot = _safe_float(shooting.get("G/SoT", 0))
    dist = _safe_float(shooting.get("Dist", 0))  # Avg shot distance
    npxg_per_sh = _safe_float(shooting.get("npxG/Sh", 0))

    # Passing stats
    pass_completed = _safe_float(passing.get("Cmp", 0))
    pass_pct = _safe_float(passing.get("Cmp%", 0))
    progressive_passes = _safe_float(passing.get("PrgP", 0))
    key_passes = _safe_float(passing.get("KP", 0))
    final_third = _safe_float(passing.get("1/3", 0))
    ppa = _safe_float(passing.get("PPA", 0))  # Passes into penalty area

    return {
        "source": "fbref",
        "games": games,
        "minutes": minutes,
        # Shooting
        "shots": round(shots, 0),
        "shots_on_target": round(sot, 0),
        "shots_per_90": round(sh_per_90, 2),
        "sot_per_90": round(sot_per_90, 2),
        "sot_percentage": round(sot_pct, 1),
        "goals_per_shot": round(g_per_sh, 3),
        "goals_per_sot": round(g_per_sot, 3),
        "avg_shot_distance": round(dist, 1),
        "npxg_per_shot": round(npxg_per_sh, 3),
        # Passing
        "passes_completed": round(pass_completed, 0),
        "pass_completion_pct": round(pass_pct, 1),
        "progressive_passes": round(progressive_passes, 0),
        "progressive_passes_per_90": round(progressive_passes / nineties, 2) if nineties > 0 else 0,
        "key_passes": round(key_passes, 0),
        "key_passes_per_90": round(key_passes / nineties, 2) if nineties > 0 else 0,
        "final_third_passes": round(final_third, 0),
        "passes_into_penalty_area": round(ppa, 0),
        "ppa_per_90": round(ppa / nineties, 2) if nineties > 0 else 0,
    }


async def _load_understat_data(season: str = "2024") -> dict[str, dict[str, Any]]:
    """
    Load all EPL player data from Understat for a season.

    Returns dict of normalized_name -> player data.
    """
    cache_key = f"understat_epl_{season}"
    cached = await _get_cached(cache_key)
    if cached is not None:
        logger.info(f"Loaded {len(cached)} Understat players from cache")
        return cached

    try:
        from understatapi import UnderstatClient

        with UnderstatClient() as understat:
            raw_data = understat.league(league="EPL").get_player_data(season=season)

        # Index by normalized name for matching
        player_data: dict[str, dict[str, Any]] = {}
        for player in raw_data:
            name = player.get("player_name", "")
            norm_name = _normalize_name(name)
            player_data[norm_name] = player

        await _set_cached(cache_key, player_data)
        logger.info(f"Loaded {len(player_data)} Understat players for {season}")
        return player_data

    except ImportError:
        logger.warning("understatapi not installed, Understat enrichment disabled")
        return {}
    except Exception as e:
        logger.error(f"Failed to load Understat data: {e}")
        return {}


async def _load_fbref_data(season: str = "2024-2025") -> dict[str, dict[str, Any]]:
    """
    Load player stats from FBref for a season.

    Returns dict of normalized_name -> player data.
    """
    cache_key = f"fbref_epl_{season}"
    cached = await _get_cached(cache_key)
    if cached is not None:
        logger.info(f"Loaded {len(cached)} FBref players from cache")
        return cached

    try:
        import soccerdata as sd

        fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)

        # Fetch multiple stat types
        player_data: dict[str, dict[str, Any]] = {}

        try:
            standard = fbref.read_player_season_stats(stat_type="standard")
            if standard is not None and not standard.empty:
                for idx, row in standard.iterrows():
                    player_name = str(idx[1]) if isinstance(idx, tuple) else str(idx)
                    norm_name = _normalize_name(player_name)
                    player_data[norm_name] = {"name": player_name, "standard": row.to_dict()}
        except Exception as e:
            logger.warning(f"FBref standard stats failed: {e}")

        try:
            shooting = fbref.read_player_season_stats(stat_type="shooting")
            if shooting is not None and not shooting.empty:
                for idx, row in shooting.iterrows():
                    player_name = str(idx[1]) if isinstance(idx, tuple) else str(idx)
                    norm_name = _normalize_name(player_name)
                    if norm_name in player_data:
                        player_data[norm_name]["shooting"] = row.to_dict()
                    else:
                        player_data[norm_name] = {"name": player_name, "shooting": row.to_dict()}
        except Exception as e:
            logger.warning(f"FBref shooting stats failed: {e}")

        try:
            passing = fbref.read_player_season_stats(stat_type="passing")
            if passing is not None and not passing.empty:
                for idx, row in passing.iterrows():
                    player_name = str(idx[1]) if isinstance(idx, tuple) else str(idx)
                    norm_name = _normalize_name(player_name)
                    if norm_name in player_data:
                        player_data[norm_name]["passing"] = row.to_dict()
                    else:
                        player_data[norm_name] = {"name": player_name, "passing": row.to_dict()}
        except Exception as e:
            logger.warning(f"FBref passing stats failed: {e}")

        if player_data:
            await _set_cached(cache_key, player_data)
            logger.info(f"Loaded {len(player_data)} FBref players for {season}")

        return player_data

    except ImportError:
        logger.warning("soccerdata not installed, FBref enrichment disabled")
        return {}
    except Exception as e:
        logger.error(f"Failed to load FBref data: {e}")
        return {}


def _find_player_in_data(
    player_name: str,
    team_name: str | None,
    data: dict[str, dict[str, Any]],
    name_key: str = "player_name",
    team_key: str = "team_title",
) -> dict[str, Any] | None:
    """Find a player in external data by name matching."""
    norm_name = _normalize_name(player_name)

    # Try exact match first
    if norm_name in data:
        return data[norm_name]

    # Fuzzy matching
    best_match = None
    best_score = 0.0

    for stored_name, player_data in data.items():
        # Get the original name from the data
        original_name = player_data.get(name_key, player_data.get("name", stored_name))
        score = _name_similarity(player_name, original_name)

        # Boost score if team matches
        if team_name:
            stored_team = player_data.get(team_key, "")
            if stored_team and _normalize_name(team_name) in _normalize_name(stored_team):
                score += 0.1

        if score > best_score and score >= 0.75:
            best_score = score
            best_match = player_data

    return best_match


def _create_summary(
    understat: dict[str, Any] | None,
    fbref: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create a summary combining best data from each source."""
    summary: dict[str, Any] = {}

    # Prefer Understat for xG-related metrics (their specialty)
    if understat:
        summary["shots_per_90"] = understat.get("shots_per_90", 0)
        summary["xg_per_90"] = understat.get("xg_per_90", 0)
        summary["xg_per_shot"] = understat.get("xg_per_shot", 0)
        summary["xa_per_90"] = understat.get("xa_per_90", 0)
        summary["key_passes_per_90"] = understat.get("key_passes_per_90", 0)
        summary["conversion_rate"] = understat.get("conversion_rate", 0)
        summary["npxg_per_90"] = understat.get("npxg_per_90", 0)
        summary["goals_minus_xg"] = understat.get("goals_minus_xg", 0)
        summary["assists_minus_xa"] = understat.get("assists_minus_xa", 0)

    # Prefer FBref for SOT and passing metrics
    if fbref:
        summary["sot_per_90"] = fbref.get("sot_per_90", 0)
        summary["sot_percentage"] = fbref.get("sot_percentage", 0)
        summary["goals_per_sot"] = fbref.get("goals_per_sot", 0)
        summary["progressive_passes_per_90"] = fbref.get("progressive_passes_per_90", 0)
        summary["passes_into_penalty_area_per_90"] = fbref.get("ppa_per_90", 0)
        summary["avg_shot_distance"] = fbref.get("avg_shot_distance", 0)

    return summary


# Global data cache (loaded once, stored in Redis)
_understat_data: dict[str, dict[str, Any]] | None = None
_fbref_data: dict[str, dict[str, Any]] | None = None


async def enrich_player_async(player_name: str, team_name: str | None = None) -> dict[str, Any]:
    """
    Get enriched stats for a player from Understat and FBref.

    Data is cached in Redis for 24 hours.

    Args:
        player_name: Player name (from FPL)
        team_name: Optional team name for disambiguation

    Returns:
        Dict with enriched stats from all available sources
    """
    global _understat_data, _fbref_data

    result: dict[str, Any] = {
        "enrichment_available": False,
        "sources": [],
    }

    # Load data if not already loaded
    if _understat_data is None:
        _understat_data = await _load_understat_data()

    if _fbref_data is None:
        _fbref_data = await _load_fbref_data()

    # Find player in Understat data
    if _understat_data:
        understat_player = _find_player_in_data(
            player_name, team_name, _understat_data,
            name_key="player_name", team_key="team_title"
        )
        if understat_player:
            result["understat"] = _format_understat_stats(understat_player)
            result["sources"].append("understat")
            result["enrichment_available"] = True

    # Find player in FBref data
    if _fbref_data:
        fbref_player = _find_player_in_data(
            player_name, team_name, _fbref_data,
            name_key="name", team_key=""
        )
        if fbref_player:
            result["fbref"] = _format_fbref_stats(fbref_player)
            result["sources"].append("fbref")
            result["enrichment_available"] = True

    # Create combined summary
    if result["enrichment_available"]:
        result["summary"] = _create_summary(
            result.get("understat"),
            result.get("fbref"),
        )

    return result
