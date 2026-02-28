from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Literal

import httpx
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
import uvicorn

from fpl_mcp.enrichment import enrich_player_async

logger = logging.getLogger("fpl-mcp")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

FPL_API_BASE = "https://fantasy.premierleague.com/api"
USER_AGENT = os.getenv("FPL_USER_AGENT", "fpl-mcp/1.0")

# Optional auth for public HTTPS endpoint
BEARER_TOKEN = os.getenv("MCP_BEARER_TOKEN", "")

Position = Literal["GKP", "DEF", "MID", "FWD"]
POS_MAP: dict[int, Position] = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# FPL Rules Constants
SQUAD_SIZE = 15
SQUAD_COMPOSITION: dict[Position, int] = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PLAYERS_PER_TEAM = 3
STARTING_BUDGET = 100.0  # £100m

# Expected points coefficients (based on historical FPL scoring)
# Goals: GKP=6, DEF=6, MID=5, FWD=4
# Assists: 3 for all
# Clean sheets: GKP=4, DEF=4, MID=1, FWD=0
# Minutes: 1pt for 1-59min, 2pt for 60+
# Saves: 1pt per 3 saves (GKP only)
# Bonus: avg ~1.5 for top performers
XPT_COEFFS = {
    "GKP": {"goal": 6, "assist": 3, "cs": 4, "saves_per_pt": 3},
    "DEF": {"goal": 6, "assist": 3, "cs": 4},
    "MID": {"goal": 5, "assist": 3, "cs": 1},
    "FWD": {"goal": 4, "assist": 3, "cs": 0},
}

DEFAULT_PLAYER_FIELDS = [
    "id",
    "first_name",
    "second_name",
    "web_name",
    "team",
    "element_type",
    "now_cost",
    "status",
    "chance_of_playing_next_round",
    "news",
    "minutes",
    "total_points",
    "points_per_game",
    "form",
    "ict_index",
    "creativity",
    "threat",
    "influence",
    "expected_goal_involvements",
    "expected_goals",
    "expected_assists",
    "goals_scored",
    "assists",
    "clean_sheets",
    "yellow_cards",
    "red_cards",
]

# Per-endpoint cache TTLs (seconds)
TTL_BOOTSTRAP = int(os.getenv("FPL_TTL_BOOTSTRAP", "120"))
TTL_FIXTURES = int(os.getenv("FPL_TTL_FIXTURES", "300"))
TTL_ELEMENT_SUMMARY = int(os.getenv("FPL_TTL_ELEMENT_SUMMARY", "1800"))
TTL_EVENT = int(os.getenv("FPL_TTL_EVENT", "120"))
TTL_EVENT_LIVE = int(os.getenv("FPL_TTL_EVENT_LIVE", "30"))

# Simple in-memory cache with per-key expiry
_cache: dict[str, tuple[float, Any]] = {}  # url -> (expires_at, data)

# Team report endpoint cache (60-minute TTL)
_report_cache: dict[str, tuple[float, Any]] = {}
REPORT_CACHE_TTL = 3600  # 60 minutes


def _cache_get(url: str) -> Any | None:
    item = _cache.get(url)
    if not item:
        return None
    expires_at, data = item
    if time.time() >= expires_at:
        _cache.pop(url, None)
        return None
    return data


def _cache_set(url: str, data: Any, ttl: int) -> None:
    _cache[url] = (time.time() + max(1, ttl), data)


async def _get_json(path: str, ttl: int) -> Any:
    url = f"{FPL_API_BASE}/{path.lstrip('/')}"
    cached = _cache_get(url)
    if cached is not None:
        return cached

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()

    _cache_set(url, data, ttl)
    return data


async def _bootstrap() -> dict[str, Any]:
    return await _get_json("bootstrap-static/", ttl=TTL_BOOTSTRAP)


async def _fixtures() -> list[dict[str, Any]]:
    return await _get_json("fixtures/", ttl=TTL_FIXTURES)


async def _element_summary(player_id: int) -> dict[str, Any]:
    return await _get_json(f"element-summary/{player_id}/", ttl=TTL_ELEMENT_SUMMARY)


async def _event(event_id: int) -> dict[str, Any]:
    return await _get_json(f"event/{event_id}/", ttl=TTL_EVENT)


async def _event_live(event_id: int) -> dict[str, Any]:
    return await _get_json(f"event/{event_id}/live/", ttl=TTL_EVENT_LIVE)


async def _manager_info(manager_id: int) -> dict[str, Any]:
    return await _get_json(f"entry/{manager_id}/", ttl=TTL_BOOTSTRAP)


async def _manager_history(manager_id: int) -> dict[str, Any]:
    return await _get_json(f"entry/{manager_id}/history/", ttl=TTL_BOOTSTRAP)


async def _manager_picks(manager_id: int, event_id: int) -> dict[str, Any]:
    return await _get_json(f"entry/{manager_id}/event/{event_id}/picks/", ttl=TTL_EVENT)


async def _manager_transfers(manager_id: int) -> dict[str, Any]:
    return await _get_json(f"entry/{manager_id}/transfers/", ttl=TTL_BOOTSTRAP)


def _price_m(now_cost: int) -> float:
    return now_cost / 10.0


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _current_event_id(events: list[dict[str, Any]]) -> int | None:
    cur = next((e for e in events if e.get("is_current")), None)
    if cur:
        return int(cur["id"])
    nxt = next((e for e in events if e.get("is_next")), None)
    if nxt:
        return int(nxt["id"])
    return None


def _fixture_difficulty_for_team(fx: dict[str, Any], team_id: int) -> int | None:
    if fx.get("team_h") == team_id:
        return int(fx.get("team_h_difficulty") or 0) or None
    if fx.get("team_a") == team_id:
        return int(fx.get("team_a_difficulty") or 0) or None
    return None


def _calculate_team_strength(teams: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """
    Calculate attack and defense strength multipliers for each team relative to league average.

    Uses bootstrap strength_attack_home, strength_attack_away, strength_defence_home,
    strength_defence_away fields. Returns multipliers where 1.0 = league average.

    Returns dict of team_id -> {
        "attack_home": float,
        "attack_away": float,
        "defense_home": float,  # Lower = better defense
        "defense_away": float,
    }
    """
    if not teams:
        return {}

    # Calculate league averages
    attack_home_vals = [_to_float(t.get("strength_attack_home", 1000)) for t in teams]
    attack_away_vals = [_to_float(t.get("strength_attack_away", 1000)) for t in teams]
    defense_home_vals = [_to_float(t.get("strength_defence_home", 1000)) for t in teams]
    defense_away_vals = [_to_float(t.get("strength_defence_away", 1000)) for t in teams]

    avg_attack_home = sum(attack_home_vals) / len(teams) if teams else 1000.0
    avg_attack_away = sum(attack_away_vals) / len(teams) if teams else 1000.0
    avg_defense_home = sum(defense_home_vals) / len(teams) if teams else 1000.0
    avg_defense_away = sum(defense_away_vals) / len(teams) if teams else 1000.0

    strength: dict[int, dict[str, float]] = {}
    for t in teams:
        team_id = int(t["id"])
        strength[team_id] = {
            "attack_home": _to_float(t.get("strength_attack_home", 1000)) / avg_attack_home if avg_attack_home else 1.0,
            "attack_away": _to_float(t.get("strength_attack_away", 1000)) / avg_attack_away if avg_attack_away else 1.0,
            "defense_home": _to_float(t.get("strength_defence_home", 1000)) / avg_defense_home if avg_defense_home else 1.0,
            "defense_away": _to_float(t.get("strength_defence_away", 1000)) / avg_defense_away if avg_defense_away else 1.0,
        }

    return strength


# Position mean per-90 stats for Bayesian shrinkage (based on typical PL season averages)
POSITION_MEANS_PER90 = {
    "GKP": {"xg": 0.0, "xa": 0.01, "xgi": 0.01, "points": 3.5},
    "DEF": {"xg": 0.04, "xa": 0.05, "xgi": 0.09, "points": 4.0},
    "MID": {"xg": 0.15, "xa": 0.12, "xgi": 0.27, "points": 4.5},
    "FWD": {"xg": 0.35, "xa": 0.10, "xgi": 0.45, "points": 4.8},
}


def _calculate_team_attacking_context(
    team_id: int,
    team_strength: dict[int, dict[str, float]],
    elements: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Calculate team attacking quality context for a player's team.

    Uses ACTUAL goals scored (not just FPL strength ratings) to determine
    attack quality. This catches teams like Arsenal where goals come from
    all positions, not just traditional attackers.

    Returns dict with attack strength, team goals, goal distribution, and quality label.
    """
    # Get FPL strength ratings (for reference, but not primary quality measure)
    ts = team_strength.get(team_id, {})
    attack_home = ts.get("attack_home", 1.0)
    attack_away = ts.get("attack_away", 1.0)
    fpl_attack_rating = (attack_home + attack_away) / 2

    # Calculate team's total goals and xG by position
    team_players = [el for el in elements if int(el.get("team", 0)) == team_id]
    team_xg = sum(_to_float(el.get("expected_goals", 0)) for el in team_players)
    team_xa = sum(_to_float(el.get("expected_assists", 0)) for el in team_players)
    team_goals = sum(int(_to_float(el.get("goals_scored", 0))) for el in team_players)
    team_assists = sum(int(_to_float(el.get("assists", 0))) for el in team_players)

    # Goals by position (to understand goal spread)
    def_goals = sum(
        int(_to_float(el.get("goals_scored", 0)))
        for el in team_players
        if int(el.get("element_type", 0)) == 2  # DEF
    )
    mid_goals = sum(
        int(_to_float(el.get("goals_scored", 0)))
        for el in team_players
        if int(el.get("element_type", 0)) == 3  # MID
    )
    fwd_goals = sum(
        int(_to_float(el.get("goals_scored", 0)))
        for el in team_players
        if int(el.get("element_type", 0)) == 4  # FWD
    )

    # Get team FPL points for attackers (MID + FWD)
    attacker_points = sum(
        int(_to_float(el.get("total_points", 0)))
        for el in team_players
        if int(el.get("element_type", 0)) in (3, 4)
    )

    # Calculate ALL teams' goals to rank this team
    all_team_ids = set(int(el.get("team", 0)) for el in elements)
    team_goal_totals: list[tuple[int, int]] = []
    for tid in all_team_ids:
        t_goals = sum(
            int(_to_float(el.get("goals_scored", 0)))
            for el in elements
            if int(el.get("team", 0)) == tid
        )
        team_goal_totals.append((tid, t_goals))

    # Sort by goals descending to get ranking
    team_goal_totals.sort(key=lambda x: x[1], reverse=True)
    goals_rank = next((i + 1 for i, (tid, _) in enumerate(team_goal_totals) if tid == team_id), 10)

    # Calculate league average goals
    total_goals = sum(g for _, g in team_goal_totals)
    avg_team_goals = total_goals / len(team_goal_totals) if team_goal_totals else 30.0

    # Use ACTUAL goals (rank) to determine quality, not FPL strength rating
    # This fixes the Arsenal issue where goals come from all positions
    if goals_rank <= 3:
        attack_quality = "elite_attack"
    elif goals_rank <= 6:
        attack_quality = "strong_attack"
    elif goals_rank <= 12:
        attack_quality = "average_attack"
    elif goals_rank <= 17:
        attack_quality = "weak_attack"
    else:
        attack_quality = "poor_attack"

    # Goal spread description
    if team_goals > 0:
        def_pct = round(def_goals / team_goals * 100, 1) if team_goals else 0
        fwd_pct = round(fwd_goals / team_goals * 100, 1) if team_goals else 0
        if def_pct >= 20:
            goal_spread = "goals_from_all_positions"
        elif fwd_pct >= 60:
            goal_spread = "striker_dependent"
        else:
            goal_spread = "balanced_midfield_attack"
    else:
        goal_spread = "no_goals"

    return {
        "attack_quality": attack_quality,
        "goals_rank": goals_rank,
        "team_goals": team_goals,
        "team_xg": round(team_xg, 1),
        "goals_vs_league_avg": round(team_goals - avg_team_goals, 1),
        "goal_distribution": {
            "from_defenders": def_goals,
            "from_midfielders": mid_goals,
            "from_forwards": fwd_goals,
            "spread_type": goal_spread,
        },
        "team_xa": round(team_xa, 1),
        "team_assists": team_assists,
        "fpl_attack_rating": round(fpl_attack_rating, 2),
        "fpl_attack_home": round(attack_home, 2),
        "fpl_attack_away": round(attack_away, 2),
        "attacker_fpl_points": attacker_points,
    }


def _shrink_per90_stats(
    player_per90: dict[str, float],
    minutes: int,
    position: str,
) -> dict[str, float]:
    """
    Apply Bayesian shrinkage to per-90 stats based on sample size (minutes).

    For players with low minutes, shrink their per-90 stats toward the position mean.
    This prevents overweighting fluky performances from small samples.

    Args:
        player_per90: Dict with keys like "xg", "xa", "xgi", "points" (per-90 values)
        minutes: Total minutes played this season
        position: Player position (GKP, DEF, MID, FWD)

    Returns:
        Dict with shrunk per-90 values. Full weight at 900+ minutes,
        linear interpolation below.
    """
    # Full trust at 900+ minutes (10 full games), zero trust at 0 minutes
    full_weight_minutes = 900
    weight = min(1.0, minutes / full_weight_minutes)

    position_means = POSITION_MEANS_PER90.get(position, POSITION_MEANS_PER90["MID"])

    shrunk: dict[str, float] = {}
    for stat, player_val in player_per90.items():
        mean_val = position_means.get(stat, 0.0)
        # Weighted average: (weight * player) + ((1 - weight) * mean)
        shrunk[stat] = round((weight * player_val) + ((1.0 - weight) * mean_val), 4)

    return shrunk


def _calculate_confidence_score(
    el: dict[str, Any],
    avg_minutes: float,
    season_minutes: int,
) -> dict[str, Any]:
    """
    Calculate a 0-100 confidence score for projections.

    Higher score = more confident in the projection.
    Based on: minutes sample, nailed status, availability, form consistency.

    Returns dict with score and breakdown.
    """
    components: dict[str, float] = {}

    # 1. Sample size confidence (0-30 points)
    # Full confidence at 1800+ minutes (20 games), scales linearly
    sample_score = min(30, (season_minutes / 1800) * 30)
    components["sample_size"] = round(sample_score, 1)

    # 2. Nailed starter confidence (0-25 points)
    # Full confidence at 85+ avg minutes, scales linearly from 45
    if avg_minutes >= 85:
        nailed_score = 25
    elif avg_minutes >= 45:
        nailed_score = ((avg_minutes - 45) / 40) * 25
    else:
        nailed_score = 0
    components["nailed_status"] = round(nailed_score, 1)

    # 3. Availability confidence (0-25 points)
    status = str(el.get("status", "a"))
    chance = el.get("chance_of_playing_next_round")

    if status == "a" and chance is None:
        avail_score = 25  # Fully available, no concerns
    elif status == "a" and chance is not None:
        avail_score = 15 + (_to_float(chance, 100) / 100) * 10  # 15-25 based on chance
    elif status == "d":  # Doubtful
        avail_score = 5
    else:  # Injured, suspended, unavailable
        avail_score = 0
    components["availability"] = round(avail_score, 1)

    # 4. Form consistency confidence (0-20 points)
    # Based on form rating (higher form = more consistent recent performance)
    form = _to_float(el.get("form", 0))
    if form >= 6:
        form_score = 20
    elif form >= 4:
        form_score = 15
    elif form >= 2:
        form_score = 10
    elif form > 0:
        form_score = 5
    else:
        form_score = 0
    components["form_consistency"] = round(form_score, 1)

    total_score = sum(components.values())

    # Confidence level label
    if total_score >= 80:
        level = "very_high"
    elif total_score >= 60:
        level = "high"
    elif total_score >= 40:
        level = "medium"
    elif total_score >= 20:
        level = "low"
    else:
        level = "very_low"

    return {
        "score": round(total_score, 0),
        "level": level,
        "components": components,
    }


def _calculate_head_to_head_probability(
    player_a: dict[str, Any],
    player_b: dict[str, Any],
) -> dict[str, Any]:
    """
    Estimate probability that player A outscores player B in the next gameweek.

    Uses expected points as baseline, adjusted by:
    - Confidence scores (more certain projections get more weight)
    - Recent form variance (inconsistent players have wider outcome distributions)

    Returns dict with probability breakdown.
    """
    # Get expected points
    xpts_a = player_a.get("expected_points", 0.0)
    xpts_b = player_b.get("expected_points", 0.0)

    # Get confidence scores (default to medium if not present)
    conf_a = player_a.get("confidence", {}).get("score", 50) / 100
    conf_b = player_b.get("confidence", {}).get("score", 50) / 100

    # Get form variance proxy from signals
    signals_a = player_a.get("signals", {})
    signals_b = player_b.get("signals", {})

    # Higher playing probability = more predictable
    play_prob_a = signals_a.get("playing_probability", 0.75)
    play_prob_b = signals_b.get("playing_probability", 0.75)

    # Estimate standard deviation based on expected points and confidence
    # Lower confidence or lower playing prob = higher uncertainty
    # Typical FPL std dev is ~2-4 points per game
    base_std = 3.0
    std_a = base_std * (1.5 - (conf_a * 0.5)) * (1.5 - (play_prob_a * 0.5))
    std_b = base_std * (1.5 - (conf_b * 0.5)) * (1.5 - (play_prob_b * 0.5))

    # Using simplified probability calculation:
    # P(A > B) where A ~ N(μ_a, σ_a²) and B ~ N(μ_b, σ_b²)
    # A - B ~ N(μ_a - μ_b, σ_a² + σ_b²)
    # P(A > B) = P(A - B > 0) = Φ((μ_a - μ_b) / sqrt(σ_a² + σ_b²))
    import math

    mean_diff = xpts_a - xpts_b
    combined_std = math.sqrt(std_a**2 + std_b**2)

    # Standard normal CDF approximation
    if combined_std < 0.001:
        # Avoid division by zero
        prob_a_wins = 0.5 if abs(mean_diff) < 0.001 else (1.0 if mean_diff > 0 else 0.0)
    else:
        z = mean_diff / combined_std
        # Approximation of standard normal CDF
        prob_a_wins = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    # Calculate expected points advantage
    xpts_diff = round(xpts_a - xpts_b, 2)

    # Confidence in the comparison
    comparison_confidence = round((conf_a + conf_b) / 2 * 100, 0)

    # Determine recommendation strength
    if prob_a_wins >= 0.7:
        recommendation = "strongly_prefer_a"
    elif prob_a_wins >= 0.55:
        recommendation = "slightly_prefer_a"
    elif prob_a_wins <= 0.3:
        recommendation = "strongly_prefer_b"
    elif prob_a_wins <= 0.45:
        recommendation = "slightly_prefer_b"
    else:
        recommendation = "toss_up"

    return {
        "player_a_id": player_a.get("id"),
        "player_b_id": player_b.get("id"),
        "player_a_name": player_a.get("name"),
        "player_b_name": player_b.get("name"),
        "prob_a_outscores_b": round(prob_a_wins * 100, 1),
        "prob_b_outscores_a": round((1 - prob_a_wins) * 100, 1),
        "xpts_difference": xpts_diff,
        "comparison_confidence": comparison_confidence,
        "recommendation": recommendation,
    }


def _calculate_outcome_range(
    el: dict[str, Any],
    expected_points: float,
    playing_probability: float,
    confidence_score: float,
) -> dict[str, Any]:
    """
    Calculate floor, ceiling, and haul probability for a player.

    Uses expected points and variance estimation to project outcome ranges.

    Returns dict with floor (10th percentile), ceiling (90th percentile),
    and probability of haul (10+ points).
    """
    import math

    pos = POS_MAP.get(int(el.get("element_type", 3)), "MID")

    # Base standard deviation varies by position
    # Attackers more volatile, defenders more consistent
    position_volatility = {
        "GKP": 2.5,  # Lower variance - mostly 2-6 points
        "DEF": 3.0,  # Moderate variance - clean sheet dependent
        "MID": 3.5,  # Higher variance - goal involvements vary
        "FWD": 4.0,  # Highest variance - feast or famine
    }
    base_std = position_volatility.get(pos, 3.5)

    # Adjust standard deviation based on confidence
    # Lower confidence = wider range of outcomes
    confidence_factor = 1.5 - (confidence_score / 100 * 0.5)  # 1.0 to 1.5
    adjusted_std = base_std * confidence_factor

    # Account for playing probability - not playing = 0 points
    # This creates a bimodal distribution but we'll approximate

    # Floor (10th percentile)
    # If playing_prob < 0.9, floor could be 0 (doesn't play)
    if playing_probability < 0.5:
        floor = 0.0
    elif playing_probability < 0.9:
        # Blend between 0 and statistical floor
        stat_floor = max(0, expected_points - 1.28 * adjusted_std)
        floor = stat_floor * (playing_probability - 0.5) / 0.4
    else:
        floor = max(0, expected_points - 1.28 * adjusted_std)

    # Ceiling (90th percentile)
    ceiling = expected_points + 1.28 * adjusted_std

    # Haul probability (10+ points)
    # P(X >= 10) = 1 - Phi((10 - μ) / σ)
    if adjusted_std > 0.001:
        z_haul = (10 - expected_points) / adjusted_std
        prob_haul = 0.5 * (1 - math.erf(z_haul / math.sqrt(2)))
    else:
        prob_haul = 1.0 if expected_points >= 10 else 0.0

    # Adjust haul probability by playing probability
    prob_haul *= playing_probability

    # Blank probability (2 or fewer points, i.e., didn't start or no returns)
    if adjusted_std > 0.001:
        z_blank = (2 - expected_points) / adjusted_std
        prob_blank_if_plays = 0.5 * (1 + math.erf(z_blank / math.sqrt(2)))
    else:
        prob_blank_if_plays = 1.0 if expected_points <= 2 else 0.0

    # Account for not playing as a blank
    prob_blank = prob_blank_if_plays * playing_probability + (1 - playing_probability)

    return {
        "floor": round(floor, 1),
        "ceiling": round(ceiling, 1),
        "haul_probability": round(prob_haul * 100, 1),
        "blank_probability": round(prob_blank * 100, 1),
        "expected_range": f"{round(floor, 1)}-{round(ceiling, 1)}",
    }


def _calculate_form_trajectory(
    history: list[dict[str, Any]],
    recent_matches: int = 5,
) -> dict[str, Any]:
    """
    Analyze form trajectory to detect hot/cold streaks.

    Compares recent performance vs season average to identify:
    - Hot streaks (significantly outperforming season average)
    - Cold streaks (significantly underperforming)
    - Heating up (improving trend)
    - Cooling down (declining trend)

    Args:
        history: Player match history from element-summary
        recent_matches: How many recent matches to consider

    Returns dict with trajectory analysis.
    """
    if not history or len(history) < 3:
        return {
            "trajectory": "insufficient_data",
            "trend": None,
            "streak": None,
            "recent_vs_season": None,
        }

    # Get season totals
    season_pts = sum(int(_to_float(h.get("total_points", 0))) for h in history)
    season_mins = sum(int(_to_float(h.get("minutes", 0))) for h in history)
    season_ppg = season_pts / len(history) if history else 0

    # Get recent matches
    recent = history[-min(recent_matches, len(history)):]
    recent_pts = sum(int(_to_float(h.get("total_points", 0))) for h in recent)
    recent_mins = sum(int(_to_float(h.get("minutes", 0))) for h in recent)
    recent_ppg = recent_pts / len(recent) if recent else 0

    # Performance ratio: recent vs season
    perf_ratio = recent_ppg / season_ppg if season_ppg > 0 else 1.0

    # Calculate trend: compare first half of recent vs second half
    if len(recent) >= 4:
        mid = len(recent) // 2
        first_half_ppg = sum(int(_to_float(h.get("total_points", 0))) for h in recent[:mid]) / mid
        second_half_ppg = sum(int(_to_float(h.get("total_points", 0))) for h in recent[mid:]) / (len(recent) - mid)
        trend_ratio = second_half_ppg / first_half_ppg if first_half_ppg > 0 else 1.0
    else:
        trend_ratio = 1.0

    # Detect streaks (consecutive good or bad games)
    streak_count = 0
    streak_type = None
    for h in reversed(recent):
        pts = int(_to_float(h.get("total_points", 0)))
        if pts >= 6:  # Good game
            if streak_type is None:
                streak_type = "returns"
            elif streak_type == "returns":
                streak_count += 1
            else:
                break
        elif pts <= 2:  # Blank
            if streak_type is None:
                streak_type = "blanks"
            elif streak_type == "blanks":
                streak_count += 1
            else:
                break
        else:
            break

    streak = None
    if streak_count >= 2 and streak_type:
        streak = f"{streak_count + 1}_consecutive_{streak_type}"

    # Determine trajectory label
    if perf_ratio >= 1.3 and trend_ratio >= 1.0:
        trajectory = "hot_streak"
    elif perf_ratio >= 1.15 or trend_ratio >= 1.2:
        trajectory = "heating_up"
    elif perf_ratio <= 0.7 and trend_ratio <= 1.0:
        trajectory = "cold_streak"
    elif perf_ratio <= 0.85 or trend_ratio <= 0.8:
        trajectory = "cooling_down"
    else:
        trajectory = "stable"

    # Check for big haul recently
    recent_max = max(int(_to_float(h.get("total_points", 0))) for h in recent) if recent else 0
    has_recent_haul = recent_max >= 10

    return {
        "trajectory": trajectory,
        "recent_ppg": round(recent_ppg, 2),
        "season_ppg": round(season_ppg, 2),
        "performance_vs_season": round((perf_ratio - 1) * 100, 1),  # % above/below season avg
        "trend": "improving" if trend_ratio >= 1.15 else ("declining" if trend_ratio <= 0.85 else "steady"),
        "trend_strength": round((trend_ratio - 1) * 100, 1),
        "streak": streak,
        "recent_max_points": recent_max,
        "has_recent_haul": has_recent_haul,
        "matches_analyzed": len(recent),
    }


def _fixture_congestion_factor(
    fixtures: list[dict[str, Any]],
    team_id: int,
    target_event: int,
    window: int = 3,
) -> float:
    """
    Calculate a congestion multiplier based on fixture density around target_event.

    Returns multiplier (0.9-1.0) where lower values indicate higher congestion
    and potential for rotation/fatigue.

    Args:
        fixtures: Full fixtures list
        team_id: Team to check
        target_event: The gameweek to analyze
        window: How many gameweeks before/after to consider

    Returns:
        Multiplier between 0.9 (congested) and 1.0 (normal)
    """
    event_range = range(target_event - window, target_event + window + 1)
    fixture_count = 0

    for fx in fixtures:
        ev = fx.get("event")
        if ev is None:
            continue
        if int(ev) not in event_range:
            continue
        if fx.get("team_h") == team_id or fx.get("team_a") == team_id:
            fixture_count += 1

    # Normal is ~window*2+1 fixtures (one per GW)
    expected_fixtures = window * 2 + 1
    congestion_ratio = fixture_count / expected_fixtures if expected_fixtures else 1.0

    # If ratio > 1.2 (20% more fixtures than normal), apply penalty
    if congestion_ratio > 1.2:
        # Scale: 1.2 ratio = 0.98, 1.5 ratio = 0.92, 2.0 ratio = 0.90
        penalty = min(0.10, (congestion_ratio - 1.0) * 0.20)
        return max(0.90, 1.0 - penalty)

    return 1.0


def _availability_penalty(el: dict[str, Any], base_xpts: float | None = None) -> float:
    """
    Calculate availability penalty for flagged/injured players.

    If base_xpts is provided, returns a proportional penalty (percentage of base_xpts).
    Otherwise returns legacy fixed penalties for backward compatibility.

    Proportional mode penalties:
    - Non-available status: 30% of base_xpts
    - chance < 75%: additional 20% of base_xpts
    - chance < 50%: additional 20% of base_xpts (cumulative with above)
    """
    status = str(el.get("status", "a"))
    chance = el.get("chance_of_playing_next_round", None)

    if base_xpts is not None and base_xpts > 0:
        # Proportional penalty mode
        penalty_pct = 0.0
        if status != "a":
            penalty_pct += 0.30  # 30% penalty for non-available status
        if chance is not None:
            c = _to_float(chance, 100.0)
            if c < 75:
                penalty_pct += 0.20  # 20% penalty for low chance
            if c < 50:
                penalty_pct += 0.20  # Additional 20% for very low chance
        return base_xpts * min(penalty_pct, 0.70)  # Cap at 70% of base_xpts

    # Legacy fixed penalty mode for backward compatibility
    penalty = 0.0
    if status != "a":
        penalty += 3.0
    if chance is not None:
        c = _to_float(chance, 100.0)
        if c < 75:
            penalty += 2.0
        if c < 50:
            penalty += 2.0
    return penalty


def _playing_probability(el: dict[str, Any], avg_minutes: float = 90.0) -> float:
    """
    Estimate probability of playing any minutes based on status, news, and recent minutes.
    This is NOT a "60+ minutes" probability—it's the chance of appearing at all,
    adjusted for rotation risk based on avg_minutes.
    Returns 0.0 to 1.0.
    """
    status = str(el.get("status", "a"))
    chance = el.get("chance_of_playing_next_round")

    # Base probability from status
    if status == "a":
        base = 1.0
    elif status == "d":  # Doubtful
        base = 0.5
    elif status == "i":  # Injured
        base = 0.0
    elif status == "s":  # Suspended
        base = 0.0
    elif status == "u":  # Unavailable
        base = 0.0
    else:
        base = 0.75

    # Override with explicit chance if available
    if chance is not None:
        base = min(base, _to_float(chance, 100.0) / 100.0)

    # Adjust for minutes trend (rotation risk)
    if avg_minutes < 45:
        base *= 0.5
    elif avg_minutes < 60:
        base *= 0.7
    elif avg_minutes < 75:
        base *= 0.85

    return min(1.0, max(0.0, base))


def _calculate_expected_points(
    el: dict[str, Any],
    fixture_difficulty: float,
    is_home: bool,
    playing_prob: float,
    avg_minutes: float,
    opponent_defense_strength: float | None = None,
) -> dict[str, Any]:
    """
    Calculate expected points for a single gameweek using xG/xA and fixture context.

    Args:
        opponent_defense_strength: Opponent's defensive strength multiplier (1.0 = average).
            Lower values = weaker defense = better attacking opportunities.
            If provided, used to further adjust xG/xA output.

    Returns breakdown of expected points by category.
    """
    pos = POS_MAP.get(int(el.get("element_type", 3)), "MID")
    coeffs = XPT_COEFFS.get(pos, XPT_COEFFS["MID"])

    minutes = int(_to_float(el.get("minutes")))
    games_played = max(1, minutes / 90.0)

    # Per-game xG and xA from season totals
    xg_season = _to_float(el.get("expected_goals"))
    xa_season = _to_float(el.get("expected_assists"))
    xg_per_game = xg_season / games_played if games_played > 0 else 0.0
    xa_per_game = xa_season / games_played if games_played > 0 else 0.0

    # Adjust for fixture difficulty (1=easy, 5=hard)
    # Easy fixtures boost attacking output, hard fixtures reduce it
    difficulty_factor = 1.0 + (3.0 - fixture_difficulty) * 0.1  # Range: 0.8 to 1.2
    home_boost = 1.1 if is_home else 1.0

    # Adjust for opponent defense strength (if provided)
    # Lower defense strength = easier to score against
    # Multiplier: invert so weaker defense (0.8) becomes boost (1.25), stronger defense (1.2) becomes penalty (0.83)
    opp_defense_factor = 1.0
    if opponent_defense_strength is not None and opponent_defense_strength > 0:
        # Dampen the effect to avoid extreme swings (max ±15% adjustment)
        opp_defense_factor = 1.0 + (1.0 - opponent_defense_strength) * 0.15
        opp_defense_factor = max(0.85, min(1.15, opp_defense_factor))

    xg_adj = xg_per_game * difficulty_factor * home_boost * opp_defense_factor
    xa_adj = xa_per_game * difficulty_factor * home_boost * opp_defense_factor

    # Expected points from goals and assists
    xpts_goals = xg_adj * coeffs["goal"]
    xpts_assists = xa_adj * coeffs["assist"]

    # Clean sheet probability (for defenders/GKPs)
    # Base CS rate adjusted by opponent's xG
    cs_prob = 0.0
    if pos in ("GKP", "DEF"):
        # Rough estimate: easier fixtures = higher CS probability
        base_cs_rate = 0.35  # ~35% of games result in CS on average
        cs_prob = base_cs_rate * (1.0 + (3.0 - fixture_difficulty) * 0.15)
        cs_prob = min(0.6, max(0.1, cs_prob))  # Cap between 10-60%
    xpts_cs = cs_prob * coeffs.get("cs", 0)

    # Minutes points (2 for 60+, 1 for 1-59)
    prob_60_plus = playing_prob * min(avg_minutes / 90.0, 1.0)
    prob_1_to_59 = playing_prob * (1.0 - prob_60_plus / playing_prob) if playing_prob > 0 else 0.0
    xpts_minutes = (prob_60_plus * 2.0) + (prob_1_to_59 * 1.0)

    # Bonus points estimate (based on ICT index ranking)
    ict = _to_float(el.get("ict_index"))
    ict_per_game = ict / games_played if games_played > 0 else 0.0
    # Top ICT performers (~10+) get avg 1.5 bonus, mid-range (~5) get ~0.5
    xpts_bonus = min(2.0, max(0.0, (ict_per_game - 3.0) * 0.2))

    # GKP saves (rough estimate)
    xpts_saves = 0.0
    if pos == "GKP":
        # Assume ~3 saves per game on average
        saves_per_game = 3.0 * (1.0 + (fixture_difficulty - 3.0) * 0.1)
        xpts_saves = saves_per_game / coeffs.get("saves_per_pt", 3)

    # Apply playing_prob to all components EXCEPT minutes (which already has it baked in)
    xpts_from_play = (xpts_goals + xpts_assists + xpts_cs + xpts_bonus + xpts_saves) * playing_prob
    total_xpts = xpts_from_play + xpts_minutes

    return {
        "expected_points": round(total_xpts, 2),
        "breakdown": {
            "xpts_goals": round(xpts_goals * playing_prob, 3),
            "xpts_assists": round(xpts_assists * playing_prob, 3),
            "xpts_clean_sheet": round(xpts_cs * playing_prob, 3),
            "xpts_minutes": round(xpts_minutes, 3),  # Already includes playing_prob
            "xpts_bonus": round(xpts_bonus * playing_prob, 3),
            "xpts_saves": round(xpts_saves * playing_prob, 3),
        },
        "adjustments": {
            "playing_probability": round(playing_prob, 3),
            "fixture_difficulty": fixture_difficulty,
            "difficulty_factor": round(difficulty_factor, 3),
            "opponent_defense_factor": round(opp_defense_factor, 3),
            "is_home": is_home,
            "xg_per_game": round(xg_per_game, 3),
            "xa_per_game": round(xa_per_game, 3),
        },
    }


def _calculate_multi_gw_xpts(
    el: dict[str, Any],
    fixtures: list[dict[str, Any]],
    teams_by_id: dict[int, dict[str, Any]],
    current_event: int | None,
    horizon_gws: int,
    avg_minutes: float = 90.0,
    team_strength: dict[int, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    Calculate expected points over multiple gameweeks.

    Args:
        team_strength: Optional dict of team_id -> strength metrics from _calculate_team_strength.
            If provided, opponent defensive strength is used to adjust xG/xA.

    Applies fixture congestion factor to reduce expected points during busy periods.
    """
    team_id = int(el["team"])
    playing_prob = _playing_probability(el, avg_minutes)

    gw_xpts: list[dict[str, Any]] = []
    total_xpts = 0.0
    congestion_applied = False

    if current_event is not None:
        for fx in fixtures:
            ev = fx.get("event")
            if ev is None:
                continue
            ev = int(ev)
            if ev < current_event or ev >= current_event + horizon_gws:
                continue

            # Check if this fixture involves our team
            if fx.get("team_h") == team_id:
                difficulty = int(_to_float(fx.get("team_h_difficulty"), 3))
                is_home = True
                opp_id = fx.get("team_a")
            elif fx.get("team_a") == team_id:
                difficulty = int(_to_float(fx.get("team_a_difficulty"), 3))
                is_home = False
                opp_id = fx.get("team_h")
            else:
                continue

            # Get opponent defense strength if available
            opp_defense = None
            if team_strength and opp_id:
                opp_strength = team_strength.get(int(opp_id), {})
                # Use away defense if we're home, home defense if we're away
                opp_defense = opp_strength.get("defense_away" if is_home else "defense_home")

            # Calculate fixture congestion factor for this gameweek
            congestion_factor = _fixture_congestion_factor(fixtures, team_id, ev)
            if congestion_factor < 1.0:
                congestion_applied = True

            # Adjust playing probability for congestion (rotation risk increases)
            adjusted_playing_prob = playing_prob * congestion_factor

            gw_calc = _calculate_expected_points(
                el, difficulty, is_home, adjusted_playing_prob, avg_minutes, opp_defense
            )

            gw_xpts_value = gw_calc["expected_points"]
            gw_xpts.append({
                "event": ev,
                "opponent": teams_by_id.get(int(opp_id), {}).get("name") if opp_id else None,
                "is_home": is_home,
                "difficulty": difficulty,
                "expected_points": gw_xpts_value,
                "congestion_factor": round(congestion_factor, 3),
            })
            total_xpts += gw_xpts_value

    return {
        "total_expected_points": round(total_xpts, 2),
        "gameweeks": gw_xpts,
        "avg_xpts_per_gw": round(total_xpts / len(gw_xpts), 2) if gw_xpts else 0.0,
        "playing_probability": round(playing_prob, 3),
        "congestion_adjusted": congestion_applied,
    }


def _validate_squad(
    squad_ids: list[int],
    elements_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate a squad against FPL rules.

    Returns validation result with any violations.
    """
    violations: list[str] = []
    warnings: list[str] = []

    # Count positions
    pos_counts: dict[str, int] = {"GKP": 0, "DEF": 0, "MID": 0, "FWD": 0}
    team_counts: dict[int, int] = {}
    total_value = 0.0

    for pid in squad_ids:
        el = elements_by_id.get(pid)
        if el is None:
            violations.append(f"Unknown player ID: {pid}")
            continue

        pos = POS_MAP.get(int(el.get("element_type", 3)), "MID")
        team_id = int(el.get("team", 0))
        price = _price_m(int(el.get("now_cost", 0)))

        pos_counts[pos] = pos_counts.get(pos, 0) + 1
        team_counts[team_id] = team_counts.get(team_id, 0) + 1
        total_value += price

    # Check squad size
    if len(squad_ids) != SQUAD_SIZE:
        violations.append(f"Squad size is {len(squad_ids)}, must be {SQUAD_SIZE}")

    # Check position limits
    for pos, required in SQUAD_COMPOSITION.items():
        actual = pos_counts.get(pos, 0)
        if actual != required:
            violations.append(f"{pos}: have {actual}, need exactly {required}")

    # Check team limits
    for team_id, count in team_counts.items():
        if count > MAX_PLAYERS_PER_TEAM:
            violations.append(f"Team {team_id}: have {count} players, max is {MAX_PLAYERS_PER_TEAM}")

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
        "position_counts": pos_counts,
        "team_counts": team_counts,
        "total_value_m": round(total_value, 1),
    }


def _can_transfer_in(
    player_in: dict[str, Any],
    player_out: dict[str, Any],
    current_squad_ids: set[int],
    elements_by_id: dict[int, dict[str, Any]],
    bank: float,
    selling_price: float | None = None,
) -> dict[str, Any]:
    """
    Check if a transfer is valid according to FPL rules.

    Returns dict with 'valid' bool and 'reasons' list if invalid.

    NOTE: selling_price defaults to now_cost, but FPL uses the half-profit rule:
    selling_price = purchase_price + (now_cost - purchase_price) // 2
    Without access to purchase history, we use now_cost as an approximation.
    This may overestimate available funds for players who have risen in price.
    """
    reasons: list[str] = []

    pid_in = int(player_in["id"])
    pid_out = int(player_out["id"])
    pos_in = POS_MAP.get(int(player_in.get("element_type", 3)), "MID")
    pos_out = POS_MAP.get(int(player_out.get("element_type", 3)), "MID")
    team_in = int(player_in.get("team", 0))

    # Must be same position
    if pos_in != pos_out:
        reasons.append(f"Position mismatch: {pos_out} → {pos_in}")

    # Check if player already in squad
    if pid_in in current_squad_ids:
        reasons.append(f"Player {pid_in} already in squad")

    # Check team limit (count current team members excluding player out)
    team_count = sum(
        1 for pid in current_squad_ids
        if pid != pid_out and int(elements_by_id.get(pid, {}).get("team", 0)) == team_in
    )
    if team_count >= MAX_PLAYERS_PER_TEAM:
        reasons.append(f"Already have {MAX_PLAYERS_PER_TEAM} players from team {team_in}")

    # Check budget
    price_in = _price_m(int(player_in.get("now_cost", 0)))
    if selling_price is None:
        selling_price = _price_m(int(player_out.get("now_cost", 0)))

    available = bank + selling_price
    if price_in > available:
        reasons.append(f"Cannot afford: need £{price_in}m, have £{round(available, 1)}m")

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "price_in": price_in,
        "selling_price": selling_price,
        "bank_after": round(available - price_in, 1) if len(reasons) == 0 else None,
    }


def _player_identity(el: dict[str, Any], teams_by_id: dict[int, dict[str, Any]]) -> dict[str, Any]:
    team_id = int(el["team"])
    name = f"{el.get('first_name','')} {el.get('second_name','')}".strip() or str(el.get("web_name"))
    return {
        "id": int(el["id"]),
        "name": name,
        "web_name": el.get("web_name"),
        "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
        "team_id": team_id,
        "position": POS_MAP.get(int(el["element_type"]), "MID"),
        "price_m": round(_price_m(int(el["now_cost"])), 1),
        "status": el.get("status"),
        "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
    }


def _resolve_players(
    elements: list[dict[str, Any]],
    ids: list[int],
    names: list[str],
    limit_per_name: int,
) -> tuple[list[dict[str, Any]], list[int], list[str]]:
    """
    Resolve players by ids and/or partial name matches.
    Returns (elements, missing_ids, unmatched_names).
    """
    by_id: dict[int, dict[str, Any]] = {int(el["id"]): el for el in elements}
    seen_ids: set[int] = set()
    resolved: list[dict[str, Any]] = []
    missing_ids: list[int] = []
    unmatched_names: list[str] = []

    for pid in ids:
        pid_int = int(pid)
        el = by_id.get(pid_int)
        if el is None:
            missing_ids.append(pid_int)
            continue
        if pid_int not in seen_ids:
            resolved.append(el)
            seen_ids.add(pid_int)

    for raw_name in names:
        q = str(raw_name).strip().lower()
        if not q:
            continue
        hits: list[dict[str, Any]] = []
        for el in elements:
            if len(hits) >= max(1, limit_per_name):
                break
            full = f"{el.get('first_name','')} {el.get('second_name','')}".strip().lower()
            web = str(el.get("web_name", "")).lower()
            if q in full or q in web:
                pid_int = int(el["id"])
                if pid_int in seen_ids:
                    continue
                hits.append(el)
                seen_ids.add(pid_int)
        if hits:
            resolved.extend(hits)
        else:
            unmatched_names.append(raw_name)

    return resolved, missing_ids, unmatched_names


def _history_vs_opponent(
    history: list[dict[str, Any]], opponent_id: int, sample: int
) -> dict[str, Any] | None:
    games = [h for h in history if int(_to_float(h.get("opponent_team"))) == opponent_id]
    if not games:
        return None
    window = games[-max(1, sample) :]
    mins = sum(int(_to_float(h.get("minutes"))) for h in window)
    pts = sum(int(_to_float(h.get("total_points"))) for h in window)
    blanks = sum(1 for h in window if int(_to_float(h.get("total_points"))) <= 2)
    return {
        "matches_used": len(window),
        "avg_points": round(pts / len(window), 3) if window else 0.0,
        "points_per_90": round((pts / mins) * 90.0, 3) if mins else 0.0,
        "avg_minutes": round(mins / len(window), 2) if window else 0.0,
        "blank_rate": round(blanks / len(window), 3) if window else None,
    }


def _player_snapshot(
    el: dict[str, Any],
    es: dict[str, Any],
    teams_by_id: dict[int, dict[str, Any]],
    fixture_horizon: int,
    last_matches: int,
    history_slice: int,
) -> dict[str, Any]:
    ident = _player_identity(el, teams_by_id)
    recent = _recent_form_from_element_summary(es, last_matches=last_matches)
    history: list[dict[str, Any]] = es.get("history", []) or []
    season_ppg = _to_float(el.get("points_per_game"))
    form = _to_float(el.get("form"))
    price = ident["price_m"]
    value = (season_ppg / price) if price else 0.0

    upcoming_raw = es.get("fixtures", []) or []
    upcoming: list[dict[str, Any]] = []
    for fx in upcoming_raw:
        if len(upcoming) >= max(1, fixture_horizon):
            break
        opp_id = fx.get("opponent_team")
        opp_id_int = int(_to_float(opp_id)) if opp_id is not None else None
        upcoming.append(
            {
                "event": fx.get("event"),
                "opponent_team": opp_id_int,
                "opponent_name": teams_by_id.get(opp_id_int, {}).get("name") if opp_id_int else None,
                "is_home": fx.get("is_home"),
                "difficulty": fx.get("difficulty"),
                "kickoff_time": fx.get("kickoff_time"),
            }
        )

    matchup_insights: list[dict[str, Any]] = []
    if history:
        for fx in upcoming:
            opp_id = fx.get("opponent_team")
            if opp_id is None:
                continue
            vs = _history_vs_opponent(history, opp_id, sample=history_slice)
            if vs:
                matchup_insights.append(
                    {
                        "opponent_team": opp_id,
                        "opponent_name": fx.get("opponent_name"),
                        **vs,
                    }
                )

    risk_flags: list[str] = []
    if str(el.get("status", "a")) != "a":
        risk_flags.append("flagged_or_unavailable")
    cop = el.get("chance_of_playing_next_round")
    if cop is not None and _to_float(cop, 100.0) < 75:
        risk_flags.append("low_chance_next_round")
    if recent.get("avg_minutes", 0.0) < 60:
        risk_flags.append("rotation_risk")
    if (recent.get("blank_rate") or 0.0) > 0.6:
        risk_flags.append("high_recent_blank_rate")

    history_recent = history[-max(0, history_slice) :] if history_slice > 0 else []

    return {
        "player": ident,
        "season": {
            "points_per_game": season_ppg,
            "form": form,
            "ict_index": _to_float(el.get("ict_index")),
            "expected_goal_involvements": _to_float(el.get("expected_goal_involvements", 0.0)),
            "value_ppg_per_million": round(value, 3),
            "minutes": int(_to_float(el.get("minutes"))),
            "threat": _to_float(el.get("threat")),
            "creativity": _to_float(el.get("creativity")),
            "influence": _to_float(el.get("influence")),
        },
        "recent": recent,
        "trend": {
            "recent_points_per_90_vs_season_ppg_delta": round(recent.get("points_per_90", 0.0) - season_ppg, 3),
            "minutes_trend_flag": "low" if recent.get("avg_minutes", 0.0) < 60 else "stable",
            "value_signal": "good_value" if value >= 1.0 else "neutral",
        },
        "upcoming_fixtures": upcoming,
        "history_vs_next_opponents": matchup_insights,
        "history_recent": history_recent,
        "risk_flags": risk_flags,
    }


def _trim_players(
    elements: list[dict[str, Any]],
    limit: int,
    fields: list[str] | None,
) -> list[dict[str, Any]]:
    """
    Trim the players list to keep responses manageable.
    Default ordering is by total_points then form to preserve top performers.
    """
    chosen = elements
    if limit and limit > 0:
        chosen = sorted(
            elements,
            key=lambda el: (
                _to_float(el.get("total_points")),
                _to_float(el.get("form")),
            ),
            reverse=True,
        )[:limit]

    if not fields:
        return chosen
    return [{f: el.get(f) for f in fields} for el in chosen]


def _team_fixture_outlook(
    fixtures: list[dict[str, Any]],
    teams_by_id: dict[int, dict[str, Any]],
    current_event: int | None,
    horizon_gws: int,
) -> dict[int, dict[str, Any]]:
    """
    Summarise upcoming fixture difficulty per team for quick tactical context.
    """
    outlook: dict[int, dict[str, Any]] = {}
    if current_event is None:
        return outlook

    for fx in fixtures:
        ev = fx.get("event")
        if ev is None:
            continue
        ev = int(ev)
        if ev < current_event or ev >= current_event + max(1, horizon_gws):
            continue

        for team_key, opp_key, diff_key, home_flag in (
            ("team_h", "team_a", "team_h_difficulty", True),
            ("team_a", "team_h", "team_a_difficulty", False),
        ):
            team_id = fx.get(team_key)
            opp_id = fx.get(opp_key)
            if team_id is None or opp_id is None:
                continue
            team_id = int(team_id)
            opp_id = int(opp_id)
            difficulty = int(_to_float(fx.get(diff_key), 0))
            entry = outlook.setdefault(
                team_id,
                {
                    "team": teams_by_id.get(team_id, {}).get("name"),
                    "team_id": team_id,
                    "matches": 0,
                    "total_difficulty": 0,
                    "easy": 0,
                    "hard": 0,
                    "next_opponents": [],
                },
            )
            entry["matches"] += 1
            entry["total_difficulty"] += difficulty
            if difficulty <= 2:
                entry["easy"] += 1
            if difficulty >= 4:
                entry["hard"] += 1
            entry["next_opponents"].append(
                {
                    "event": ev,
                    "opponent_team": opp_id,
                    "opponent_name": teams_by_id.get(opp_id, {}).get("name"),
                    "difficulty": difficulty,
                    "is_home": home_flag,
                }
            )

    for entry in outlook.values():
        matches = max(1, entry["matches"])
        entry["avg_difficulty"] = round(entry["total_difficulty"] / matches, 3)

    return outlook


def _score_player_first_pass(
    el: dict[str, Any],
    teams_by_id: dict[int, dict[str, Any]],
    fixtures: list[dict[str, Any]],
    horizon_gws: int,
    current_event: int | None,
    avg_minutes: float | None = None,
    weights: dict[str, float] | None = None,
    team_strength: dict[int, dict[str, float]] | None = None,
    elements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Score player using expected points model.

    Uses xG/xA-based projections adjusted for fixture difficulty.

    Args:
        weights: Optional dict with keys "xpts", "form", "value" to customize scoring.
            Defaults: {"xpts": 1.0, "form": 0.3, "value": 0.5}
        team_strength: Optional dict from _calculate_team_strength for opponent-adjusted xG/xA.
        elements: Optional list of all elements for team context calculation.
    """
    # Default weights
    w = {"xpts": 1.0, "form": 0.3, "value": 0.5}
    if weights:
        w.update(weights)

    team_id = int(el["team"])
    pos = POS_MAP.get(int(el["element_type"]), "MID")
    price = _price_m(int(el["now_cost"]))

    ppg = _to_float(el.get("points_per_game"))
    form = _to_float(el.get("form"))
    ict = _to_float(el.get("ict_index"))
    minutes = int(_to_float(el.get("minutes")))
    games_played = max(1, minutes / 90.0)

    # Calculate average minutes per game for playing probability
    if avg_minutes is None:
        avg_minutes = minutes / games_played if games_played > 1 else 90.0

    # Get multi-gameweek expected points (with team strength adjustment if available)
    xpts_data = _calculate_multi_gw_xpts(
        el, fixtures, teams_by_id, current_event, horizon_gws, avg_minutes, team_strength
    )

    total_xpts = xpts_data["total_expected_points"]
    playing_prob = xpts_data["playing_probability"]

    # Value metric: expected points per million spent over horizon
    value = (total_xpts / price) if price else 0.0

    # Availability penalty for flagged players
    penalty = _availability_penalty(el)

    # Final score using customizable weights
    # We prioritize xPts but give small boosts for:
    # - Recent form (shows current performance level)
    # - Value (points per million is important for budget)
    score = (total_xpts * w["xpts"]) + (form * w["form"]) + (value * w["value"]) - penalty

    team_name = teams_by_id.get(team_id, {}).get("name", str(team_id))
    name = f"{el.get('first_name','')} {el.get('second_name','')}".strip() or str(el.get("web_name"))

    # xG and xA stats
    xg = _to_float(el.get("expected_goals"))
    xa = _to_float(el.get("expected_assists"))
    xgi = _to_float(el.get("expected_goal_involvements"))

    # Actual goals and assists
    actual_goals = int(_to_float(el.get("goals_scored", 0)))
    actual_assists = int(_to_float(el.get("assists", 0)))

    # Overperformance/underperformance calculation
    # Positive = overperforming (scoring above xG, may regress DOWN)
    # Negative = underperforming (scoring below xG, may regress UP - "buy low" candidate)
    goal_overperformance = actual_goals - xg
    assist_overperformance = actual_assists - xa
    total_overperformance = goal_overperformance + assist_overperformance

    # Determine regression signal
    if total_overperformance > 1.5:
        regression_risk = "high_overperformer_regression_risk"
    elif total_overperformance > 0.5:
        regression_risk = "slight_overperformer"
    elif total_overperformance < -1.5:
        regression_risk = "strong_buy_signal_underperformer"
    elif total_overperformance < -0.5:
        regression_risk = "potential_buy_underperformer"
    else:
        regression_risk = "performing_as_expected"

    # Calculate confidence score for this projection
    confidence = _calculate_confidence_score(el, avg_minutes, minutes)

    # Calculate outcome range (floor, ceiling, haul probability)
    outcome_range = _calculate_outcome_range(
        el, total_xpts, playing_prob, confidence["score"]
    )

    # Calculate team attacking context (if elements available)
    team_context = None
    if elements and team_strength:
        team_context = _calculate_team_attacking_context(team_id, team_strength, elements)

    result = {
        "id": int(el["id"]),
        "name": name,
        "team": team_name,
        "team_id": team_id,
        "position": pos,
        "price_m": round(price, 1),
        "expected_points": round(total_xpts, 2),
        "base_score": round(score, 3),
        "signals": {
            "points_per_game": ppg,
            "form": form,
            "ict_index": ict,
            "xg_season": round(xg, 2),
            "xa_season": round(xa, 2),
            "xgi_season": round(xgi, 2),
            "xg_per_game": round(xg / games_played, 3) if games_played > 0 else 0.0,
            "xa_per_game": round(xa / games_played, 3) if games_played > 0 else 0.0,
            "value_xpts_per_million": round(value, 3),
            "playing_probability": playing_prob,
            "availability_penalty": penalty,
            "minutes_season": minutes,
            "avg_minutes_per_game": round(avg_minutes, 1),
            # Actual vs expected
            "goals_scored": actual_goals,
            "assists": actual_assists,
            "goal_overperformance": round(goal_overperformance, 2),
            "assist_overperformance": round(assist_overperformance, 2),
            "total_overperformance": round(total_overperformance, 2),
            "regression_risk": regression_risk,
            # Shot volume proxies (threat = best proxy for shots/chances in FPL data)
            "threat": _to_float(el.get("threat", 0)),
            "threat_per_90": round(_to_float(el.get("threat", 0)) / games_played, 2) if games_played > 0 else 0.0,
            "creativity": _to_float(el.get("creativity", 0)),
            "creativity_per_90": round(_to_float(el.get("creativity", 0)) / games_played, 2) if games_played > 0 else 0.0,
            "influence": _to_float(el.get("influence", 0)),
            "influence_per_90": round(_to_float(el.get("influence", 0)) / games_played, 2) if games_played > 0 else 0.0,
        },
        "confidence": confidence,
        "outcome_range": outcome_range,
        "team_context": team_context,
        "fixture_xpts": xpts_data["gameweeks"],
    }
    # Generate explanation after building the result
    result["explanation"] = _generate_explanation(result)
    return result


def _recent_form_from_element_summary(es: dict[str, Any], last_matches: int) -> dict[str, Any]:
    """
    Extract recent trend signals from element-summary history.

    Works even if some expected fields are missing by falling back safely.
    Also includes form trajectory analysis (hot/cold detection).
    """
    history: list[dict[str, Any]] = es.get("history", []) or []
    if not history:
        return {
            "matches_used": 0,
            "avg_minutes": 0.0,
            "points_per_90": 0.0,
            "xgi_per_90": 0.0,
            "blank_rate": None,
            "form_trajectory": {"trajectory": "insufficient_data"},
        }

    last = history[-max(1, min(last_matches, len(history))):]

    mins = sum(int(_to_float(h.get("minutes"))) for h in last)
    pts = sum(int(_to_float(h.get("total_points"))) for h in last)

    # Prefer expected_goal_involvements if present; otherwise fall back to expected_goals + expected_assists
    xgi_sum = 0.0
    blanks = 0
    for h in last:
        xgi = h.get("expected_goal_involvements", None)
        if xgi is not None:
            xgi_sum += _to_float(xgi)
        else:
            xgi_sum += _to_float(h.get("expected_goals")) + _to_float(h.get("expected_assists"))

        if int(_to_float(h.get("total_points"))) <= 2:
            blanks += 1

    matches_used = len(last)
    avg_minutes = mins / matches_used if matches_used else 0.0
    points_per_90 = (pts / mins) * 90.0 if mins > 0 else 0.0
    xgi_per_90 = (xgi_sum / mins) * 90.0 if mins > 0 else 0.0
    blank_rate = (blanks / matches_used) if matches_used else None

    # Calculate form trajectory using full history
    form_trajectory = _calculate_form_trajectory(history, recent_matches=last_matches)

    return {
        "matches_used": matches_used,
        "avg_minutes": round(avg_minutes, 2),
        "points_per_90": round(points_per_90, 3),
        "xgi_per_90": round(xgi_per_90, 3),
        "blank_rate": None if blank_rate is None else round(blank_rate, 3),
        "form_trajectory": form_trajectory,
    }


def _generate_explanation(
    scored: dict[str, Any],
    recent: dict[str, Any] | None = None,
) -> str:
    """
    Generate a human-readable explanation of why a player scored well/poorly.

    Returns a concise string highlighting key factors.
    """
    factors: list[str] = []

    # Expected points assessment
    xpts = scored.get("expected_points", 0.0)
    if xpts >= 6.0:
        factors.append("strong xPts output")
    elif xpts >= 4.0:
        factors.append("solid xPts")
    elif xpts < 2.5:
        factors.append("low xPts projection")

    # Fixture assessment
    fixture_xpts = scored.get("fixture_xpts", [])
    if fixture_xpts:
        easy_fixtures = sum(1 for f in fixture_xpts if f.get("difficulty", 3) <= 2)
        hard_fixtures = sum(1 for f in fixture_xpts if f.get("difficulty", 3) >= 4)
        if easy_fixtures >= len(fixture_xpts) * 0.6:
            factors.append("favorable fixtures")
        elif hard_fixtures >= len(fixture_xpts) * 0.6:
            factors.append("difficult fixture run")

    # Signals assessment
    signals = scored.get("signals", {})
    form = signals.get("form", 0.0)
    if form >= 7.0:
        factors.append("excellent form")
    elif form >= 5.0:
        factors.append("good form")
    elif form < 3.0 and form > 0:
        factors.append("poor recent form")

    value = signals.get("value_xpts_per_million", 0.0)
    if value >= 1.5:
        factors.append("great value")
    elif value >= 1.0:
        factors.append("good value")

    penalty = signals.get("availability_penalty", 0.0)
    if penalty > 2.0:
        factors.append("injury/availability concern")
    elif penalty > 0:
        factors.append("minor doubt flag")

    playing_prob = signals.get("playing_probability", 1.0)
    if playing_prob < 0.7:
        factors.append("rotation risk")

    # Overperformance/underperformance signals
    regression_risk = signals.get("regression_risk", "")
    if regression_risk == "high_overperformer_regression_risk":
        factors.append("CAUTION: overperforming xG, regression likely")
    elif regression_risk == "strong_buy_signal_underperformer":
        factors.append("BUY SIGNAL: underperforming xG, due for goals")
    elif regression_risk == "potential_buy_underperformer":
        factors.append("underperforming xG slightly")

    # Recent trends (if available from refinement)
    if recent:
        avg_mins = recent.get("avg_minutes", 0.0)
        if avg_mins >= 80:
            factors.append("nailed starter")
        elif avg_mins < 60 and avg_mins > 0:
            factors.append("minutes concern")

        xgi90 = recent.get("xgi_per_90", 0.0)
        if xgi90 >= 0.6:
            factors.append("elite xGI/90")
        elif xgi90 >= 0.4:
            factors.append("strong underlying stats")

        blank_rate = recent.get("blank_rate")
        if blank_rate is not None and blank_rate > 0.5:
            factors.append("recent blank streak")

    if not factors:
        return "Average profile across key metrics"

    return "; ".join(factors)


def _refine_score(
    base: dict[str, Any],
    recent: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Second-pass adjustment:
    - Reward: recent xGI/90, recent points/90, strong minutes trend
    - Penalise: low average minutes (rotation risk)
    - Apply Bayesian shrinkage to per-90 stats for low-minutes players

    Args:
        weights: Optional dict with keys "xgi90", "p90" to customize scoring.
            Defaults: {"xgi90": 2.2, "p90": 0.9}
    """
    # Default weights
    w = {"xgi90": 2.2, "p90": 0.9}
    if weights:
        w.update(weights)

    base_score = float(base.get("base_score", 0.0))
    avg_minutes = float(recent.get("avg_minutes", 0.0))
    p90_raw = float(recent.get("points_per_90", 0.0))
    xgi90_raw = float(recent.get("xgi_per_90", 0.0))

    # Apply Bayesian shrinkage to per-90 stats based on season minutes
    # This regresses stats toward position mean for low-sample players
    position = base.get("position", "MID")
    season_minutes = base.get("signals", {}).get("minutes_season", 0)

    shrunk_stats = _shrink_per90_stats(
        {"xgi": xgi90_raw, "points": p90_raw},
        minutes=season_minutes,
        position=position,
    )
    xgi90 = shrunk_stats.get("xgi", xgi90_raw)
    p90 = shrunk_stats.get("points", p90_raw)

    minutes_factor = min(avg_minutes / 90.0, 1.0)
    rotation_pen = 0.0
    if avg_minutes > 0 and avg_minutes < 60:
        rotation_pen = 1.5
    elif avg_minutes >= 60 and avg_minutes < 75:
        rotation_pen = 0.6

    refined = (
        base_score
        + (xgi90 * w["xgi90"])
        + (p90 * w["p90"])
        + (minutes_factor * 1.2)
        - rotation_pen
    )

    out = dict(base)
    out["refined_score"] = round(refined, 3)
    out["recent_signals"] = recent
    out["adjustments"] = {
        "rotation_penalty": rotation_pen,
        "minutes_factor": round(minutes_factor, 3),
        "xgi90_weight": w["xgi90"],
        "p90_weight": w["p90"],
        "shrinkage_applied": season_minutes < 900,
        "xgi90_raw": round(xgi90_raw, 4),
        "xgi90_shrunk": round(xgi90, 4),
        "p90_raw": round(p90_raw, 4),
        "p90_shrunk": round(p90, 4),
    }
    # Update explanation with recent data context
    out["explanation"] = _generate_explanation(out, recent)
    return out


def _apply_risk_profile(
    scored: dict[str, Any],
    risk_level: str,
    elements_by_id: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Apply risk profile adjustments to a scored player.

    Args:
        scored: Scored player dict (from first pass or refined)
        risk_level: "low", "medium", or "high"
        elements_by_id: Optional dict for looking up ownership data

    Returns:
        Updated scored dict with risk_adjusted_score
    """
    out = dict(scored)
    base = float(scored.get("refined_score", scored.get("base_score", 0.0)))

    # Get ownership percentage
    player_id = scored.get("id")
    ownership_pct = 0.0
    if elements_by_id and player_id:
        el = elements_by_id.get(int(player_id), {})
        ownership_pct = _to_float(el.get("selected_by_percent", 0.0))

    signals = scored.get("signals", {})
    recent = scored.get("recent_signals", {})

    adjustment = 0.0
    risk_factors: list[str] = []

    if risk_level == "low":
        # Prefer proven performers, penalize uncertainty
        # Bonus for high ownership (template players)
        if ownership_pct >= 20:
            adjustment += 0.5
            risk_factors.append("high_ownership_bonus")

        # Bonus for consistent minutes
        avg_mins = recent.get("avg_minutes", 0.0) if recent else signals.get("avg_minutes_per_game", 0.0)
        if avg_mins >= 80:
            adjustment += 0.8
            risk_factors.append("nailed_starter_bonus")

        # Penalty for rotation risk
        playing_prob = signals.get("playing_probability", 1.0)
        if playing_prob < 0.9:
            adjustment -= 1.0
            risk_factors.append("rotation_penalty")

        # Penalty for flagged players
        availability_penalty = signals.get("availability_penalty", 0.0)
        if availability_penalty > 0:
            adjustment -= 1.5
            risk_factors.append("availability_penalty")

        # Penalty for low minutes sample
        minutes = signals.get("minutes_season", 0)
        if minutes < 500:
            adjustment -= 0.5
            risk_factors.append("small_sample_penalty")

    elif risk_level == "high":
        # Prefer differentials and recent outperformers
        # Bonus for low ownership
        if ownership_pct <= 5:
            adjustment += 1.2
            risk_factors.append("low_ownership_bonus")
        elif ownership_pct <= 10:
            adjustment += 0.6
            risk_factors.append("differential_bonus")

        # Bonus for recent xGI outperformance
        xgi90 = recent.get("xgi_per_90", 0.0) if recent else 0.0
        if xgi90 >= 0.5:
            adjustment += 0.8
            risk_factors.append("hot_underlying_stats")

        # Bonus for form spike (form > PPG suggests hot streak)
        form = signals.get("form", 0.0)
        ppg = signals.get("points_per_game", 0.0)
        if form > ppg * 1.3 and form >= 5.0:
            adjustment += 0.6
            risk_factors.append("form_spike_bonus")

        # Small penalty for template players (already priced in)
        if ownership_pct >= 30:
            adjustment -= 0.3
            risk_factors.append("template_penalty")

    # Medium risk = no adjustment (baseline)

    out["risk_adjusted_score"] = round(base + adjustment, 3)
    out["risk_profile"] = {
        "level": risk_level,
        "adjustment": round(adjustment, 3),
        "ownership_pct": round(ownership_pct, 1),
        "factors": risk_factors,
    }
    return out


def _require_bearer(request: Request) -> Response | None:
    if not BEARER_TOKEN:
        return None
    auth = request.headers.get("authorization", "")
    if auth != f"Bearer {BEARER_TOKEN}":
        return JSONResponse({"error": "unauthorised"}, status_code=401)
    return None


# --------------------
# MCP server + tools
# --------------------
server = Server("fpl-advisor")

TOOLS: list[Tool] = [
    Tool(
        name="fpl_find_players",
        description="Find FPL players by partial name match (first/second/web_name).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name search string"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="fpl_player_summary",
        description="Player snapshot using element-summary history (recent minutes, points/90, xGI/90) + upcoming fixtures.",
        inputSchema={
            "type": "object",
            "properties": {
                "player_id": {"type": "integer", "description": "FPL element id"},
                "last_matches": {"type": "integer", "default": 5, "description": "How many recent matches to analyse"},
            },
            "required": ["player_id"],
        },
    ),
    Tool(
        name="fpl_players_bulk",
        description="Bulk fetch players by ids or partial names with rich context (season form, recent trends, upcoming fixtures, matchup signals).",
        inputSchema={
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": "integer"}, "description": "List of FPL element ids"},
                "names": {"type": "array", "items": {"type": "string"}, "description": "Partial names/web_names to match"},
                "limit_per_name": {
                    "type": "integer",
                    "default": 3,
                    "description": "When searching by names, how many matches to return per query",
                },
                "last_matches": {
                    "type": "integer",
                    "default": 5,
                    "description": "Window used for recent trend metrics (points/90, xGI/90, minutes)",
                },
                "fixture_horizon": {"type": "integer", "default": 5, "description": "How many upcoming fixtures to include"},
                "history_slice": {
                    "type": "integer",
                    "default": 5,
                    "description": "How many recent past matches to include and to check vs upcoming opponents",
                },
                "concurrency": {"type": "integer", "default": 8, "description": "Max concurrent element-summary fetches"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_dataset",
        description="Full-context snapshot for LLM reasoning: bootstrap, fixtures, team outlook, optional live/event details, trimmed player list.",
        inputSchema={
            "type": "object",
            "properties": {
                "include_bootstrap": {"type": "boolean", "default": True, "description": "Include bootstrap-static summary"},
                "include_fixtures": {"type": "boolean", "default": True, "description": "Include full fixtures list"},
                "include_element_types": {"type": "boolean", "default": True, "description": "Include position metadata"},
                "events_window": {"type": "integer", "default": 5, "description": "How many upcoming events to keep from bootstrap events"},
                "fixture_horizon": {
                    "type": "integer",
                    "default": 5,
                    "description": "Horizon (gameweeks) for team fixture outlook summaries",
                },
                "include_team_outlook": {"type": "boolean", "default": True, "description": "Summarise upcoming ease per team"},
                "trim_players": {
                    "type": "integer",
                    "default": 200,
                    "description": "Trim players to top N by total_points (0 = include all; beware size)",
                },
                "player_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional subset of player fields to return from bootstrap elements",
                },
                "include_event_live_for": {
                    "type": "integer",
                    "description": "Event id for live stats (per-player live points, BPS, minutes) if in play",
                },
                "include_event_detail_for": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Event ids to fetch detail pages (chip plays, most captained, top elements). If empty, defaults to current event if known.",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_best_players",
        description="Fast rank using bootstrap + fixtures only (good for quick shortlist).",
        inputSchema={
            "type": "object",
            "properties": {
                "position": {"type": "string", "description": "Optional: GKP/DEF/MID/FWD"},
                "max_price_m": {"type": "number", "description": "Optional price cap in £m"},
                "horizon_gws": {"type": "integer", "default": 5, "description": "Fixture horizon (gameweeks)"},
                "limit": {"type": "integer", "default": 25, "description": "Max results"},
                "min_minutes": {"type": "integer", "default": 0, "description": "Filter low-minutes players"},
                "include_unavailable": {"type": "boolean", "default": False, "description": "Include flagged/unavailable"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_best_players_refined",
        description="Two-pass rank: quick shortlist (bootstrap+fixtures) then refine with element-summary trends (minutes/points/90/xGI/90). Supports custom scoring weights.",
        inputSchema={
            "type": "object",
            "properties": {
                "position": {"type": "string", "description": "Optional: GKP/DEF/MID/FWD"},
                "max_price_m": {"type": "number", "description": "Optional price cap in £m"},
                "horizon_gws": {"type": "integer", "default": 5, "description": "Fixture horizon (gameweeks)"},
                "limit": {"type": "integer", "default": 25, "description": "Max results"},
                "min_minutes": {"type": "integer", "default": 0, "description": "Filter low-minutes players"},
                "include_unavailable": {"type": "boolean", "default": False, "description": "Include flagged/unavailable"},
                "refine_pool": {"type": "integer", "default": 60, "description": "How many top candidates to enrich via element-summary"},
                "last_matches": {"type": "integer", "default": 5, "description": "Recent matches window for refinement"},
                "concurrency": {"type": "integer", "default": 8, "description": "Max concurrent element-summary fetches"},
                "xpts_weight": {"type": "number", "default": 1.0, "description": "Weight for expected points in scoring (default 1.0)"},
                "form_weight": {"type": "number", "default": 0.3, "description": "Weight for current form (default 0.3)"},
                "value_weight": {"type": "number", "default": 0.5, "description": "Weight for value (xPts/million) (default 0.5)"},
                "xgi90_weight": {"type": "number", "default": 2.2, "description": "Weight for recent xGI/90 in refinement (default 2.2)"},
                "p90_weight": {"type": "number", "default": 0.9, "description": "Weight for recent points/90 in refinement (default 0.9)"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium", "description": "Risk profile: 'low' favors proven performers, 'high' favors differentials"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_my_team",
        description="Analyze a manager's FPL team: current squad, form, fixtures, transfer history, overall rank progression.",
        inputSchema={
            "type": "object",
            "properties": {
                "manager_id": {"type": "integer", "description": "FPL manager/entry ID"},
                "event_id": {"type": "integer", "description": "Gameweek to analyze (defaults to current)"},
                "fixture_horizon": {"type": "integer", "default": 5, "description": "Upcoming fixtures to include"},
            },
            "required": ["manager_id"],
        },
    ),
    Tool(
        name="fpl_transfer_suggestions",
        description="Suggest transfers respecting ALL FPL rules: same position, max 3 per team, budget. Uses xG/xA expected points model.",
        inputSchema={
            "type": "object",
            "properties": {
                "manager_id": {"type": "integer", "description": "FPL manager/entry ID"},
                "positions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Positions to consider (GKP/DEF/MID/FWD). Defaults to all.",
                },
                "max_transfers": {"type": "integer", "default": 5, "description": "Max transfer suggestions to return"},
                "horizon_gws": {"type": "integer", "default": 5, "description": "Fixture horizon for expected points"},
            },
            "required": ["manager_id"],
        },
    ),
    Tool(
        name="fpl_validate_squad",
        description="Validate a squad against FPL rules: 15 players, position limits (2 GKP, 5 DEF, 5 MID, 3 FWD), max 3 per team.",
        inputSchema={
            "type": "object",
            "properties": {
                "manager_id": {"type": "integer", "description": "FPL manager ID to validate their current squad"},
                "player_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Alternative: provide player IDs directly instead of manager_id",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_differentials",
        description="Find low-ownership players with strong underlying stats—useful for rank climbing.",
        inputSchema={
            "type": "object",
            "properties": {
                "max_ownership_pct": {"type": "number", "default": 10.0, "description": "Maximum ownership percentage"},
                "min_form": {"type": "number", "default": 4.0, "description": "Minimum form rating"},
                "position": {"type": "string", "description": "Optional: GKP/DEF/MID/FWD"},
                "max_price_m": {"type": "number", "description": "Optional price cap in £m"},
                "min_minutes": {"type": "integer", "default": 200, "description": "Minimum season minutes"},
                "limit": {"type": "integer", "default": 20, "description": "Max results"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_dgw_bgw",
        description="Detect double and blank gameweeks—teams with multiple or zero fixtures in upcoming events.",
        inputSchema={
            "type": "object",
            "properties": {
                "event_start": {"type": "integer", "description": "Start gameweek (defaults to current)"},
                "event_end": {"type": "integer", "description": "End gameweek (defaults to start + 5)"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_captain_picks",
        description="Captaincy recommendations weighted for fixtures, home advantage, and penalty duties.",
        inputSchema={
            "type": "object",
            "properties": {
                "event_id": {"type": "integer", "description": "Gameweek (defaults to next)"},
                "limit": {"type": "integer", "default": 10, "description": "Max results"},
                "min_minutes": {"type": "integer", "default": 400, "description": "Minimum season minutes"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_compare",
        description="Side-by-side comparison of multiple players.",
        inputSchema={
            "type": "object",
            "properties": {
                "player_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of player IDs to compare (2-6 players)",
                },
                "last_matches": {"type": "integer", "default": 5, "description": "Recent matches for trend analysis"},
            },
            "required": ["player_ids"],
        },
    ),
    Tool(
        name="fpl_price_changes",
        description="Players likely to rise or fall in price based on transfer activity.",
        inputSchema={
            "type": "object",
            "properties": {
                "direction": {"type": "string", "description": "Filter by 'rising' or 'falling' (defaults to both)"},
                "limit": {"type": "integer", "default": 20, "description": "Max results per direction"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_deadline",
        description="Get the next gameweek deadline and time remaining.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="fpl_set_piece_takers",
        description="Identify set piece takers (penalties, corners, free kicks) by team.",
        inputSchema={
            "type": "object",
            "properties": {
                "team_id": {"type": "integer", "description": "Optional: filter to specific team"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_live_bps",
        description="Current bonus point standings for live/recent gameweek matches.",
        inputSchema={
            "type": "object",
            "properties": {
                "event_id": {"type": "integer", "description": "Gameweek (defaults to current)"},
            },
            "required": [],
        },
    ),
    Tool(
        name="fpl_player_enriched",
        description="Deep player analysis with external data from Understat and FBref. Provides shots on target, xG per shot, conversion rate, progressive passes, and more detailed stats than FPL API.",
        inputSchema={
            "type": "object",
            "properties": {
                "player_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "FPL player IDs to get enriched data for (1-5 players)",
                },
                "player_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Alternative: player names to search and enrich",
                },
            },
            "required": [],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "fpl_find_players":
        q = str(arguments.get("query", "")).strip().lower()
        limit = int(arguments.get("limit", 10))

        data = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in data.get("teams", [])}

        hits = []
        for el in data.get("elements", []):
            full = f"{el.get('first_name','')} {el.get('second_name','')}".strip().lower()
            web = str(el.get("web_name", "")).lower()
            if q and (q in full or q in web):
                team_id = int(el["team"])
                hits.append(
                    {
                        "id": int(el["id"]),
                        "name": f"{el.get('first_name','')} {el.get('second_name','')}".strip()
                        or el.get("web_name"),
                        "web_name": el.get("web_name"),
                        "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                        "position": POS_MAP.get(int(el["element_type"]), "MID"),
                        "price_m": round(_price_m(int(el["now_cost"])), 1),
                        "status": el.get("status"),
                    }
                )
            if len(hits) >= limit:
                break

        return [TextContent(type="text", text=json.dumps(hits, ensure_ascii=False))]

    if name == "fpl_player_summary":
        player_id = int(arguments.get("player_id"))
        last_matches = int(arguments.get("last_matches", 5))

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])

        el = next((x for x in elements if int(x.get("id")) == player_id), None)
        if el is None:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown player_id={player_id}"}))]

        es = await _element_summary(player_id)
        recent = _recent_form_from_element_summary(es, last_matches=last_matches)

        team_id = int(el["team"])
        payload = {
            "player": {
                "id": player_id,
                "name": f"{el.get('first_name','')} {el.get('second_name','')}".strip() or el.get("web_name"),
                "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                "position": POS_MAP.get(int(el["element_type"]), "MID"),
                "price_m": round(_price_m(int(el["now_cost"])), 1),
                "status": el.get("status"),
                "chance_of_playing_next_round": el.get("chance_of_playing_next_round"),
            },
            "recent": recent,
            "upcoming_fixtures": es.get("fixtures", [])[:10],
            "history_count": len(es.get("history", []) or []),
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_players_bulk":
        ids = list(arguments.get("ids") or [])
        names = list(arguments.get("names") or [])
        limit_per_name = int(arguments.get("limit_per_name", 3))
        last_matches = int(arguments.get("last_matches", 5))
        fixture_horizon = int(arguments.get("fixture_horizon", 5))
        history_slice = int(arguments.get("history_slice", 5))
        concurrency = int(arguments.get("concurrency", 8))

        if not ids and not names:
            return [TextContent(type="text", text=json.dumps({"error": "Provide at least one id or name"}, ensure_ascii=False))]

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])

        resolved, missing_ids, unmatched_names = _resolve_players(
            elements=elements,
            ids=[int(i) for i in ids],
            names=[str(n) for n in names],
            limit_per_name=max(1, limit_per_name),
        )

        if not resolved:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": "No players matched", "missing_ids": missing_ids, "unmatched_names": unmatched_names},
                        ensure_ascii=False,
                    ),
                )
            ]

        sem = asyncio.Semaphore(max(1, concurrency))

        async def enrich(el: dict[str, Any]) -> dict[str, Any]:
            pid = int(el["id"])
            async with sem:
                es = await _element_summary(pid)
            return _player_snapshot(
                el,
                es,
                teams_by_id=teams_by_id,
                fixture_horizon=fixture_horizon,
                last_matches=last_matches,
                history_slice=history_slice,
            )

        enriched = await asyncio.gather(*(enrich(el) for el in resolved))
        payload = {
            "count": len(enriched),
            "players": enriched,
            "missing_ids": missing_ids,
            "unmatched_names": unmatched_names,
            "params": {
                "fixture_horizon": fixture_horizon,
                "last_matches": last_matches,
                "history_slice": history_slice,
                "limit_per_name": limit_per_name,
            },
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_dataset":
        include_bootstrap = bool(arguments.get("include_bootstrap", True))
        include_fixtures = bool(arguments.get("include_fixtures", True))
        include_element_types = bool(arguments.get("include_element_types", True))
        events_window = int(arguments.get("events_window", 5))
        fixture_horizon = int(arguments.get("fixture_horizon", 5))
        include_team_outlook = bool(arguments.get("include_team_outlook", True))
        trim_players = int(arguments.get("trim_players", 200))
        player_fields_arg = arguments.get("player_fields")
        player_fields: list[str] | None = None
        if player_fields_arg:
            player_fields = [str(f) for f in player_fields_arg]
        else:
            player_fields = list(DEFAULT_PLAYER_FIELDS)

        live_event = arguments.get("include_event_live_for", None)
        event_detail_for = arguments.get("include_event_detail_for", None)

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        events = bs.get("events", [])
        current_event = _current_event_id(events)

        fixtures: list[dict[str, Any]] = []
        if include_fixtures or include_team_outlook:
            fixtures = await _fixtures()

        payload: dict[str, Any] = {"current_event": current_event}

        if include_bootstrap:
            events_subset = events
            if events_window > 0 and current_event is not None:
                events_subset = [
                    e
                    for e in events
                    if int(_to_float(e.get("id"))) >= current_event - 1
                    and int(_to_float(e.get("id"))) < current_event + events_window
                ]
            bootstrap_block: dict[str, Any] = {
                "total_players": bs.get("total_players"),
                "game_settings": bs.get("game_settings"),
                "teams": bs.get("teams"),
                "events": events_subset,
                "players": _trim_players(bs.get("elements", []), limit=trim_players, fields=player_fields),
            }
            if include_element_types:
                bootstrap_block["element_types"] = bs.get("element_types")
            payload["bootstrap"] = bootstrap_block

        if include_fixtures:
            payload["fixtures"] = fixtures

        if include_team_outlook:
            outlook = _team_fixture_outlook(fixtures, teams_by_id, current_event, horizon_gws=fixture_horizon)
            payload["team_outlook"] = list(outlook.values())

        if live_event is not None:
            live_event_id = int(live_event)
            live_data = await _event_live(live_event_id)
            payload["event_live"] = {"event": live_event_id, "live": live_data}

        detail_ids: list[int] = []
        if event_detail_for is None:
            if current_event is not None:
                detail_ids = [int(current_event)]
        else:
            detail_ids = [int(_to_float(ev)) for ev in event_detail_for if ev is not None]

        if detail_ids:
            detail_payloads = await asyncio.gather(*(_event(ev) for ev in detail_ids))
            payload["event_details"] = {"event_ids": detail_ids, "data": detail_payloads}

        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name in ("fpl_best_players", "fpl_best_players_refined"):
        position = arguments.get("position")
        max_price_m = arguments.get("max_price_m")
        horizon_gws = int(arguments.get("horizon_gws", 5))
        limit = int(arguments.get("limit", 25))
        min_minutes = int(arguments.get("min_minutes", 0))
        include_unavailable = bool(arguments.get("include_unavailable", False))

        refine_pool = int(arguments.get("refine_pool", 60))
        last_matches = int(arguments.get("last_matches", 5))
        concurrency = int(arguments.get("concurrency", 8))

        # Custom scoring weights (for fpl_best_players_refined)
        xpts_weight = float(arguments.get("xpts_weight", 1.0))
        form_weight = float(arguments.get("form_weight", 0.3))
        value_weight = float(arguments.get("value_weight", 0.5))
        xgi90_weight = float(arguments.get("xgi90_weight", 2.2))
        p90_weight = float(arguments.get("p90_weight", 0.9))
        risk_level = str(arguments.get("risk_level", "medium")).lower()
        if risk_level not in ("low", "medium", "high"):
            risk_level = "medium"

        first_pass_weights = {"xpts": xpts_weight, "form": form_weight, "value": value_weight}
        refine_weights = {"xgi90": xgi90_weight, "p90": p90_weight}

        bs = await _bootstrap()
        fx = await _fixtures()

        events = bs.get("events", [])
        current_event = _current_event_id(events)

        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])
        elements_by_id = {int(el["id"]): el for el in elements}

        # Calculate team strength once for opponent-adjusted projections
        team_strength = _calculate_team_strength(bs.get("teams", []))

        pos_filter: Position | None = None
        if isinstance(position, str) and position.strip():
            pos_filter = position.strip().upper()  # type: ignore[assignment]

        # First pass: fast score
        first_pass: list[dict[str, Any]] = []
        for el in elements:
            if pos_filter and POS_MAP.get(int(el["element_type"]), "MID") != pos_filter:
                continue
            if int(_to_float(el.get("minutes"))) < min_minutes:
                continue

            price = _price_m(int(el["now_cost"]))
            if max_price_m is not None and price > float(max_price_m):
                continue

            if not include_unavailable and str(el.get("status", "a")) != "a":
                continue

            first_pass.append(
                _score_player_first_pass(
                    el, teams_by_id, fx, horizon_gws=horizon_gws, current_event=current_event,
                    weights=first_pass_weights, team_strength=team_strength, elements=elements
                )
            )

        first_pass.sort(key=lambda r: r["base_score"], reverse=True)

        # If caller asked for fast only, return immediately
        if name == "fpl_best_players":
            payload = {
                "method": "first_pass_composite_v1",
                "current_event": current_event,
                "results": first_pass[: max(1, limit)],
            }
            return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

        # Second pass: refine top pool with element-summary trends
        pool = first_pass[: max(1, min(refine_pool, len(first_pass)))]
        sem = asyncio.Semaphore(max(1, concurrency))

        async def enrich_one(item: dict[str, Any]) -> dict[str, Any]:
            pid = int(item["id"])
            async with sem:
                es = await _element_summary(pid)
            recent = _recent_form_from_element_summary(es, last_matches=last_matches)
            return _refine_score(item, recent, weights=refine_weights)

        enriched = await asyncio.gather(*(enrich_one(it) for it in pool))

        # Apply risk profile adjustments
        enriched_with_risk = [
            _apply_risk_profile(item, risk_level, elements_by_id)
            for item in enriched
        ]

        # Rank by risk-adjusted score (or refined if no risk adjustment)
        sort_key = "risk_adjusted_score" if risk_level != "medium" else "refined_score"
        enriched_with_risk.sort(
            key=lambda r: r.get(sort_key, r.get("refined_score", r.get("base_score", 0.0))),
            reverse=True
        )
        results = enriched_with_risk[: max(1, limit)]

        payload = {
            "method": "two_pass_refined_v1",
            "current_event": current_event,
            "params": {
                "horizon_gws": horizon_gws,
                "limit": limit,
                "min_minutes": min_minutes,
                "position": position,
                "max_price_m": max_price_m,
                "include_unavailable": include_unavailable,
                "refine_pool": refine_pool,
                "last_matches": last_matches,
                "concurrency": concurrency,
                "risk_level": risk_level,
            },
            "weights": {
                "xpts_weight": xpts_weight,
                "form_weight": form_weight,
                "value_weight": value_weight,
                "xgi90_weight": xgi90_weight,
                "p90_weight": p90_weight,
            },
            "results": results,
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_my_team":
        manager_id = int(arguments.get("manager_id"))
        event_id_arg = arguments.get("event_id")
        fixture_horizon = int(arguments.get("fixture_horizon", 5))

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])
        elements_by_id = {int(el["id"]): el for el in elements}
        events = bs.get("events", [])
        current_event = _current_event_id(events)

        event_id = int(event_id_arg) if event_id_arg is not None else current_event
        if event_id is None:
            return [TextContent(type="text", text=json.dumps({"error": "No current event found"}))]

        try:
            manager_info, manager_hist, picks, transfers = await asyncio.gather(
                _manager_info(manager_id),
                _manager_history(manager_id),
                _manager_picks(manager_id, event_id),
                _manager_transfers(manager_id),
            )
        except httpx.HTTPStatusError as e:
            return [TextContent(type="text", text=json.dumps({"error": f"Manager not found or API error: {e.response.status_code}"}))]

        fixtures = await _fixtures()
        team_outlook = _team_fixture_outlook(fixtures, teams_by_id, current_event, horizon_gws=fixture_horizon)

        # Get squad IDs and validate
        squad_ids = [int(p["element"]) for p in picks.get("picks", [])]
        squad_validation = _validate_squad(squad_ids, elements_by_id)

        # Count players per team for constraint display
        team_player_count: dict[int, int] = {}
        for pid in squad_ids:
            el = elements_by_id.get(pid, {})
            team_id = int(el.get("team", 0))
            team_player_count[team_id] = team_player_count.get(team_id, 0) + 1

        squad = []
        total_expected_points = 0.0

        for pick in picks.get("picks", []):
            el_id = int(pick["element"])
            el = elements_by_id.get(el_id, {})
            team_id = int(el.get("team", 0))
            team_fixtures = team_outlook.get(team_id, {}).get("next_opponents", [])[:fixture_horizon]

            # Calculate expected points for this player
            minutes = int(_to_float(el.get("minutes")))
            games_played = max(1, minutes / 90.0)
            avg_minutes = minutes / games_played if games_played > 1 else 90.0

            xpts_data = _calculate_multi_gw_xpts(
                el, fixtures, teams_by_id, current_event, fixture_horizon, avg_minutes
            )

            expected_pts = xpts_data["total_expected_points"]
            total_expected_points += expected_pts

            # xG/xA stats
            xg = _to_float(el.get("expected_goals"))
            xa = _to_float(el.get("expected_assists"))
            xgi = _to_float(el.get("expected_goal_involvements"))

            squad.append({
                "id": el_id,
                "name": el.get("web_name", str(el_id)),
                "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                "team_id": team_id,
                "position": POS_MAP.get(int(el.get("element_type", 3)), "MID"),
                "price_m": round(_price_m(int(el.get("now_cost", 0))), 1),
                "is_captain": pick.get("is_captain", False),
                "is_vice_captain": pick.get("is_vice_captain", False),
                "multiplier": pick.get("multiplier", 1),
                "form": _to_float(el.get("form")),
                "points_per_game": _to_float(el.get("points_per_game")),
                "total_points": int(_to_float(el.get("total_points"))),
                "selected_by_percent": _to_float(el.get("selected_by_percent")),
                "status": el.get("status"),
                "chance_of_playing": el.get("chance_of_playing_next_round"),
                "playing_probability": xpts_data["playing_probability"],
                # xG/xA based metrics
                "xg_season": round(xg, 2),
                "xa_season": round(xa, 2),
                "xgi_season": round(xgi, 2),
                "xg_per_game": round(xg / games_played, 3) if games_played > 0 else 0.0,
                "xa_per_game": round(xa / games_played, 3) if games_played > 0 else 0.0,
                # Expected points over horizon
                "expected_points": expected_pts,
                "fixture_xpts": xpts_data["gameweeks"],
                "upcoming_fixtures": team_fixtures,
            })

        # Sort squad by position for display
        pos_order = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
        squad.sort(key=lambda x: (pos_order.get(x["position"], 4), -x["expected_points"]))

        current_history = manager_hist.get("current", [])
        recent_gws = current_history[-5:] if current_history else []

        recent_transfers = transfers[-10:] if isinstance(transfers, list) else []

        # Team counts with names
        team_counts_display = {
            teams_by_id.get(tid, {}).get("name", str(tid)): {
                "count": count,
                "at_limit": count >= MAX_PLAYERS_PER_TEAM,
            }
            for tid, count in team_player_count.items()
        }

        payload = {
            "manager": {
                "id": manager_id,
                "name": f"{manager_info.get('player_first_name', '')} {manager_info.get('player_last_name', '')}".strip(),
                "team_name": manager_info.get("name"),
                "overall_rank": manager_info.get("summary_overall_rank"),
                "overall_points": manager_info.get("summary_overall_points"),
                "gameweek_points": manager_info.get("summary_event_points"),
                "value": round(_price_m(int(manager_info.get("last_deadline_value", 0))), 1),
                "bank": round(_price_m(int(manager_info.get("last_deadline_bank", 0))), 1),
                "free_transfers": manager_info.get("last_deadline_total_transfers"),
            },
            "event_id": event_id,
            "active_chip": picks.get("active_chip"),
            "squad_analysis": {
                "total_expected_points": round(total_expected_points, 2),
                "avg_expected_per_player": round(total_expected_points / len(squad), 2) if squad else 0.0,
                "fixture_horizon": fixture_horizon,
                "squad_valid": squad_validation["valid"],
                "violations": squad_validation["violations"],
                "position_counts": squad_validation["position_counts"],
                "team_counts": team_counts_display,
                "total_value_m": squad_validation["total_value_m"],
            },
            "fpl_rules": {
                "max_per_team": MAX_PLAYERS_PER_TEAM,
                "squad_composition": SQUAD_COMPOSITION,
            },
            "squad": squad,
            "recent_gameweeks": recent_gws,
            "recent_transfers": recent_transfers,
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_transfer_suggestions":
        manager_id = int(arguments.get("manager_id"))
        positions_arg = arguments.get("positions")
        max_transfers = int(arguments.get("max_transfers", 3))
        horizon_gws = int(arguments.get("horizon_gws", 5))

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])
        elements_by_id = {int(el["id"]): el for el in elements}
        events = bs.get("events", [])
        current_event = _current_event_id(events)

        if current_event is None:
            return [TextContent(type="text", text=json.dumps({"error": "No current event found"}))]

        try:
            manager_info = await _manager_info(manager_id)
            picks = await _manager_picks(manager_id, current_event)
        except httpx.HTTPStatusError as e:
            return [TextContent(type="text", text=json.dumps({"error": f"Manager not found: {e.response.status_code}"}))]

        fixtures = await _fixtures()

        # Calculate team strength for opponent-adjusted projections
        team_strength = _calculate_team_strength(bs.get("teams", []))

        # Get bank value
        bank = _price_m(int(manager_info.get("last_deadline_bank", 0)))

        current_squad_ids = set(int(p["element"]) for p in picks.get("picks", []))

        # Validate current squad
        squad_validation = _validate_squad(list(current_squad_ids), elements_by_id)

        pos_filters: set[str] = set()
        if positions_arg:
            pos_filters = {str(p).strip().upper() for p in positions_arg}

        # Score all current squad players
        squad_scored: list[dict[str, Any]] = []
        squad_xpts_by_id: dict[int, float] = {}  # For quick lookup
        for pid in current_squad_ids:
            el = elements_by_id.get(pid, {})
            scored = _score_player_first_pass(el, teams_by_id, fixtures, horizon_gws, current_event, team_strength=team_strength, elements=elements)
            squad_scored.append(scored)
            squad_xpts_by_id[int(scored["id"])] = scored["expected_points"]

        # Calculate total team expected points before any transfer
        team_xpts_before = sum(squad_xpts_by_id.values())

        # Sort by expected points (ascending) to find weakest players
        squad_scored.sort(key=lambda x: x["expected_points"])

        suggestions = []
        seen_transfers: set[tuple[int, int]] = set()  # (out_id, in_id)

        # For each player in squad (starting from weakest), find valid upgrades
        for player_out in squad_scored:
            pos = player_out["position"]
            if pos_filters and pos not in pos_filters:
                continue

            out_id = player_out["id"]
            out_el = elements_by_id.get(out_id, {})
            selling_price = player_out["price_m"]

            # Find candidates for this position
            for el in elements:
                in_id = int(el["id"])

                # Skip if already in squad or same player
                if in_id in current_squad_ids:
                    continue

                # Must be same position
                in_pos = POS_MAP.get(int(el.get("element_type", 3)), "MID")
                if in_pos != pos:
                    continue

                # Skip unavailable players
                if str(el.get("status", "a")) != "a":
                    continue

                # Check all FPL constraints
                transfer_check = _can_transfer_in(
                    player_in=el,
                    player_out=out_el,
                    current_squad_ids=current_squad_ids,
                    elements_by_id=elements_by_id,
                    bank=bank,
                    selling_price=selling_price,
                )

                if not transfer_check["valid"]:
                    continue

                # Skip if we've already suggested this transfer pair
                if (out_id, in_id) in seen_transfers:
                    continue

                # Score the incoming player
                scored_in = _score_player_first_pass(el, teams_by_id, fixtures, horizon_gws, current_event, team_strength=team_strength, elements=elements)

                # Calculate improvement
                xpts_gain = scored_in["expected_points"] - player_out["expected_points"]

                # Only suggest if it's actually an improvement
                if xpts_gain <= 0:
                    continue

                seen_transfers.add((out_id, in_id))

                # Calculate team-level impact
                team_xpts_after = team_xpts_before - player_out["expected_points"] + scored_in["expected_points"]

                suggestions.append({
                    "out": {
                        "id": out_id,
                        "name": player_out["name"],
                        "team": player_out["team"],
                        "position": pos,
                        "price_m": selling_price,
                        "expected_points": player_out["expected_points"],
                    },
                    "in": {
                        "id": in_id,
                        "name": scored_in["name"],
                        "team": scored_in["team"],
                        "position": in_pos,
                        "price_m": scored_in["price_m"],
                        "expected_points": scored_in["expected_points"],
                        "form": scored_in["signals"]["form"],
                        "xg_per_game": scored_in["signals"]["xg_per_game"],
                        "xa_per_game": scored_in["signals"]["xa_per_game"],
                    },
                    "xpts_gain": round(xpts_gain, 2),
                    "cost_diff": round(scored_in["price_m"] - selling_price, 1),
                    "bank_after": transfer_check["bank_after"],
                    "constraints_check": {
                        "valid": True,
                        "same_position": True,
                        "within_budget": True,
                        "team_limit_ok": True,
                    },
                    "team_impact": {
                        "team_xpts_before": round(team_xpts_before, 2),
                        "team_xpts_after": round(team_xpts_after, 2),
                        "net_team_delta": round(team_xpts_after - team_xpts_before, 2),
                    },
                })

        # Sort by expected points gain
        suggestions.sort(key=lambda x: x["xpts_gain"], reverse=True)

        payload = {
            "manager_id": manager_id,
            "current_event": current_event,
            "bank": bank,
            "squad_validation": squad_validation,
            "squad_total_xpts": round(team_xpts_before, 2),
            "fpl_rules": {
                "max_per_team": MAX_PLAYERS_PER_TEAM,
                "squad_composition": SQUAD_COMPOSITION,
            },
            "suggestions": suggestions[:max_transfers],
            "note": "All suggestions respect FPL rules: same position, max 3 per team, and budget constraints.",
            "disclaimer": "Selling prices use current value (now_cost) as approximation. FPL uses half-profit rule which may reduce actual selling price for players who have risen.",
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_validate_squad":
        manager_id = arguments.get("manager_id")
        player_ids_arg = arguments.get("player_ids")

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])
        elements_by_id = {int(el["id"]): el for el in elements}
        events = bs.get("events", [])
        current_event = _current_event_id(events)

        squad_ids: list[int] = []

        if manager_id is not None:
            if current_event is None:
                return [TextContent(type="text", text=json.dumps({"error": "No current event found"}))]
            try:
                picks = await _manager_picks(int(manager_id), current_event)
                squad_ids = [int(p["element"]) for p in picks.get("picks", [])]
            except httpx.HTTPStatusError as e:
                return [TextContent(type="text", text=json.dumps({"error": f"Manager not found: {e.response.status_code}"}))]
        elif player_ids_arg:
            squad_ids = [int(pid) for pid in player_ids_arg]
        else:
            return [TextContent(type="text", text=json.dumps({"error": "Provide manager_id or player_ids"}))]

        validation = _validate_squad(squad_ids, elements_by_id)

        # Add detailed player breakdown
        squad_details: list[dict[str, Any]] = []
        for pid in squad_ids:
            el = elements_by_id.get(pid, {})
            team_id = int(el.get("team", 0))
            squad_details.append({
                "id": pid,
                "name": el.get("web_name", str(pid)),
                "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                "team_id": team_id,
                "position": POS_MAP.get(int(el.get("element_type", 3)), "MID"),
                "price_m": round(_price_m(int(el.get("now_cost", 0))), 1),
            })

        # Add team names to team_counts
        team_counts_named = {
            teams_by_id.get(tid, {}).get("name", str(tid)): count
            for tid, count in validation["team_counts"].items()
        }

        payload = {
            "valid": validation["valid"],
            "violations": validation["violations"],
            "warnings": validation["warnings"],
            "fpl_rules": {
                "squad_size": SQUAD_SIZE,
                "squad_composition": SQUAD_COMPOSITION,
                "max_per_team": MAX_PLAYERS_PER_TEAM,
            },
            "squad_summary": {
                "total_players": len(squad_ids),
                "total_value_m": validation["total_value_m"],
                "position_counts": validation["position_counts"],
                "team_counts": team_counts_named,
            },
            "squad_details": squad_details,
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_differentials":
        max_ownership = float(arguments.get("max_ownership_pct", 10.0))
        min_form = float(arguments.get("min_form", 4.0))
        position = arguments.get("position")
        max_price_m = arguments.get("max_price_m")
        min_minutes = int(arguments.get("min_minutes", 200))
        limit = int(arguments.get("limit", 20))

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])
        events = bs.get("events", [])
        current_event = _current_event_id(events)
        fixtures = await _fixtures()
        team_strength = _calculate_team_strength(bs.get("teams", []))

        pos_filter: str | None = None
        if position:
            pos_filter = str(position).strip().upper()

        differentials = []
        for el in elements:
            ownership = _to_float(el.get("selected_by_percent"))
            if ownership > max_ownership:
                continue

            form = _to_float(el.get("form"))
            if form < min_form:
                continue

            if int(_to_float(el.get("minutes"))) < min_minutes:
                continue

            if str(el.get("status", "a")) != "a":
                continue

            pos = POS_MAP.get(int(el["element_type"]), "MID")
            if pos_filter and pos != pos_filter:
                continue

            price = _price_m(int(el["now_cost"]))
            if max_price_m is not None and price > float(max_price_m):
                continue

            scored = _score_player_first_pass(el, teams_by_id, fixtures, horizon_gws=5, current_event=current_event, team_strength=team_strength, elements=elements)
            scored["ownership_pct"] = ownership
            differentials.append(scored)

        differentials.sort(key=lambda x: x["base_score"], reverse=True)

        payload = {
            "params": {
                "max_ownership_pct": max_ownership,
                "min_form": min_form,
                "position": position,
                "min_minutes": min_minutes,
            },
            "count": len(differentials[:limit]),
            "differentials": differentials[:limit],
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_dgw_bgw":
        bs = await _bootstrap()
        events = bs.get("events", [])
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        current_event = _current_event_id(events)

        event_start = arguments.get("event_start")
        event_end = arguments.get("event_end")

        if event_start is None:
            event_start = current_event or 1
        event_start = int(event_start)

        if event_end is None:
            event_end = event_start + 5
        event_end = int(event_end)

        fixtures = await _fixtures()

        team_fixture_count: dict[int, dict[int, int]] = {}
        for fx in fixtures:
            ev = fx.get("event")
            if ev is None:
                continue
            ev = int(ev)
            if ev < event_start or ev > event_end:
                continue

            for team_key in ("team_h", "team_a"):
                team_id = fx.get(team_key)
                if team_id is None:
                    continue
                team_id = int(team_id)
                if team_id not in team_fixture_count:
                    team_fixture_count[team_id] = {}
                team_fixture_count[team_id][ev] = team_fixture_count[team_id].get(ev, 0) + 1

        dgw_events: dict[int, list[dict[str, Any]]] = {}
        bgw_events: dict[int, list[dict[str, Any]]] = {}

        for team_id, ev_counts in team_fixture_count.items():
            team_name = teams_by_id.get(team_id, {}).get("name", str(team_id))
            for ev, count in ev_counts.items():
                if count >= 2:
                    if ev not in dgw_events:
                        dgw_events[ev] = []
                    dgw_events[ev].append({"team_id": team_id, "team": team_name, "fixtures": count})

        all_team_ids = set(teams_by_id.keys())
        for ev in range(event_start, event_end + 1):
            teams_with_fixtures = {tid for tid, ev_counts in team_fixture_count.items() if ev in ev_counts}
            blanking = all_team_ids - teams_with_fixtures
            if blanking:
                if ev not in bgw_events:
                    bgw_events[ev] = []
                for tid in blanking:
                    bgw_events[ev].append({
                        "team_id": tid,
                        "team": teams_by_id.get(tid, {}).get("name", str(tid)),
                        "fixtures": 0,
                    })

        payload = {
            "event_range": [event_start, event_end],
            "double_gameweeks": {str(k): v for k, v in sorted(dgw_events.items())},
            "blank_gameweeks": {str(k): v for k, v in sorted(bgw_events.items())},
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_captain_picks":
        event_id_arg = arguments.get("event_id")
        limit = int(arguments.get("limit", 10))
        min_minutes = int(arguments.get("min_minutes", 400))

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])
        events = bs.get("events", [])
        current_event = _current_event_id(events)

        target_event = int(event_id_arg) if event_id_arg is not None else current_event
        if target_event is None:
            return [TextContent(type="text", text=json.dumps({"error": "No event found"}))]

        fixtures = await _fixtures()

        event_fixtures: dict[int, list[dict[str, Any]]] = {}
        for fx in fixtures:
            ev = fx.get("event")
            if ev is None or int(ev) != target_event:
                continue
            for team_key, opp_key, diff_key, is_home in [
                ("team_h", "team_a", "team_h_difficulty", True),
                ("team_a", "team_h", "team_a_difficulty", False),
            ]:
                team_id = fx.get(team_key)
                if team_id is None:
                    continue
                team_id = int(team_id)
                if team_id not in event_fixtures:
                    event_fixtures[team_id] = []
                event_fixtures[team_id].append({
                    "opponent": fx.get(opp_key),
                    "difficulty": int(_to_float(fx.get(diff_key))),
                    "is_home": is_home,
                })

        captain_scores = []
        for el in elements:
            if int(_to_float(el.get("minutes"))) < min_minutes:
                continue
            if str(el.get("status", "a")) != "a":
                continue

            team_id = int(el["team"])
            team_fx = event_fixtures.get(team_id, [])
            if not team_fx:
                continue

            ppg = _to_float(el.get("points_per_game"))
            form = _to_float(el.get("form"))
            ict = _to_float(el.get("ict_index"))
            threat = _to_float(el.get("threat"))

            home_bonus = 0.5 if any(f["is_home"] for f in team_fx) else 0.0
            avg_diff = sum(f["difficulty"] for f in team_fx) / len(team_fx)
            fixture_ease = 6.0 - avg_diff
            dgw_multiplier = len(team_fx)

            penalty_bonus = 0.0
            if _to_float(el.get("penalties_order", 99)) <= 2:
                penalty_bonus = 1.5

            captain_score = (
                (form * 2.0)
                + (ppg * 1.5)
                + (threat * 0.01)
                + (ict * 0.05)
                + (fixture_ease * 1.2)
                + home_bonus
                + penalty_bonus
            ) * dgw_multiplier

            name = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip() or el.get("web_name")
            captain_scores.append({
                "id": int(el["id"]),
                "name": name,
                "web_name": el.get("web_name"),
                "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                "position": POS_MAP.get(int(el["element_type"]), "MID"),
                "captain_score": round(captain_score, 3),
                "form": form,
                "points_per_game": ppg,
                "fixtures": team_fx,
                "is_home": any(f["is_home"] for f in team_fx),
                "on_penalties": _to_float(el.get("penalties_order", 99)) <= 2,
            })

        captain_scores.sort(key=lambda x: x["captain_score"], reverse=True)

        payload = {
            "event_id": target_event,
            "captain_picks": captain_scores[:limit],
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_compare":
        player_ids = arguments.get("player_ids", [])
        last_matches = int(arguments.get("last_matches", 5))

        if not player_ids or len(player_ids) < 2:
            return [TextContent(type="text", text=json.dumps({"error": "Provide at least 2 player IDs"}))]
        if len(player_ids) > 6:
            player_ids = player_ids[:6]

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])
        elements_by_id = {int(el["id"]): el for el in elements}
        events = bs.get("events", [])
        current_event = _current_event_id(events)
        fixtures = await _fixtures()
        team_strength = _calculate_team_strength(bs.get("teams", []))

        sem = asyncio.Semaphore(6)

        async def fetch_player(pid: int) -> dict[str, Any] | None:
            el = elements_by_id.get(pid)
            if el is None:
                return {"id": pid, "error": "Player not found"}

            async with sem:
                es = await _element_summary(pid)

            team_id = int(el["team"])
            recent = _recent_form_from_element_summary(es, last_matches=last_matches)
            scored = _score_player_first_pass(el, teams_by_id, fixtures, horizon_gws=5, current_event=current_event, team_strength=team_strength, elements=elements)

            upcoming = es.get("fixtures", [])[:5]
            upcoming_simple = []
            for fx in upcoming:
                opp_id = fx.get("opponent_team")
                upcoming_simple.append({
                    "event": fx.get("event"),
                    "opponent": teams_by_id.get(int(opp_id), {}).get("name") if opp_id else None,
                    "difficulty": fx.get("difficulty"),
                    "is_home": fx.get("is_home"),
                })

            return {
                "id": pid,
                "name": f"{el.get('first_name', '')} {el.get('second_name', '')}".strip() or el.get("web_name"),
                "web_name": el.get("web_name"),
                "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                "position": POS_MAP.get(int(el["element_type"]), "MID"),
                "price_m": round(_price_m(int(el["now_cost"])), 1),
                "ownership_pct": _to_float(el.get("selected_by_percent")),
                "season_stats": {
                    "total_points": int(_to_float(el.get("total_points"))),
                    "points_per_game": _to_float(el.get("points_per_game")),
                    "minutes": int(_to_float(el.get("minutes"))),
                    "goals": int(_to_float(el.get("goals_scored"))),
                    "assists": int(_to_float(el.get("assists"))),
                    "clean_sheets": int(_to_float(el.get("clean_sheets"))),
                    "xGI": _to_float(el.get("expected_goal_involvements")),
                    "ict_index": _to_float(el.get("ict_index")),
                },
                "recent": recent,
                "base_score": scored["base_score"],
                "expected_points": scored.get("expected_points", 0.0),
                "confidence": scored.get("confidence", {}),
                "signals": scored.get("signals", {}),
                "upcoming_fixtures": upcoming_simple,
                "status": el.get("status"),
                "explanation": scored.get("explanation", ""),
            }

        results = await asyncio.gather(*(fetch_player(int(pid)) for pid in player_ids))

        # Generate head-to-head probability comparisons for all pairs
        head_to_head = []
        valid_results = [r for r in results if "error" not in r]
        for i in range(len(valid_results)):
            for j in range(i + 1, len(valid_results)):
                h2h = _calculate_head_to_head_probability(valid_results[i], valid_results[j])
                head_to_head.append(h2h)

        payload = {
            "comparison": results,
            "head_to_head": head_to_head,
            "params": {"player_ids": player_ids, "last_matches": last_matches},
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_price_changes":
        direction = arguments.get("direction")
        limit = int(arguments.get("limit", 20))

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])

        rising = []
        falling = []

        for el in elements:
            transfers_in = int(_to_float(el.get("transfers_in_event")))
            transfers_out = int(_to_float(el.get("transfers_out_event")))
            net = transfers_in - transfers_out

            team_id = int(el["team"])
            player_data = {
                "id": int(el["id"]),
                "name": el.get("web_name"),
                "team": teams_by_id.get(team_id, {}).get("name", str(team_id)),
                "position": POS_MAP.get(int(el["element_type"]), "MID"),
                "price_m": round(_price_m(int(el["now_cost"])), 1),
                "transfers_in": transfers_in,
                "transfers_out": transfers_out,
                "net_transfers": net,
                "ownership_pct": _to_float(el.get("selected_by_percent")),
            }

            if net > 0:
                rising.append(player_data)
            elif net < 0:
                falling.append(player_data)

        rising.sort(key=lambda x: x["net_transfers"], reverse=True)
        falling.sort(key=lambda x: x["net_transfers"])

        payload: dict[str, Any] = {}
        if direction is None or direction == "rising":
            payload["rising"] = rising[:limit]
        if direction is None or direction == "falling":
            payload["falling"] = falling[:limit]

        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_deadline":
        bs = await _bootstrap()
        events = bs.get("events", [])

        next_event = next((e for e in events if e.get("is_next")), None)
        current_event = next((e for e in events if e.get("is_current")), None)

        if next_event:
            deadline = next_event.get("deadline_time")
            payload = {
                "event_id": next_event.get("id"),
                "event_name": next_event.get("name"),
                "deadline_time": deadline,
                "finished": next_event.get("finished"),
                "is_current": False,
                "is_next": True,
            }
        elif current_event:
            deadline = current_event.get("deadline_time")
            payload = {
                "event_id": current_event.get("id"),
                "event_name": current_event.get("name"),
                "deadline_time": deadline,
                "finished": current_event.get("finished"),
                "is_current": True,
                "is_next": False,
            }
        else:
            payload = {"error": "No upcoming deadline found"}

        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_set_piece_takers":
        team_id_filter = arguments.get("team_id")

        bs = await _bootstrap()
        teams = bs.get("teams", [])
        elements = bs.get("elements", [])
        elements_by_id = {int(el["id"]): el for el in elements}

        result = []
        for team in teams:
            tid = int(team["id"])
            if team_id_filter is not None and tid != int(team_id_filter):
                continue

            penalties_order = team.get("penalties_order") or []
            corners_order = team.get("corners_and_indirect_freekicks_order") or []
            fk_order = team.get("direct_freekicks_order") or []

            def resolve_names(id_list: list) -> list[dict[str, Any]]:
                out = []
                for pid in id_list[:3]:
                    el = elements_by_id.get(int(pid), {})
                    out.append({
                        "id": int(pid),
                        "name": el.get("web_name", str(pid)),
                    })
                return out

            result.append({
                "team_id": tid,
                "team": team.get("name"),
                "penalties": resolve_names(penalties_order),
                "corners_indirect_fks": resolve_names(corners_order),
                "direct_fks": resolve_names(fk_order),
            })

        payload = {"teams": result}
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_live_bps":
        event_id_arg = arguments.get("event_id")

        bs = await _bootstrap()
        events = bs.get("events", [])
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements_by_id = {int(el["id"]): el for el in bs.get("elements", [])}
        current_event = _current_event_id(events)

        event_id = int(event_id_arg) if event_id_arg is not None else current_event
        if event_id is None:
            return [TextContent(type="text", text=json.dumps({"error": "No event found"}))]

        fixtures = await _fixtures()
        live_data = await _event_live(event_id)

        event_fixtures = [fx for fx in fixtures if fx.get("event") == event_id]

        live_elements = live_data.get("elements", [])
        bps_by_id = {int(el["id"]): el.get("stats", {}).get("bps", 0) for el in live_elements}

        fixture_bps = []
        for fx in event_fixtures:
            fx_id = fx.get("id")
            team_h = fx.get("team_h")
            team_a = fx.get("team_a")
            team_h_name = teams_by_id.get(int(team_h), {}).get("name") if team_h else None
            team_a_name = teams_by_id.get(int(team_a), {}).get("name") if team_a else None

            bps_stats = fx.get("stats", [])
            bps_data = next((s for s in bps_stats if s.get("identifier") == "bps"), None)

            leaders = []
            if bps_data:
                all_bps = bps_data.get("h", []) + bps_data.get("a", [])
                all_bps.sort(key=lambda x: x.get("value", 0), reverse=True)
                for entry in all_bps[:5]:
                    el_id = entry.get("element")
                    el = elements_by_id.get(int(el_id), {})
                    leaders.append({
                        "id": el_id,
                        "name": el.get("web_name", str(el_id)),
                        "team": teams_by_id.get(int(el.get("team", 0)), {}).get("name"),
                        "bps": entry.get("value"),
                    })

            fixture_bps.append({
                "fixture_id": fx_id,
                "teams": f"{team_h_name} vs {team_a_name}",
                "started": fx.get("started"),
                "finished": fx.get("finished"),
                "minutes": fx.get("minutes"),
                "score": f"{fx.get('team_h_score', '-')} - {fx.get('team_a_score', '-')}",
                "bps_leaders": leaders,
            })

        payload = {
            "event_id": event_id,
            "fixtures": fixture_bps,
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    if name == "fpl_player_enriched":
        player_ids = arguments.get("player_ids", [])
        player_names = arguments.get("player_names", [])

        if not player_ids and not player_names:
            return [TextContent(type="text", text=json.dumps({"error": "Provide player_ids or player_names"}))]

        bs = await _bootstrap()
        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements_by_id = {int(el["id"]): el for el in bs.get("elements", [])}

        # Resolve player_names to IDs if provided
        if player_names and not player_ids:
            player_ids = []
            for name_query in player_names[:5]:  # Limit to 5
                q = name_query.lower().strip()
                for el in bs.get("elements", []):
                    fn = str(el.get("first_name", "")).lower()
                    sn = str(el.get("second_name", "")).lower()
                    wn = str(el.get("web_name", "")).lower()
                    if q in fn or q in sn or q in wn or q == wn:
                        player_ids.append(int(el["id"]))
                        break

        if not player_ids:
            return [TextContent(type="text", text=json.dumps({"error": "No players found"}))]

        # Limit to 5 players to avoid slow responses
        player_ids = player_ids[:5]

        results = []
        for pid in player_ids:
            el = elements_by_id.get(pid)
            if el is None:
                results.append({"id": pid, "error": "Player not found"})
                continue

            team_id = int(el["team"])
            team_name = teams_by_id.get(team_id, {}).get("name", "")
            player_name = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
            web_name = el.get("web_name", "")

            # Get enriched data from external sources
            try:
                enriched = await enrich_player_async(player_name, team_name)
            except Exception as e:
                logger.warning(f"Enrichment failed for {player_name}: {e}")
                enriched = {"enrichment_available": False, "error": str(e)}

            # Combine FPL data with enriched data
            result = {
                "id": pid,
                "name": player_name,
                "web_name": web_name,
                "team": team_name,
                "position": POS_MAP.get(int(el["element_type"]), "MID"),
                "price_m": round(_price_m(int(el["now_cost"])), 1),
                # FPL stats
                "fpl_stats": {
                    "total_points": int(_to_float(el.get("total_points"))),
                    "points_per_game": _to_float(el.get("points_per_game")),
                    "minutes": int(_to_float(el.get("minutes"))),
                    "goals": int(_to_float(el.get("goals_scored"))),
                    "assists": int(_to_float(el.get("assists"))),
                    "xG": _to_float(el.get("expected_goals")),
                    "xA": _to_float(el.get("expected_assists")),
                    "xGI": _to_float(el.get("expected_goal_involvements")),
                    "form": _to_float(el.get("form")),
                    "ict_index": _to_float(el.get("ict_index")),
                    "threat": _to_float(el.get("threat")),
                    "creativity": _to_float(el.get("creativity")),
                    "influence": _to_float(el.get("influence")),
                },
                # External enriched data
                "enriched": enriched,
            }
            results.append(result)

        payload = {
            "players": results,
            "sources_note": "External data from Understat (xG per shot, conversion rate) and FBref (SOT, progressive passes). Data is cached for 6 hours.",
        }
        return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]

    raise ValueError(f"Unknown tool: {name}")


# --------------------
# REST API: Team Report Endpoint
# --------------------


async def team_report(request: Request) -> Response:
    """GET /api/team-report/{team_id} — comprehensive team analysis JSON."""
    raw_id = request.path_params.get("team_id", "")
    try:
        team_id = int(raw_id)
    except (ValueError, TypeError):
        return JSONResponse({"error": "Invalid team_id — must be an integer"}, status_code=400)

    if team_id <= 0:
        return JSONResponse({"error": "team_id must be positive"}, status_code=400)

    # Check report cache
    cache_key = f"report:{team_id}"
    cached = _report_cache.get(cache_key)
    if cached:
        expires_at, data = cached
        if time.time() < expires_at:
            return JSONResponse(data)
        _report_cache.pop(cache_key, None)

    # ---------- Fetch shared data ----------
    try:
        bs = await _bootstrap()
    except Exception as e:
        logger.error("Bootstrap fetch failed: %s", e)
        return JSONResponse({"error": "Failed to fetch FPL data"}, status_code=502)

    teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
    elements = bs.get("elements", [])
    elements_by_id = {int(el["id"]): el for el in elements}
    events = bs.get("events", [])
    current_event = _current_event_id(events)

    if current_event is None:
        return JSONResponse({"error": "No current gameweek found"}, status_code=503)

    next_event: int | None = None
    for ev in events:
        if ev.get("is_next"):
            next_event = int(ev["id"])
            break
    if next_event is None:
        next_event = current_event + 1

    # Fetch manager data
    try:
        manager_info, manager_hist, picks, transfers = await asyncio.gather(
            _manager_info(team_id),
            _manager_history(team_id),
            _manager_picks(team_id, current_event),
            _manager_transfers(team_id),
        )
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code == 404:
            return JSONResponse({"error": f"Manager {team_id} not found"}, status_code=404)
        return JSONResponse({"error": f"FPL API error: {code}"}, status_code=502)
    except Exception as e:
        logger.error("Manager data fetch failed: %s", e)
        return JSONResponse({"error": "Failed to fetch manager data"}, status_code=502)

    fixtures = await _fixtures()
    team_strength = _calculate_team_strength(bs.get("teams", []))
    fixture_horizon = 5
    team_outlook = _team_fixture_outlook(fixtures, teams_by_id, current_event, horizon_gws=fixture_horizon)

    # ====== META ======
    meta = {
        "currentGW": current_event,
        "nextGW": next_event,
        "analysisDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ====== OVERVIEW ======
    chips_history = manager_hist.get("chips", [])
    chips_used = [{"name": c.get("name"), "event": c.get("event")} for c in chips_history]
    all_chips = {"wildcard", "freehit", "bboost", "3xc"}
    used_chip_names = {c.get("name") for c in chips_history}
    wildcard_count = sum(1 for c in chips_history if c.get("name") == "wildcard")
    chips_available: list[str] = []
    for chip in sorted(all_chips):
        if chip == "wildcard":
            if wildcard_count < 2:
                chips_available.append(chip)
        elif chip not in used_chip_names:
            chips_available.append(chip)

    bank = round(_price_m(int(manager_info.get("last_deadline_bank", 0))), 1)
    overview = {
        "manager": f"{manager_info.get('player_first_name', '')} {manager_info.get('player_last_name', '')}".strip(),
        "teamName": manager_info.get("name", ""),
        "teamId": team_id,
        "overallRank": manager_info.get("summary_overall_rank"),
        "overallPoints": manager_info.get("summary_overall_points"),
        "bank": bank,
        "teamValue": round(_price_m(int(manager_info.get("last_deadline_value", 0))), 1),
        "chipsUsed": chips_used,
        "chipsAvailable": chips_available,
    }

    # ====== SQUAD (Starting XI + Bench) ======
    squad_ids = [int(p["element"]) for p in picks.get("picks", [])]
    starting_xi: list[dict[str, Any]] = []
    bench_list: list[dict[str, Any]] = []
    all_squad_data: list[dict[str, Any]] = []  # enriched player rows for reuse

    for idx, pick in enumerate(picks.get("picks", [])):
        el_id = int(pick["element"])
        el = elements_by_id.get(el_id, {})
        tid = int(el.get("team", 0))
        pos = POS_MAP.get(int(el.get("element_type", 3)), "MID")

        minutes = int(_to_float(el.get("minutes")))
        games_played = max(1, minutes / 90.0)
        avg_minutes = minutes / games_played if games_played > 1 else 90.0

        xpts_data = _calculate_multi_gw_xpts(
            el, fixtures, teams_by_id, current_event, fixture_horizon, avg_minutes, team_strength,
        )

        playing_prob = _playing_probability(el, avg_minutes)
        if playing_prob < 0.5:
            rotation_risk = "high"
        elif playing_prob < 0.75:
            rotation_risk = "medium"
        else:
            rotation_risk = "low"

        # Next 3 fixtures for this player's team
        team_fxs = team_outlook.get(tid, {}).get("next_opponents", [])[:3]
        fixtures_next3 = [
            {
                "opponent": teams_by_id.get(fx.get("opponent_team"), {}).get("short_name", ""),
                "difficulty": fx.get("difficulty"),
                "is_home": fx.get("is_home"),
            }
            for fx in team_fxs
        ]

        xg = round(_to_float(el.get("expected_goals")), 2)
        xa = round(_to_float(el.get("expected_assists")), 2)
        xgi = round(_to_float(el.get("expected_goal_involvements")), 2)

        player_row: dict[str, Any] = {
            "name": el.get("web_name", str(el_id)),
            "position": pos,
            "team": teams_by_id.get(tid, {}).get("short_name", str(tid)),
            "cost": round(_price_m(int(el.get("now_cost", 0))), 1),
            "form": _to_float(el.get("form")),
            "ppg": _to_float(el.get("points_per_game")),
            "xG": xg,
            "xA": xa,
            "xGI": xgi,
            "xP_next5": xpts_data["total_expected_points"],
            "rotation_risk": rotation_risk,
            "status": el.get("status", "a"),
            "news": el.get("news", "") or "",
            "fixtures_next3": fixtures_next3,
        }

        is_bench = idx >= 11  # picks 12-15 are bench

        if not is_bench:
            player_row["is_captain"] = pick.get("is_captain", False)
            player_row["is_vice_captain"] = pick.get("is_vice_captain", False)
            starting_xi.append(player_row)
        else:
            bench_list.append(player_row)

        # Keep enriched copy for downstream sections
        all_squad_data.append({
            **player_row,
            "id": el_id,
            "_el": el,
            "team_id": tid,
            "is_bench": is_bench,
            "playing_probability": playing_prob,
            "avg_minutes": avg_minutes,
        })

    squad_section = {"startingXI": starting_xi, "bench": bench_list}

    # ====== NEXT-GW FIXTURES ======
    next_gw_fixtures: list[dict[str, Any]] = []
    for fx in fixtures:
        ev = fx.get("event")
        if ev is None or int(ev) != next_event:
            continue
        h_id = fx.get("team_h")
        a_id = fx.get("team_a")
        if h_id is None or a_id is None:
            continue
        next_gw_fixtures.append({
            "home": teams_by_id.get(int(h_id), {}).get("short_name", str(h_id)),
            "away": teams_by_id.get(int(a_id), {}).get("short_name", str(a_id)),
            "home_diff": int(_to_float(fx.get("team_h_difficulty", 0))),
            "away_diff": int(_to_float(fx.get("team_a_difficulty", 0))),
        })

    # ====== FLAGGED PLAYERS ======
    flagged_players: list[dict[str, Any]] = []
    for pd in all_squad_data:
        issues: list[str] = []
        severity = "low"
        el = pd["_el"]
        status = str(el.get("status", "a"))
        form_val = pd["form"]
        p_prob = pd["playing_probability"]

        # Injury / suspension / unavailable
        if status in ("i", "s", "u"):
            issues.append("injured/suspended/unavailable")
            severity = "critical"
        elif status == "d":
            issues.append("doubtful")
            severity = "moderate"

        chance = el.get("chance_of_playing_next_round")
        if chance is not None and _to_float(chance, 100) < 75:
            pct = _to_float(chance, 100)
            issues.append(f"chance of playing {int(pct)}%")
            if pct <= 25:
                severity = "critical"
            elif severity != "critical":
                severity = "moderate"

        if 0 < form_val < 3.5:
            issues.append(f"low form ({form_val})")
            if severity == "low":
                severity = "moderate" if form_val < 2.0 else "low"

        if pd["rotation_risk"] == "high":
            issues.append("high rotation risk")
            if severity == "low":
                severity = "moderate"

        fxs = pd["fixtures_next3"]
        if fxs:
            avg_fdr = sum(f.get("difficulty", 3) for f in fxs) / len(fxs)
            if avg_fdr > 3.5:
                issues.append(f"tough fixtures (avg FDR {avg_fdr:.1f})")

        avg_m = pd["avg_minutes"]
        if avg_m < 60 and int(_to_float(el.get("minutes"))) > 0:
            issues.append(f"low avg minutes ({avg_m:.0f})")
            if severity == "low":
                severity = "moderate"

        if issues:
            flagged_players.append({
                "name": pd["name"],
                "position": pd["position"],
                "team": pd["team"],
                "cost": pd["cost"],
                "form": form_val,
                "severity": severity,
                "issues": issues,
                "bench": pd["is_bench"],
            })

    severity_order = {"critical": 0, "moderate": 1, "low": 2}
    flagged_players.sort(key=lambda x: severity_order.get(x["severity"], 3))

    # ====== TRANSFER RECOMMENDATIONS ======
    current_squad_ids = set(squad_ids)

    # Score every squad player once via first-pass model
    squad_scored: list[dict[str, Any]] = []
    for pd in all_squad_data:
        scored = _score_player_first_pass(
            pd["_el"], teams_by_id, fixtures, fixture_horizon,
            current_event, team_strength=team_strength, elements=elements,
        )
        scored["_pd"] = pd
        squad_scored.append(scored)
    squad_scored.sort(key=lambda x: x["expected_points"])  # weakest first

    transfer_recs: list[dict[str, Any]] = []
    for out_scored in squad_scored[:8]:  # check weakest 8
        out_el = elements_by_id.get(out_scored["id"], {})
        pos = out_scored["position"]
        sell_price = out_scored["price_m"]
        out_pd = out_scored["_pd"]

        # Issues carried from flagged
        out_issues = next(
            (fp["issues"] for fp in flagged_players if fp["name"] == out_pd["name"]),
            [],
        )
        out_fxs = out_pd["fixtures_next3"]
        out_avg_fdr = round(sum(f.get("difficulty", 3) for f in out_fxs) / len(out_fxs), 1) if out_fxs else 0.0

        # Find valid replacements
        candidates: list[dict[str, Any]] = []
        for el in elements:
            in_id = int(el["id"])
            if in_id in current_squad_ids:
                continue
            if POS_MAP.get(int(el.get("element_type", 3)), "MID") != pos:
                continue
            if str(el.get("status", "a")) != "a":
                continue

            tc = _can_transfer_in(
                player_in=el, player_out=out_el,
                current_squad_ids=current_squad_ids,
                elements_by_id=elements_by_id,
                bank=bank, selling_price=sell_price,
            )
            if not tc["valid"]:
                continue

            scored_in = _score_player_first_pass(
                el, teams_by_id, fixtures, fixture_horizon,
                current_event, team_strength=team_strength, elements=elements,
            )
            gain = scored_in["expected_points"] - out_scored["expected_points"]
            if gain <= 0:
                continue

            in_tid = int(el.get("team", 0))
            in_fxs_raw = team_outlook.get(in_tid, {}).get("next_opponents", [])[:3]
            in_fxs = [
                {
                    "opponent": teams_by_id.get(f.get("opponent_team"), {}).get("short_name", ""),
                    "difficulty": f.get("difficulty"),
                    "is_home": f.get("is_home"),
                }
                for f in in_fxs_raw
            ]
            in_avg_fdr = round(sum(f.get("difficulty", 3) for f in in_fxs) / len(in_fxs), 1) if in_fxs else 0.0

            candidates.append({
                "name": scored_in["name"],
                "team": scored_in["team"],
                "position": pos,
                "cost": scored_in["price_m"],
                "form": scored_in["signals"]["form"],
                "xP_next5": scored_in["expected_points"],
                "transfer_score": round(gain, 2),
                "fixtures_next3": in_fxs,
                "avg_fdr": in_avg_fdr,
            })

        candidates.sort(key=lambda x: x["transfer_score"], reverse=True)
        if not candidates:
            continue

        transfer_recs.append({
            "out": {
                "name": out_pd["name"],
                "team": out_pd["team"],
                "position": pos,
                "cost": out_pd["cost"],
                "form": out_pd["form"],
                "xP_next5": out_scored["expected_points"],
                "issues": out_issues,
                "avg_fdr": out_avg_fdr,
                "fixtures_next3": out_pd["fixtures_next3"],
            },
            "in": candidates[0],
            "alternatives": candidates[1:4],
            "cost_change": round(candidates[0]["cost"] - out_pd["cost"], 1),
            "is_free": False,  # set after sort
        })

    transfer_recs.sort(key=lambda x: x["in"]["transfer_score"], reverse=True)
    transfer_recs = transfer_recs[:5]
    for i, rec in enumerate(transfer_recs):
        rec["is_free"] = i == 0  # first suggestion is the free transfer

    # ====== CAPTAINCY PICKS (squad players only) ======
    event_fixtures_map: dict[int, list[dict[str, Any]]] = {}
    for fx in fixtures:
        ev = fx.get("event")
        if ev is None or int(ev) != next_event:
            continue
        for team_key, opp_key, diff_key, is_home in [
            ("team_h", "team_a", "team_h_difficulty", True),
            ("team_a", "team_h", "team_a_difficulty", False),
        ]:
            tid = fx.get(team_key)
            if tid is None:
                continue
            tid = int(tid)
            event_fixtures_map.setdefault(tid, []).append({
                "opponent": fx.get(opp_key),
                "difficulty": int(_to_float(fx.get(diff_key))),
                "is_home": is_home,
            })

    captaincy_picks: list[dict[str, Any]] = []
    for pd in all_squad_data:
        if pd["is_bench"]:
            continue
        el = pd["_el"]
        tid = pd["team_id"]
        team_fx = event_fixtures_map.get(tid, [])
        if not team_fx:
            continue

        ppg = _to_float(el.get("points_per_game"))
        form_val = _to_float(el.get("form"))
        ict = _to_float(el.get("ict_index"))
        threat = _to_float(el.get("threat"))

        home_bonus = 0.5 if any(f["is_home"] for f in team_fx) else 0.0
        avg_diff = sum(f["difficulty"] for f in team_fx) / len(team_fx)
        fixture_ease = 6.0 - avg_diff
        dgw_mult = len(team_fx)
        penalty_bonus = 1.5 if _to_float(el.get("penalties_order", 99)) <= 2 else 0.0

        cap_score = (
            (form_val * 2.0) + (ppg * 1.5) + (threat * 0.01) + (ict * 0.05)
            + (fixture_ease * 1.2) + home_bonus + penalty_bonus
        ) * dgw_mult

        nfx = team_fx[0]
        opp_id = nfx.get("opponent")
        opp_short = teams_by_id.get(int(opp_id), {}).get("short_name", "") if opp_id else ""

        captaincy_picks.append({
            "name": pd["name"],
            "team": pd["team"],
            "form": form_val,
            "xP_next5": pd["xP_next5"],
            "captaincy_score": round(cap_score, 2),
            "next_fixture": f"{'vs' if nfx.get('is_home') else '@'} {opp_short}",
            "next_fdr": nfx.get("difficulty", 0),
        })

    captaincy_picks.sort(key=lambda x: x["captaincy_score"], reverse=True)
    captaincy_picks = captaincy_picks[:5]

    # ====== BGW / DGW OUTLOOK ======
    ev_start = current_event
    ev_end = current_event + 5
    team_fixture_count: dict[int, dict[int, int]] = {}
    for fx in fixtures:
        ev = fx.get("event")
        if ev is None:
            continue
        ev = int(ev)
        if ev < ev_start or ev > ev_end:
            continue
        for tk in ("team_h", "team_a"):
            tid = fx.get(tk)
            if tid is None:
                continue
            tid = int(tid)
            team_fixture_count.setdefault(tid, {})
            team_fixture_count[tid][ev] = team_fixture_count[tid].get(ev, 0) + 1

    all_team_ids = set(teams_by_id.keys())
    bgw_dgw_outlook: list[dict[str, Any]] = []
    for ev in range(ev_start, ev_end + 1):
        teams_with = {tid for tid, ec in team_fixture_count.items() if ev in ec}
        blanking = sorted(
            teams_by_id.get(tid, {}).get("short_name", str(tid))
            for tid in (all_team_ids - teams_with)
        )
        doubling = sorted(
            teams_by_id.get(tid, {}).get("short_name", str(tid))
            for tid, ec in team_fixture_count.items()
            if ev in ec and ec[ev] >= 2
        )
        if blanking or doubling:
            bgw_dgw_outlook.append({"gw": ev, "blanking": blanking, "doubling": doubling})

    # ====== BGW EXPOSURE (squad players affected) ======
    bgw_exposure: list[dict[str, Any]] = []
    for ev in range(ev_start, ev_end + 1):
        teams_with = {tid for tid, ec in team_fixture_count.items() if ev in ec}
        blanking_tids = all_team_ids - teams_with
        if not blanking_tids:
            continue
        affected = [pd["name"] for pd in all_squad_data if pd["team_id"] in blanking_tids]
        if affected:
            bgw_exposure.append({
                "gw": ev,
                "blanking_teams": sorted(
                    teams_by_id.get(tid, {}).get("short_name", str(tid)) for tid in blanking_tids
                ),
                "affected_players": affected,
                "total_affected": len(affected),
            })

    # ====== BEST FIXTURE TEAMS ======
    best_fixture_teams: list[dict[str, Any]] = []
    for tid, ol in team_outlook.items():
        fxs = ol.get("next_opponents", [])[:5]
        best_fixture_teams.append({
            "team": teams_by_id.get(tid, {}).get("short_name", str(tid)),
            "avgFDR": round(ol.get("avg_difficulty", 5.0), 2),
            "fixtures": [
                {
                    "opponent": teams_by_id.get(f.get("opponent_team"), {}).get("short_name", ""),
                    "difficulty": f.get("difficulty"),
                    "is_home": f.get("is_home"),
                    "event": f.get("event"),
                }
                for f in fxs
            ],
        })
    best_fixture_teams.sort(key=lambda x: x["avgFDR"])
    best_fixture_teams = best_fixture_teams[:10]

    # ====== GW HISTORY (last 5) ======
    gw_history: list[dict[str, Any]] = []
    for gw in (manager_hist.get("current", []) or [])[-5:]:
        gw_history.append({
            "event": gw.get("event"),
            "points": gw.get("points"),
            "rank": gw.get("overall_rank"),
            "bench_points": gw.get("points_on_bench"),
            "transfers": gw.get("event_transfers"),
            "transfers_cost": gw.get("event_transfers_cost"),
        })

    # ====== RECENT TRANSFERS (last 5) ======
    raw_transfers = transfers[-5:] if isinstance(transfers, list) else []
    recent_transfers: list[dict[str, Any]] = []
    for t in reversed(raw_transfers):
        in_el = elements_by_id.get(int(t.get("element_in", 0)), {})
        out_el_t = elements_by_id.get(int(t.get("element_out", 0)), {})
        recent_transfers.append({
            "event": t.get("event"),
            "player_in": in_el.get("web_name", str(t.get("element_in"))),
            "player_out": out_el_t.get("web_name", str(t.get("element_out"))),
            "cost_in": round(_price_m(int(t.get("element_in_cost", 0))), 1),
            "cost_out": round(_price_m(int(t.get("element_out_cost", 0))), 1),
        })

    # ====== BUDGET SUMMARY ======
    budget_summary = {
        "current_bank": bank,
        "team_value": overview["teamValue"],
        "bank_after_transfers": round(bank, 1),
    }

    # ---------- Assemble response ----------
    response: dict[str, Any] = {
        "meta": meta,
        "overview": overview,
        "squad": squad_section,
        "nextGWFixtures": next_gw_fixtures,
        "flaggedPlayers": flagged_players,
        "transferRecommendations": transfer_recs,
        "captaincyPicks": captaincy_picks,
        "bgwDgwOutlook": bgw_dgw_outlook,
        "bgwExposure": bgw_exposure,
        "bestFixtureTeams": best_fixture_teams,
        "gwHistory": gw_history,
        "recentTransfers": recent_transfers,
        "budgetSummary": budget_summary,
    }

    # Cache for 60 minutes
    _report_cache[cache_key] = (time.time() + REPORT_CACHE_TTL, response)
    return JSONResponse(response)


# --------------------
# SSE Transport + Starlette wiring
# --------------------
sse = SseServerTransport("/messages/")


async def handle_sse(request: Request) -> Response:
    auth_resp = _require_bearer(request)
    if auth_resp:
        return auth_resp

    async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
    return Response(status_code=204)


async def health(_: Request) -> Response:
    return JSONResponse({"status": "ok"})


starlette_app = Starlette(
    debug=os.getenv("DEBUG", "0") == "1",
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/api/team-report/{team_id}", team_report, methods=["GET"]),
        Route("/sse", handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(starlette_app, host=host, port=port, log_level=os.getenv("UVICORN_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
