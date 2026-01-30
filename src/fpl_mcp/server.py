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


def _availability_penalty(el: dict[str, Any]) -> float:
    status = str(el.get("status", "a"))
    chance = el.get("chance_of_playing_next_round", None)
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
    Estimate probability of playing 60+ minutes based on status, news, and recent minutes.
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
) -> dict[str, Any]:
    """
    Calculate expected points for a single gameweek using xG/xA and fixture context.

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

    xg_adj = xg_per_game * difficulty_factor * home_boost
    xa_adj = xa_per_game * difficulty_factor * home_boost

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

    total_xpts = (xpts_goals + xpts_assists + xpts_cs + xpts_minutes + xpts_bonus + xpts_saves) * playing_prob

    return {
        "expected_points": round(total_xpts, 2),
        "breakdown": {
            "xpts_goals": round(xpts_goals * playing_prob, 3),
            "xpts_assists": round(xpts_assists * playing_prob, 3),
            "xpts_clean_sheet": round(xpts_cs * playing_prob, 3),
            "xpts_minutes": round(xpts_minutes, 3),
            "xpts_bonus": round(xpts_bonus * playing_prob, 3),
            "xpts_saves": round(xpts_saves * playing_prob, 3),
        },
        "adjustments": {
            "playing_probability": round(playing_prob, 3),
            "fixture_difficulty": fixture_difficulty,
            "difficulty_factor": round(difficulty_factor, 3),
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
) -> dict[str, Any]:
    """
    Calculate expected points over multiple gameweeks.
    """
    team_id = int(el["team"])
    playing_prob = _playing_probability(el, avg_minutes)

    gw_xpts: list[dict[str, Any]] = []
    total_xpts = 0.0

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

            gw_calc = _calculate_expected_points(el, difficulty, is_home, playing_prob, avg_minutes)
            gw_xpts.append({
                "event": ev,
                "opponent": teams_by_id.get(int(opp_id), {}).get("name") if opp_id else None,
                "is_home": is_home,
                "difficulty": difficulty,
                "expected_points": gw_calc["expected_points"],
            })
            total_xpts += gw_calc["expected_points"]

    return {
        "total_expected_points": round(total_xpts, 2),
        "gameweeks": gw_xpts,
        "avg_xpts_per_gw": round(total_xpts / len(gw_xpts), 2) if gw_xpts else 0.0,
        "playing_probability": round(playing_prob, 3),
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
) -> dict[str, Any]:
    """
    Score player using expected points model.

    Uses xG/xA-based projections adjusted for fixture difficulty.
    """
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

    # Get multi-gameweek expected points
    xpts_data = _calculate_multi_gw_xpts(
        el, fixtures, teams_by_id, current_event, horizon_gws, avg_minutes
    )

    total_xpts = xpts_data["total_expected_points"]
    playing_prob = xpts_data["playing_probability"]

    # Value metric: expected points per million spent over horizon
    value = (total_xpts / price) if price else 0.0

    # Availability penalty for flagged players
    penalty = _availability_penalty(el)

    # Final score is expected points with minor adjustments
    # We prioritize xPts but give small boosts for:
    # - Recent form (shows current performance level)
    # - Value (points per million is important for budget)
    score = total_xpts + (form * 0.3) + (value * 0.5) - penalty

    team_name = teams_by_id.get(team_id, {}).get("name", str(team_id))
    name = f"{el.get('first_name','')} {el.get('second_name','')}".strip() or str(el.get("web_name"))

    # xG and xA stats
    xg = _to_float(el.get("expected_goals"))
    xa = _to_float(el.get("expected_assists"))
    xgi = _to_float(el.get("expected_goal_involvements"))

    return {
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
        },
        "fixture_xpts": xpts_data["gameweeks"],
    }


def _recent_form_from_element_summary(es: dict[str, Any], last_matches: int) -> dict[str, Any]:
    """
    Extract recent trend signals from element-summary history.

    Works even if some expected fields are missing by falling back safely.
    """
    history: list[dict[str, Any]] = es.get("history", []) or []
    if not history:
        return {
            "matches_used": 0,
            "avg_minutes": 0.0,
            "points_per_90": 0.0,
            "xgi_per_90": 0.0,
            "blank_rate": None,
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

    return {
        "matches_used": matches_used,
        "avg_minutes": round(avg_minutes, 2),
        "points_per_90": round(points_per_90, 3),
        "xgi_per_90": round(xgi_per_90, 3),
        "blank_rate": None if blank_rate is None else round(blank_rate, 3),
    }


def _refine_score(base: dict[str, Any], recent: dict[str, Any]) -> dict[str, Any]:
    """
    Second-pass adjustment:
    - Reward: recent xGI/90, recent points/90, strong minutes trend
    - Penalise: low average minutes (rotation risk)
    """
    base_score = float(base.get("base_score", 0.0))
    avg_minutes = float(recent.get("avg_minutes", 0.0))
    p90 = float(recent.get("points_per_90", 0.0))
    xgi90 = float(recent.get("xgi_per_90", 0.0))

    minutes_factor = min(avg_minutes / 90.0, 1.0)
    rotation_pen = 0.0
    if avg_minutes > 0 and avg_minutes < 60:
        rotation_pen = 1.5
    elif avg_minutes >= 60 and avg_minutes < 75:
        rotation_pen = 0.6

    refined = (
        base_score
        + (xgi90 * 2.2)
        + (p90 * 0.9)
        + (minutes_factor * 1.2)
        - rotation_pen
    )

    out = dict(base)
    out["refined_score"] = round(refined, 3)
    out["recent_signals"] = recent
    out["adjustments"] = {
        "rotation_penalty": rotation_pen,
        "minutes_factor": round(minutes_factor, 3),
        "xgi90_weight": 2.2,
        "p90_weight": 0.9,
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
        description="Two-pass rank: quick shortlist (bootstrap+fixtures) then refine with element-summary trends (minutes/points/90/xGI/90).",
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

        bs = await _bootstrap()
        fx = await _fixtures()

        events = bs.get("events", [])
        current_event = _current_event_id(events)

        teams_by_id = {int(t["id"]): t for t in bs.get("teams", [])}
        elements = bs.get("elements", [])

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
                _score_player_first_pass(el, teams_by_id, fx, horizon_gws=horizon_gws, current_event=current_event)
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
            return _refine_score(item, recent)

        enriched = await asyncio.gather(*(enrich_one(it) for it in pool))

        # Rank by refined score, then trim to limit
        enriched.sort(key=lambda r: r.get("refined_score", r.get("base_score", 0.0)), reverse=True)
        results = enriched[: max(1, limit)]

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
        for pid in current_squad_ids:
            el = elements_by_id.get(pid, {})
            scored = _score_player_first_pass(el, teams_by_id, fixtures, horizon_gws, current_event)
            squad_scored.append(scored)

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
                scored_in = _score_player_first_pass(el, teams_by_id, fixtures, horizon_gws, current_event)

                # Calculate improvement
                xpts_gain = scored_in["expected_points"] - player_out["expected_points"]

                # Only suggest if it's actually an improvement
                if xpts_gain <= 0:
                    continue

                seen_transfers.add((out_id, in_id))
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
                })

        # Sort by expected points gain
        suggestions.sort(key=lambda x: x["xpts_gain"], reverse=True)

        payload = {
            "manager_id": manager_id,
            "current_event": current_event,
            "bank": bank,
            "squad_validation": squad_validation,
            "fpl_rules": {
                "max_per_team": MAX_PLAYERS_PER_TEAM,
                "squad_composition": SQUAD_COMPOSITION,
            },
            "suggestions": suggestions[:max_transfers],
            "note": "All suggestions respect FPL rules: same position, max 3 per team, and budget constraints.",
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

            scored = _score_player_first_pass(el, teams_by_id, fixtures, horizon_gws=5, current_event=current_event)
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
        elements_by_id = {int(el["id"]): el for el in bs.get("elements", [])}
        events = bs.get("events", [])
        current_event = _current_event_id(events)
        fixtures = await _fixtures()

        sem = asyncio.Semaphore(6)

        async def fetch_player(pid: int) -> dict[str, Any] | None:
            el = elements_by_id.get(pid)
            if el is None:
                return {"id": pid, "error": "Player not found"}

            async with sem:
                es = await _element_summary(pid)

            team_id = int(el["team"])
            recent = _recent_form_from_element_summary(es, last_matches=last_matches)
            scored = _score_player_first_pass(el, teams_by_id, fixtures, horizon_gws=5, current_event=current_event)

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
                "upcoming_fixtures": upcoming_simple,
                "status": el.get("status"),
            }

        results = await asyncio.gather(*(fetch_player(int(pid)) for pid in player_ids))

        payload = {
            "comparison": results,
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

    raise ValueError(f"Unknown tool: {name}")


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
