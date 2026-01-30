"""FPL Fantasy God MCP Server - 35+ tools for ultimate FPL domination."""

import json
import logging
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse
from dataclasses import asdict

from config import get_settings
from database import init_db, close_db, fetch_all, fetch_one
from cache import init_cache, close_cache, populate_cache_from_db, cache_get_all_players, cache_get_player

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = Server("fpl-fantasy-god")
sse = SseServerTransport("/messages/")

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# === MCP Tools Definition (35 tools) ===

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ===== EXISTING TOOLS (9) =====
        Tool(
            name="get_player",
            description="Get detailed player info by ID or name",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "name": {"type": "string"}
                }
            }
        ),
        Tool(
            name="search_players",
            description="Search players by team, position, price, form",
            inputSchema={
                "type": "object",
                "properties": {
                    "team": {"type": "string"},
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "max_price": {"type": "number"},
                    "min_form": {"type": "number"},
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="get_fixtures",
            description="Get fixtures for a team or gameweek",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "integer"},
                    "gameweek": {"type": "integer"},
                    "num_fixtures": {"type": "integer", "default": 5}
                }
            }
        ),
        Tool(
            name="get_team_form",
            description="Get team's recent form (W/D/L) and strength metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "integer"},
                    "team_name": {"type": "string"}
                }
            }
        ),
        Tool(
            name="compare_players",
            description="Compare multiple players side-by-side",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_ids": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["player_ids"]
            }
        ),
        Tool(
            name="get_top_players",
            description="Get top players by form, points, or value",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "sort_by": {"type": "string", "enum": ["form", "points", "value"]},
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="get_differentials",
            description="Find low-ownership players with good form (differential picks)",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_ownership": {"type": "number", "default": 10},
                    "min_form": {"type": "number", "default": 4},
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="get_player_trend",
            description="Get player's stats trend over recent gameweeks",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "num_gameweeks": {"type": "integer", "default": 5}
                }
            }
        ),
        Tool(
            name="get_rising_players",
            description="Find players with improving form over recent gameweeks",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "min_improvement": {"type": "number", "default": 1.0},
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),

        # ===== NEW ANALYTICS TOOLS (12) =====
        Tool(
            name="get_expected_points",
            description="Get expected points (xP) projections for a player over next 5 gameweeks",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "num_fixtures": {"type": "integer", "default": 5}
                }
            }
        ),
        Tool(
            name="get_fixture_difficulty",
            description="Get fixture difficulty rating for a team's upcoming matches",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {"type": "integer"},
                    "team_name": {"type": "string"},
                    "num_fixtures": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="get_easiest_fixtures",
            description="Get teams with easiest upcoming fixtures - target these players",
            inputSchema={
                "type": "object",
                "properties": {
                    "num_gameweeks": {"type": "integer", "default": 5},
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="get_price_predictions",
            description="Get players likely to rise or fall in price",
            inputSchema={
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["risers", "fallers", "both"], "default": "both"},
                    "limit": {"type": "integer", "default": 15}
                }
            }
        ),
        Tool(
            name="analyze_luck",
            description="Analyze if a player is over/underperforming their xG/xA (regression candidates)",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "name": {"type": "string"}
                }
            }
        ),
        Tool(
            name="get_overperformers",
            description="Get players scoring above their xG/xA - regression risk candidates",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "limit": {"type": "integer", "default": 15}
                }
            }
        ),
        Tool(
            name="get_underperformers",
            description="Get players scoring below their xG/xA - upside candidates",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "limit": {"type": "integer", "default": 15}
                }
            }
        ),
        Tool(
            name="get_template_players",
            description="Get highly-owned template players (>20% ownership)",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "min_ownership": {"type": "number", "default": 20},
                    "limit": {"type": "integer", "default": 20}
                }
            }
        ),
        Tool(
            name="predict_minutes",
            description="Predict minutes and rotation risk for a player",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "name": {"type": "string"}
                }
            }
        ),
        Tool(
            name="get_dgw_bgw_outlook",
            description="Get double gameweek (DGW) and blank gameweek (BGW) outlook",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_bogey_teams",
            description="Get a player's historical performance vs specific opponents",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer"},
                    "opponent_team": {"type": "string"}
                }
            }
        ),
        Tool(
            name="get_value_picks",
            description="Get best value picks (highest xP per million)",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "max_price": {"type": "number"},
                    "limit": {"type": "integer", "default": 15}
                }
            }
        ),

        # ===== OPTIMIZATION TOOLS (8) =====
        Tool(
            name="build_dream_team",
            description="Build optimal 15-player squad using linear programming",
            inputSchema={
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "default": 100.0},
                    "strategy": {"type": "string", "enum": ["balanced", "attacking", "defensive"], "default": "balanced"},
                    "exclude_players": {"type": "array", "items": {"type": "integer"}},
                    "must_include": {"type": "array", "items": {"type": "integer"}}
                }
            }
        ),
        Tool(
            name="build_free_hit_team",
            description="Build optimal team for a single gameweek (Free Hit chip)",
            inputSchema={
                "type": "object",
                "properties": {
                    "gameweek": {"type": "integer"},
                    "budget": {"type": "number", "default": 100.0}
                },
                "required": ["gameweek"]
            }
        ),
        Tool(
            name="optimize_starting_11",
            description="Pick optimal starting 11 from your 15-player squad",
            inputSchema={
                "type": "object",
                "properties": {
                    "squad_ids": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["squad_ids"]
            }
        ),
        Tool(
            name="get_transfer_suggestions",
            description="Get transfer suggestions based on form, fixtures, and value",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {"type": "string", "enum": ["GK", "DEF", "MID", "FWD"]},
                    "budget": {"type": "number"},
                    "exclude_players": {"type": "array", "items": {"type": "integer"}},
                    "limit": {"type": "integer", "default": 10}
                }
            }
        ),
        Tool(
            name="plan_transfers",
            description="Plan transfers over multiple gameweeks with hit optimization",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_team": {"type": "array", "items": {"type": "integer"}},
                    "budget": {"type": "number"},
                    "num_weeks": {"type": "integer", "default": 5},
                    "free_transfers": {"type": "integer", "default": 1}
                },
                "required": ["current_team", "budget"]
            }
        ),
        Tool(
            name="evaluate_hit",
            description="Evaluate if a -4 hit transfer is worth it",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_out_id": {"type": "integer"},
                    "player_in_id": {"type": "integer"},
                    "horizon": {"type": "integer", "default": 5}
                },
                "required": ["player_out_id", "player_in_id"]
            }
        ),
        Tool(
            name="get_captaincy_picks",
            description="Get captain recommendations for your squad",
            inputSchema={
                "type": "object",
                "properties": {
                    "squad_ids": {"type": "array", "items": {"type": "integer"}},
                    "gameweek": {"type": "integer"},
                    "limit": {"type": "integer", "default": 5}
                },
                "required": ["squad_ids"]
            }
        ),
        Tool(
            name="suggest_wildcard_team",
            description="Suggest optimal wildcard team based on current data",
            inputSchema={
                "type": "object",
                "properties": {
                    "budget": {"type": "number", "default": 100.0},
                    "template": {"type": "boolean", "default": False}
                }
            }
        ),

        # ===== CHIP STRATEGY TOOLS (4) =====
        Tool(
            name="optimize_chip_timing",
            description="Find optimal gameweek to use a specific chip",
            inputSchema={
                "type": "object",
                "properties": {
                    "chip": {"type": "string", "enum": ["triple_captain", "bench_boost", "free_hit", "wildcard"]},
                    "remaining_gws": {"type": "integer", "default": 15}
                },
                "required": ["chip"]
            }
        ),
        Tool(
            name="analyze_triple_captain",
            description="Find best Triple Captain opportunities",
            inputSchema={
                "type": "object",
                "properties": {
                    "remaining_gws": {"type": "integer", "default": 15}
                }
            }
        ),
        Tool(
            name="analyze_bench_boost",
            description="Find best Bench Boost opportunities",
            inputSchema={
                "type": "object",
                "properties": {
                    "remaining_gws": {"type": "integer", "default": 15}
                }
            }
        ),
        Tool(
            name="get_chip_calendar",
            description="Get recommended chip usage calendar for the season",
            inputSchema={
                "type": "object",
                "properties": {
                    "remaining_gws": {"type": "integer", "default": 15}
                }
            }
        ),

        # ===== TEAM ANALYSIS TOOLS (2) =====
        Tool(
            name="analyze_my_team",
            description="Full analysis of your FPL team - weaknesses, suggestions, projected points",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_ids": {"type": "array", "items": {"type": "integer"}},
                    "budget_remaining": {"type": "number", "default": 0}
                },
                "required": ["team_ids"]
            }
        ),
        Tool(
            name="get_team_weaknesses",
            description="Identify weak spots in your team that need addressing",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_ids": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["team_ids"]
            }
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = await handle_tool(name, arguments)
        # Handle dataclass results
        if hasattr(result, '__dataclass_fields__'):
            result = asdict(result)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def handle_tool(name: str, args: dict) -> dict:
    """Route tool calls to implementations."""

    # === Existing Tools ===
    if name == "get_player":
        return await tool_get_player(args)
    elif name == "search_players":
        return await tool_search_players(args)
    elif name == "get_fixtures":
        return await tool_get_fixtures(args)
    elif name == "get_team_form":
        return await tool_get_team_form(args)
    elif name == "compare_players":
        return await tool_compare_players(args)
    elif name == "get_top_players":
        return await tool_get_top_players(args)
    elif name == "get_differentials":
        return await tool_get_differentials(args)
    elif name == "get_player_trend":
        return await tool_get_player_trend(args)
    elif name == "get_rising_players":
        return await tool_get_rising_players(args)

    # === Analytics Tools ===
    elif name == "get_expected_points":
        return await tool_get_expected_points(args)
    elif name == "get_fixture_difficulty":
        return await tool_get_fixture_difficulty(args)
    elif name == "get_easiest_fixtures":
        return await tool_get_easiest_fixtures(args)
    elif name == "get_price_predictions":
        return await tool_get_price_predictions(args)
    elif name == "analyze_luck":
        return await tool_analyze_luck(args)
    elif name == "get_overperformers":
        return await tool_get_overperformers(args)
    elif name == "get_underperformers":
        return await tool_get_underperformers(args)
    elif name == "get_template_players":
        return await tool_get_template_players(args)
    elif name == "predict_minutes":
        return await tool_predict_minutes(args)
    elif name == "get_dgw_bgw_outlook":
        return await tool_get_dgw_bgw_outlook(args)
    elif name == "get_bogey_teams":
        return await tool_get_bogey_teams(args)
    elif name == "get_value_picks":
        return await tool_get_value_picks(args)

    # === Optimization Tools ===
    elif name == "build_dream_team":
        return await tool_build_dream_team(args)
    elif name == "build_free_hit_team":
        return await tool_build_free_hit_team(args)
    elif name == "optimize_starting_11":
        return await tool_optimize_starting_11(args)
    elif name == "get_transfer_suggestions":
        return await tool_get_transfer_suggestions(args)
    elif name == "plan_transfers":
        return await tool_plan_transfers(args)
    elif name == "evaluate_hit":
        return await tool_evaluate_hit(args)
    elif name == "get_captaincy_picks":
        return await tool_get_captaincy_picks(args)
    elif name == "suggest_wildcard_team":
        return await tool_suggest_wildcard_team(args)

    # === Chip Strategy Tools ===
    elif name == "optimize_chip_timing":
        return await tool_optimize_chip_timing(args)
    elif name == "analyze_triple_captain":
        return await tool_analyze_triple_captain(args)
    elif name == "analyze_bench_boost":
        return await tool_analyze_bench_boost(args)
    elif name == "get_chip_calendar":
        return await tool_get_chip_calendar(args)

    # === Team Analysis Tools ===
    elif name == "analyze_my_team":
        return await tool_analyze_my_team(args)
    elif name == "get_team_weaknesses":
        return await tool_get_team_weaknesses(args)

    return {"error": f"Unknown tool: {name}"}


# ===== EXISTING TOOL IMPLEMENTATIONS =====

async def tool_get_player(args: dict) -> dict:
    player_id = args.get("player_id")
    name = args.get("name")

    if not player_id and name:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        player_id = row["id"] if row else None

    if not player_id:
        return {"error": "Player not found"}

    player = await cache_get_player(player_id)
    if not player:
        player = await fetch_one("""
            SELECT p.*, t.name as team_name, t.short_name as team
            FROM players p JOIN teams t ON p.team_id = t.id WHERE p.id = $1
        """, player_id)

    if not player:
        return {"error": "Player not found"}

    return {
        "id": player["id"],
        "name": player["web_name"],
        "team": player["team"],
        "position": POSITION_MAP.get(player["element_type"], "?"),
        "price": player["now_cost"] / 10 if player["now_cost"] else None,
        "form": float(player["form"]) if player["form"] else 0,
        "points": player["total_points"],
        "goals": player["goals_scored"],
        "assists": player["assists"],
        "xG": float(player["expected_goals"]) if player["expected_goals"] else 0,
        "xA": float(player["expected_assists"]) if player["expected_assists"] else 0,
        "ownership": float(player["selected_by_percent"]) if player["selected_by_percent"] else 0,
        "status": player["status"],
        "news": player["news"]
    }


async def tool_search_players(args: dict) -> dict:
    conditions = ["p.status = 'a'"]
    params = []
    idx = 1

    if args.get("team"):
        conditions.append(f"t.short_name = ${idx}")
        params.append(args["team"].upper())
        idx += 1

    if args.get("position"):
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(args["position"].upper(), 3))
        idx += 1

    if args.get("max_price"):
        conditions.append(f"p.now_cost <= ${idx}")
        params.append(int(args["max_price"] * 10))
        idx += 1

    if args.get("min_form"):
        conditions.append(f"p.form >= ${idx}")
        params.append(args["min_form"])
        idx += 1

    limit = args.get("limit", 10)
    params.append(limit)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, p.selected_by_percent, t.short_name as team
        FROM players p JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.form DESC NULLS LAST
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)
    return {
        "players": [{
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else None,
            "form": float(p["form"]) if p["form"] else 0,
            "points": p["total_points"]
        } for p in players]
    }


async def tool_get_fixtures(args: dict) -> dict:
    team_id = args.get("team_id")
    gw = args.get("gameweek")
    limit = args.get("num_fixtures", 5)

    if team_id:
        fixtures = await fetch_all("""
            SELECT f.*, th.short_name as home_team, ta.short_name as away_team
            FROM fixtures f
            JOIN teams th ON f.team_h = th.id
            JOIN teams ta ON f.team_a = ta.id
            WHERE (f.team_h = $1 OR f.team_a = $1) AND f.finished = false
            ORDER BY f.event LIMIT $2
        """, team_id, limit)
    elif gw:
        fixtures = await fetch_all("""
            SELECT f.*, th.short_name as home_team, ta.short_name as away_team
            FROM fixtures f
            JOIN teams th ON f.team_h = th.id
            JOIN teams ta ON f.team_a = ta.id
            WHERE f.event = $1
            ORDER BY f.kickoff_time
        """, gw)
    else:
        return {"error": "Provide team_id or gameweek"}

    return {
        "fixtures": [{
            "id": f["id"],
            "gameweek": f["event"],
            "home": f["home_team"],
            "away": f["away_team"],
            "home_difficulty": f["team_h_difficulty"],
            "away_difficulty": f["team_a_difficulty"],
            "kickoff": str(f["kickoff_time"]) if f["kickoff_time"] else None
        } for f in fixtures]
    }


async def tool_get_team_form(args: dict) -> dict:
    from analytics.form import calculate_team_form

    team_id = args.get("team_id")
    team_name = args.get("team_name")

    if not team_id and team_name:
        row = await fetch_one("SELECT id FROM teams WHERE name ILIKE $1", f"%{team_name}%")
        team_id = row["id"] if row else None

    if not team_id:
        return {"error": "Team not found"}

    form = await calculate_team_form(team_id)
    if not form:
        return {"error": "Could not calculate team form"}

    return asdict(form)


async def tool_compare_players(args: dict) -> dict:
    player_ids = args.get("player_ids", [])
    results = []
    for pid in player_ids:
        p = await tool_get_player({"player_id": pid})
        if "error" not in p:
            results.append(p)
    return {"players": results}


async def tool_get_top_players(args: dict) -> dict:
    pos = args.get("position")
    sort = args.get("sort_by", "form")
    limit = args.get("limit", 10)

    order_map = {
        "form": "p.form DESC NULLS LAST",
        "points": "p.total_points DESC",
        "value": "(p.total_points::float / NULLIF(p.now_cost, 0)) DESC"
    }

    conditions = ["p.status = 'a'", "p.minutes > 0"]
    params = []
    idx = 1

    if pos:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(pos.upper(), 3))
        idx += 1

    params.append(limit)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, t.short_name as team
        FROM players p JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY {order_map.get(sort, 'p.form DESC NULLS LAST')}
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)
    return {
        "players": [{
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else None,
            "form": float(p["form"]) if p["form"] else 0,
            "points": p["total_points"]
        } for p in players]
    }


async def tool_get_differentials(args: dict) -> dict:
    from analytics.ownership import get_differentials

    result = await get_differentials(
        position=args.get("position"),
        max_ownership=args.get("max_ownership", 10),
        min_form=args.get("min_form", 4),
        limit=args.get("limit", 10)
    )
    return {"differentials": result}


async def tool_get_player_trend(args: dict) -> dict:
    player_id = args.get("player_id")
    name = args.get("name")
    num_gws = args.get("num_gameweeks", 5)

    if not player_id and name:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        player_id = row["id"] if row else None

    if not player_id:
        return {"error": "Player not found"}

    player = await fetch_one("""
        SELECT p.web_name, t.short_name as team, p.element_type
        FROM players p JOIN teams t ON p.team_id = t.id WHERE p.id = $1
    """, player_id)

    if not player:
        return {"error": "Player not found"}

    snapshots = await fetch_all("""
        SELECT gameweek, form, total_points, now_cost, selected_by_percent,
               expected_goals, expected_assists, ict_index, minutes
        FROM player_snapshots
        WHERE player_id = $1
        ORDER BY gameweek DESC
        LIMIT $2
    """, player_id, num_gws)

    if not snapshots:
        return {"error": "No historical data available"}

    return {
        "player": {
            "id": player_id,
            "name": player["web_name"],
            "team": player["team"],
            "position": POSITION_MAP.get(player["element_type"], "?")
        },
        "trend": [{
            "gameweek": s["gameweek"],
            "form": float(s["form"]) if s["form"] else 0,
            "total_points": s["total_points"],
            "price": s["now_cost"] / 10 if s["now_cost"] else None,
            "ownership": float(s["selected_by_percent"]) if s["selected_by_percent"] else 0,
            "xG": float(s["expected_goals"]) if s["expected_goals"] else 0,
            "xA": float(s["expected_assists"]) if s["expected_assists"] else 0,
            "ict": float(s["ict_index"]) if s["ict_index"] else 0,
            "minutes": s["minutes"]
        } for s in snapshots]
    }


async def tool_get_rising_players(args: dict) -> dict:
    from analytics.form import detect_form_momentum

    result = await detect_form_momentum(
        position=args.get("position"),
        momentum_type="rising",
        limit=args.get("limit", 10)
    )
    return {"rising_players": result}


# ===== NEW ANALYTICS TOOL IMPLEMENTATIONS =====

async def tool_get_expected_points(args: dict) -> dict:
    from analytics.expected_points import calculate_expected_points

    player_id = args.get("player_id")
    name = args.get("name")

    if not player_id and name:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        player_id = row["id"] if row else None

    if not player_id:
        return {"error": "Player not found"}

    xp = await calculate_expected_points(player_id, args.get("num_fixtures", 5))
    if not xp:
        return {"error": "Could not calculate expected points"}

    return asdict(xp)


async def tool_get_fixture_difficulty(args: dict) -> dict:
    from analytics.fixtures import get_fixture_difficulty

    team_id = args.get("team_id")
    team_name = args.get("team_name")

    if not team_id and team_name:
        row = await fetch_one("SELECT id FROM teams WHERE name ILIKE $1 OR short_name ILIKE $1", f"%{team_name}%")
        team_id = row["id"] if row else None

    if not team_id:
        return {"error": "Team not found"}

    fd = await get_fixture_difficulty(team_id, args.get("num_fixtures", 10))
    if not fd:
        return {"error": "Could not get fixture difficulty"}

    return asdict(fd)


async def tool_get_easiest_fixtures(args: dict) -> dict:
    from analytics.fixtures import get_easiest_fixtures

    result = await get_easiest_fixtures(
        num_gameweeks=args.get("num_gameweeks", 5),
        limit=args.get("limit", 10)
    )
    return {"teams": result}


async def tool_get_price_predictions(args: dict) -> dict:
    from analytics.price_prediction import get_price_risers, get_price_fallers

    pred_type = args.get("type", "both")
    limit = args.get("limit", 15)

    result = {}
    if pred_type in ["risers", "both"]:
        result["risers"] = await get_price_risers(limit)
    if pred_type in ["fallers", "both"]:
        result["fallers"] = await get_price_fallers(limit)

    return result


async def tool_analyze_luck(args: dict) -> dict:
    from analytics.luck import analyze_luck

    player_id = args.get("player_id")
    name = args.get("name")

    if not player_id and name:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        player_id = row["id"] if row else None

    if not player_id:
        return {"error": "Player not found"}

    luck = await analyze_luck(player_id)
    if not luck:
        return {"error": "Could not analyze luck"}

    return asdict(luck)


async def tool_get_overperformers(args: dict) -> dict:
    from analytics.luck import get_overperformers

    result = await get_overperformers(
        position=args.get("position"),
        limit=args.get("limit", 15)
    )
    return {"overperformers": result, "warning": "These players may regress to their xG/xA"}


async def tool_get_underperformers(args: dict) -> dict:
    from analytics.luck import get_underperformers

    result = await get_underperformers(
        position=args.get("position"),
        limit=args.get("limit", 15)
    )
    return {"underperformers": result, "opportunity": "These players may improve to match their xG/xA"}


async def tool_get_template_players(args: dict) -> dict:
    from analytics.ownership import get_template_players

    result = await get_template_players(
        position=args.get("position"),
        min_ownership=args.get("min_ownership", 20),
        limit=args.get("limit", 20)
    )
    return {"template_players": result}


async def tool_predict_minutes(args: dict) -> dict:
    from analytics.minutes import predict_minutes

    player_id = args.get("player_id")
    name = args.get("name")

    if not player_id and name:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        player_id = row["id"] if row else None

    if not player_id:
        return {"error": "Player not found"}

    pred = await predict_minutes(player_id)
    if not pred:
        return {"error": "Could not predict minutes"}

    return asdict(pred)


async def tool_get_dgw_bgw_outlook(args: dict) -> dict:
    from analytics.chips import analyze_dgw_bgw

    return await analyze_dgw_bgw()


async def tool_get_bogey_teams(args: dict) -> dict:
    player_id = args.get("player_id")
    opponent = args.get("opponent_team")

    if not player_id:
        return {"error": "Player ID required"}

    # Get opponent team ID
    opp_team = await fetch_one("SELECT id, name FROM teams WHERE short_name ILIKE $1 OR name ILIKE $1", f"%{opponent}%")
    if not opp_team:
        return {"error": f"Opponent team '{opponent}' not found"}

    # Get historical performance
    history = await fetch_all("""
        SELECT ph.total_points, ph.goals_scored, ph.assists, ph.minutes
        FROM player_history ph
        WHERE ph.player_id = $1 AND ph.opponent_team = $2
    """, player_id, opp_team["id"])

    if not history:
        return {"message": "No historical data vs this opponent"}

    total_points = sum(h["total_points"] or 0 for h in history)
    games = len(history)
    avg_points = total_points / games if games > 0 else 0

    return {
        "player_id": player_id,
        "opponent": opp_team["name"],
        "games_played": games,
        "total_points": total_points,
        "average_points": round(avg_points, 2),
        "goals": sum(h["goals_scored"] or 0 for h in history),
        "assists": sum(h["assists"] or 0 for h in history),
        "is_bogey_team": avg_points < 3,
        "is_favourite": avg_points > 6
    }


async def tool_get_value_picks(args: dict) -> dict:
    from analytics.expected_points import get_value_picks

    result = await get_value_picks(
        position=args.get("position"),
        max_price=args.get("max_price"),
        limit=args.get("limit", 15)
    )
    return {"value_picks": result}


# ===== OPTIMIZATION TOOL IMPLEMENTATIONS =====

async def tool_build_dream_team(args: dict) -> dict:
    from optimization.dream_team import build_optimal_squad

    result = await build_optimal_squad(
        budget=args.get("budget", 100.0),
        strategy=args.get("strategy", "balanced"),
        exclude_players=args.get("exclude_players"),
        must_include=args.get("must_include")
    )

    if hasattr(result, '__dataclass_fields__'):
        return asdict(result)
    return result


async def tool_build_free_hit_team(args: dict) -> dict:
    from optimization.dream_team import build_free_hit_team

    gameweek = args.get("gameweek")
    if not gameweek:
        return {"error": "Gameweek required"}

    result = await build_free_hit_team(
        gameweek=gameweek,
        budget=args.get("budget", 100.0)
    )

    if hasattr(result, '__dataclass_fields__'):
        return asdict(result)
    return result


async def tool_optimize_starting_11(args: dict) -> dict:
    from optimization.dream_team import optimize_starting_11

    squad_ids = args.get("squad_ids", [])
    if not squad_ids:
        return {"error": "Squad IDs required"}

    return await optimize_starting_11(squad_ids)


async def tool_get_transfer_suggestions(args: dict) -> dict:
    from optimization.transfers import get_transfer_suggestions

    result = await get_transfer_suggestions(
        position=args.get("position"),
        budget=args.get("budget"),
        exclude_players=args.get("exclude_players"),
        limit=args.get("limit", 10)
    )

    return {"suggestions": [asdict(r) for r in result]}


async def tool_plan_transfers(args: dict) -> dict:
    from optimization.transfers import plan_transfers

    current_team = args.get("current_team", [])
    budget = args.get("budget")

    if not current_team or budget is None:
        return {"error": "Current team and budget required"}

    result = await plan_transfers(
        current_team=current_team,
        budget=budget,
        num_weeks=args.get("num_weeks", 5),
        free_transfers=args.get("free_transfers", 1)
    )

    if hasattr(result, '__dataclass_fields__'):
        return asdict(result)
    return result


async def tool_evaluate_hit(args: dict) -> dict:
    from analytics.hits import evaluate_hit

    out_id = args.get("player_out_id")
    in_id = args.get("player_in_id")

    if not out_id or not in_id:
        return {"error": "Both player_out_id and player_in_id required"}

    result = await evaluate_hit(out_id, in_id, args.get("horizon", 5))

    if result and hasattr(result, '__dataclass_fields__'):
        return asdict(result)
    return result or {"error": "Could not evaluate hit"}


async def tool_get_captaincy_picks(args: dict) -> dict:
    from optimization.captaincy import get_captaincy_picks

    squad_ids = args.get("squad_ids", [])
    if not squad_ids:
        return {"error": "Squad IDs required"}

    result = await get_captaincy_picks(
        squad_ids=squad_ids,
        gameweek=args.get("gameweek"),
        limit=args.get("limit", 5)
    )

    return {"captaincy_picks": [asdict(r) for r in result]}


async def tool_suggest_wildcard_team(args: dict) -> dict:
    from optimization.dream_team import suggest_wildcard_team

    result = await suggest_wildcard_team(
        budget=args.get("budget", 100.0),
        template=args.get("template", False)
    )

    if hasattr(result, '__dataclass_fields__'):
        return asdict(result)
    return result


# ===== CHIP STRATEGY TOOL IMPLEMENTATIONS =====

async def tool_optimize_chip_timing(args: dict) -> dict:
    from analytics.chips import (
        optimize_triple_captain,
        optimize_bench_boost,
        optimize_free_hit,
        optimize_wildcard
    )

    chip = args.get("chip")
    remaining_gws = args.get("remaining_gws", 15)

    chip_map = {
        "triple_captain": optimize_triple_captain,
        "bench_boost": optimize_bench_boost,
        "free_hit": optimize_free_hit,
        "wildcard": optimize_wildcard
    }

    if chip not in chip_map:
        return {"error": f"Unknown chip: {chip}"}

    result = await chip_map[chip](remaining_gws)
    return asdict(result)


async def tool_analyze_triple_captain(args: dict) -> dict:
    from analytics.chips import optimize_triple_captain

    result = await optimize_triple_captain(args.get("remaining_gws", 15))
    return asdict(result)


async def tool_analyze_bench_boost(args: dict) -> dict:
    from analytics.chips import optimize_bench_boost

    result = await optimize_bench_boost(args.get("remaining_gws", 15))
    return asdict(result)


async def tool_get_chip_calendar(args: dict) -> dict:
    from analytics.chips import get_chip_calendar

    result = await get_chip_calendar(args.get("remaining_gws", 15))

    return {
        "triple_captain": asdict(result.triple_captain),
        "bench_boost": asdict(result.bench_boost),
        "free_hit": asdict(result.free_hit),
        "wildcard": asdict(result.wildcard),
        "overall_strategy": result.overall_strategy
    }


# ===== TEAM ANALYSIS TOOL IMPLEMENTATIONS =====

async def tool_analyze_my_team(args: dict) -> dict:
    from analytics.expected_points import calculate_expected_points
    from analytics.minutes import predict_minutes
    from analytics.fixtures import get_fixture_difficulty
    from optimization.captaincy import get_captaincy_picks

    team_ids = args.get("team_ids", [])
    if not team_ids:
        return {"error": "Team IDs required"}

    budget_remaining = args.get("budget_remaining", 0)

    # Get player data
    placeholders = ", ".join(f"${i+1}" for i in range(len(team_ids)))
    players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, p.status, p.team_id, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
    """, *team_ids)

    # Analyze each player
    player_analysis = []
    total_xp = 0
    issues = []

    for p in players:
        xp_data = await calculate_expected_points(p["id"], 5)
        mins_data = await predict_minutes(p["id"])

        xp = xp_data.final_xp if xp_data else 0
        total_xp += xp

        analysis = {
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "form": float(p["form"] or 0),
            "xp_next_5": round(xp, 2)
        }

        if mins_data:
            analysis["rotation_risk"] = mins_data.rotation_risk
            if mins_data.rotation_risk == "high":
                issues.append(f"{p['web_name']} has high rotation risk")

        if p["status"] != "a":
            issues.append(f"{p['web_name']} is flagged ({p['status']})")

        if float(p["form"] or 0) < 3:
            issues.append(f"{p['web_name']} is in poor form ({p['form']})")

        player_analysis.append(analysis)

    # Get captain picks
    captain_picks = await get_captaincy_picks(team_ids, limit=3)

    # Get fixture analysis for owned teams
    team_ids_unique = list(set(p["team_id"] for p in players))
    fixture_outlook = []
    for tid in team_ids_unique[:5]:
        fd = await get_fixture_difficulty(tid, 5)
        if fd:
            fixture_outlook.append({
                "team_id": tid,
                "avg_difficulty": fd.avg_difficulty_next_5,
                "run_quality": fd.run_quality
            })

    return {
        "squad_analysis": player_analysis,
        "projected_xp_next_5": round(total_xp, 2),
        "issues": issues,
        "captain_picks": [asdict(c) for c in captain_picks[:3]],
        "fixture_outlook": fixture_outlook,
        "budget_remaining": budget_remaining
    }


async def tool_get_team_weaknesses(args: dict) -> dict:
    from analytics.minutes import predict_minutes
    from analytics.fixtures import get_fixture_difficulty
    from optimization.transfers import get_transfer_suggestions

    team_ids = args.get("team_ids", [])
    if not team_ids:
        return {"error": "Team IDs required"}

    placeholders = ", ".join(f"${i+1}" for i in range(len(team_ids)))
    players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.status, p.team_id, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
    """, *team_ids)

    weaknesses = []

    for p in players:
        issues = []
        priority = 0

        # Check status
        if p["status"] != "a":
            issues.append(f"Flagged: {p['status']}")
            priority += 30

        # Check form
        form = float(p["form"] or 0)
        if form < 2:
            issues.append(f"Very poor form: {form}")
            priority += 20
        elif form < 3:
            issues.append(f"Poor form: {form}")
            priority += 10

        # Check rotation risk
        mins = await predict_minutes(p["id"])
        if mins and mins.rotation_risk == "high":
            issues.append("High rotation risk")
            priority += 15

        # Check fixture difficulty
        fd = await get_fixture_difficulty(p["team_id"], 5)
        if fd and fd.avg_difficulty_next_5 > 3.5:
            issues.append(f"Tough fixtures ahead (avg: {fd.avg_difficulty_next_5})")
            priority += 10

        if issues:
            weaknesses.append({
                "player": {
                    "id": p["id"],
                    "name": p["web_name"],
                    "team": p["team"],
                    "position": POSITION_MAP.get(p["element_type"], "?"),
                    "price": p["now_cost"] / 10 if p["now_cost"] else 0
                },
                "issues": issues,
                "priority": priority
            })

    # Sort by priority
    weaknesses.sort(key=lambda x: -x["priority"])

    # Get suggested replacements for top weaknesses
    for w in weaknesses[:3]:
        suggestions = await get_transfer_suggestions(
            position=w["player"]["position"],
            budget=w["player"]["price"] + 2,
            exclude_players=team_ids,
            limit=3
        )
        w["suggested_replacements"] = [
            {"id": s.player_id, "name": s.player_name, "price": s.price, "form": s.form}
            for s in suggestions
        ]

    return {
        "weaknesses": weaknesses,
        "summary": f"Found {len(weaknesses)} players with issues. Top priority: {weaknesses[0]['player']['name'] if weaknesses else 'None'}"
    }


# === HTTP Routes ===

async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp.run(streams[0], streams[1], mcp.create_initialization_options())


async def handle_messages(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)


async def health_check(request: Request):
    return JSONResponse({"status": "ok", "server": "fpl-fantasy-god", "tools": 35})


# === Auth Middleware ===

class APIKeyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            settings = get_settings()
            if settings.api_key:
                path = scope.get("path", "")
                if path not in ["/", "/health"]:
                    headers = dict(scope.get("headers", []))
                    key = headers.get(b"x-api-key", b"").decode()
                    if key != settings.api_key:
                        response = JSONResponse({"error": "Invalid API key"}, status_code=401)
                        await response(scope, receive, send)
                        return
        await self.app(scope, receive, send)


# === App Lifecycle ===

async def startup():
    logger.info("=" * 60)
    logger.info("Starting FPL Fantasy God MCP Server...")
    logger.info("35 tools available for ultimate FPL domination")
    logger.info("=" * 60)
    await init_db()
    if await init_cache():
        await populate_cache_from_db()


async def shutdown():
    logger.info("Shutting down FPL Fantasy God...")
    await close_cache()
    await close_db()


# === Create App ===

app = Starlette(
    debug=False,
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
    ],
    routes=[
        Route("/", endpoint=health_check),
        Route("/health", endpoint=health_check),
        Route("/sse", endpoint=handle_sse),
        Route("/messages/", endpoint=handle_messages, methods=["POST"]),
    ],
    on_startup=[startup],
    on_shutdown=[shutdown],
)

app = APIKeyMiddleware(app)
