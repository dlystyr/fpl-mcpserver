"""FPL MCP Server with SSE transport."""

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

from config import get_settings
from database import init_db, close_db, fetch_all, fetch_one
from cache import init_cache, close_cache, populate_cache_from_db, cache_get_all_players, cache_get_player

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = Server("fpl-mcp")
sse = SseServerTransport("/messages/")

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# === MCP Tools ===

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_player",
            description="Get player info by ID or name",
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
            description="Search players by criteria",
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
            description="Get team's recent form",
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
            description="Compare multiple players",
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
            description="Find low-ownership players with good form",
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
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        result = await handle_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def handle_tool(name: str, args: dict) -> dict:
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
    return {"error": f"Unknown tool: {name}"}


# === Tool Implementations ===

async def tool_get_player(args: dict) -> dict:
    player_id = args.get("player_id")
    name = args.get("name")

    if not player_id and name:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        player_id = row["id"] if row else None

    if not player_id:
        return {"error": "Player not found"}

    # Try cache first
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
    team_id = args.get("team_id")
    team_name = args.get("team_name")

    if not team_id and team_name:
        row = await fetch_one("SELECT id FROM teams WHERE name ILIKE $1", f"%{team_name}%")
        team_id = row["id"] if row else None

    if not team_id:
        return {"error": "Team not found"}

    team = await fetch_one("SELECT * FROM teams WHERE id = $1", team_id)
    fixtures = await fetch_all("""
        SELECT f.*,
               CASE WHEN f.team_h = $1 THEN f.team_h_score ELSE f.team_a_score END as goals_for,
               CASE WHEN f.team_h = $1 THEN f.team_a_score ELSE f.team_h_score END as goals_against
        FROM fixtures f
        WHERE (f.team_h = $1 OR f.team_a = $1) AND f.finished = true
        ORDER BY f.event DESC LIMIT 5
    """, team_id)

    results = []
    for f in fixtures:
        gf, ga = f["goals_for"] or 0, f["goals_against"] or 0
        if gf > ga:
            results.append("W")
        elif gf < ga:
            results.append("L")
        else:
            results.append("D")

    return {
        "team": team["name"],
        "form": "".join(results),
        "strength": {
            "attack_home": team["strength_attack_home"],
            "attack_away": team["strength_attack_away"],
            "defence_home": team["strength_defence_home"],
            "defence_away": team["strength_defence_away"]
        }
    }


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
    max_own = args.get("max_ownership", 10)
    min_form = args.get("min_form", 4)
    pos = args.get("position")
    limit = args.get("limit", 10)

    conditions = ["p.status = 'a'", "p.selected_by_percent <= $1", "p.form >= $2"]
    params = [max_own, min_form]
    idx = 3

    if pos:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(pos.upper(), 3))
        idx += 1

    params.append(limit)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, p.selected_by_percent, t.short_name as team
        FROM players p JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.form DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)
    return {
        "differentials": [{
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else None,
            "form": float(p["form"]) if p["form"] else 0,
            "ownership": float(p["selected_by_percent"]) if p["selected_by_percent"] else 0
        } for p in players]
    }


async def tool_get_player_trend(args: dict) -> dict:
    player_id = args.get("player_id")
    name = args.get("name")
    num_gws = args.get("num_gameweeks", 5)

    if not player_id and name:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        player_id = row["id"] if row else None

    if not player_id:
        return {"error": "Player not found"}

    # Get player info
    player = await fetch_one("""
        SELECT p.web_name, t.short_name as team, p.element_type
        FROM players p JOIN teams t ON p.team_id = t.id WHERE p.id = $1
    """, player_id)

    if not player:
        return {"error": "Player not found"}

    # Get snapshots
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
    pos = args.get("position")
    min_improvement = args.get("min_improvement", 1.0)
    limit = args.get("limit", 10)

    # Compare current form to form 3 gameweeks ago
    conditions = ["p.status = 'a'", "p.minutes > 0"]
    params = []
    idx = 1

    if pos:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(pos.upper(), 3))
        idx += 1

    params.extend([min_improvement, limit])

    query = f"""
        WITH current_gw AS (
            SELECT MAX(gameweek) as gw FROM player_snapshots
        ),
        player_trends AS (
            SELECT
                ps_now.player_id,
                ps_now.form as current_form,
                ps_old.form as old_form,
                (ps_now.form - COALESCE(ps_old.form, 0)) as form_change
            FROM player_snapshots ps_now
            CROSS JOIN current_gw cg
            LEFT JOIN player_snapshots ps_old
                ON ps_now.player_id = ps_old.player_id
                AND ps_old.gameweek = cg.gw - 3
            WHERE ps_now.gameweek = cg.gw
        )
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, t.short_name as team, pt.form_change
        FROM players p
        JOIN teams t ON p.team_id = t.id
        JOIN player_trends pt ON p.id = pt.player_id
        WHERE {' AND '.join(conditions)} AND pt.form_change >= ${idx}
        ORDER BY pt.form_change DESC
        LIMIT ${idx + 1}
    """

    players = await fetch_all(query, *params)
    return {
        "rising_players": [{
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else None,
            "form": float(p["form"]) if p["form"] else 0,
            "form_change": float(p["form_change"]) if p["form_change"] else 0
        } for p in players]
    }


# === HTTP Routes ===

async def handle_sse(request: Request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await mcp.run(streams[0], streams[1], mcp.create_initialization_options())


async def handle_messages(request: Request):
    await sse.handle_post_message(request.scope, request.receive, request._send)


async def health_check(request: Request):
    return JSONResponse({"status": "ok", "server": "fpl-mcp"})


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
    logger.info("Starting FPL MCP Server...")
    await init_db()
    if await init_cache():
        await populate_cache_from_db()


async def shutdown():
    logger.info("Shutting down...")
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
