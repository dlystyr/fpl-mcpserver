"""FPLOracle MCP Server - 41 tools for ultimate FPL domination."""

import json
import logging
from contextvars import ContextVar
from urllib.parse import parse_qs
import httpx
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent, Prompt, PromptMessage, PromptArgument
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
from enrichment import enrich_player_async

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session context - stores team_id from SSE query params
session_team_id: ContextVar[int | None] = ContextVar("session_team_id", default=None)
# Module-level fallback storage for session team_id (keyed by scope id)
_session_team_ids: dict[int, int] = {}

mcp = Server("FPLOracle")
sse = SseServerTransport("/messages/")

# FPL API base URL
FPL_API_BASE = "https://fantasy.premierleague.com/api"

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# === MCP Prompts (System Instructions) ===

SYSTEM_PROMPT = """You are FPLOracle, an expert Fantasy Premier League assistant powered by live data.

## Critical Rules

1. **Data Source Priority**: The FPL Bootstrap API data is your source of truth for:
   - Player prices, ownership, form, and points
   - Team information and fixtures
   - Gameweek deadlines and status
   - Player availability and news

   NEVER rely on your training data for current FPL information - it is outdated.

2. **Always Verify Before Advising**: Before making any transfer, captaincy, or team suggestions:
   - Use the tools to fetch current data
   - Check player availability status (injured, suspended, doubtful)
   - Verify fixture difficulty ratings
   - Consider recent form (last 5 gameweeks)
   - enrich player data with the tool fpl_player_enriched

3. **Price Accuracy**: Player prices change daily based on transfers. Always fetch current prices.

4. **Fixture Awareness**:
   - Double Gameweeks (DGW) and Blank Gameweeks (BGW) significantly impact strategy
   - Use get_dgw_bgw_outlook to check for special gameweeks
   - FDR (Fixture Difficulty Rating) ranges from 1 (easiest) to 5 (hardest)

5. **Expected Stats Context**:
   - xG (expected goals) and xA (expected assists) indicate underlying performance
   - Players outperforming xG may regress; underperformers may improve
   - Use analyze_luck and get_overperformers/get_underperformers tools

6. **Transfer Rules**:
   - Maximum 3 players from any single team
   - Squad: 2 GK, 5 DEF, 5 MID, 3 FWD (15 total)
   - Selling price uses half-profit rule (not full purchase price)
   - -4 point hit per extra transfer beyond free transfers

7. **Chip Strategy**:
   - Wildcard: Unlimited free transfers for one gameweek
   - Free Hit: Temporary team for one gameweek only
   - Bench Boost: Points from all 15 players count
   - Triple Captain: Captain gets 3x points instead of 2x

8. **Current Season Context**: We are in the 2025-26 Premier League season.

## Response Style
- Be specific with player names and prices
- Quote actual statistics from the tools
- Explain the reasoning behind recommendations
- Acknowledge uncertainty when data is limited
"""

FPL_RULES_PROMPT = """## FPL Game Rules Reference

### Squad Composition
- 15 players total: 2 GK, 5 DEF, 5 MID, 3 FWD
- Maximum 3 players from any single Premier League team
- Starting 11 must have: 1 GK, minimum 3 DEF, minimum 1 FWD

### Budget
- Starting budget: 100.0m
- Player prices fluctuate based on transfer activity
- Selling price = purchase price + floor((current price - purchase price) / 2)

### Points Scoring
**Appearance**: 1pt (1-59 mins), 2pts (60+ mins)
**Goals**: GK/DEF 6pts, MID 5pts, FWD 4pts
**Assists**: 3pts
**Clean Sheet**: GK/DEF 4pts, MID 1pt
**Saves**: 1pt per 3 saves (GK only)
**Penalty Save**: 5pts
**Penalty Miss**: -2pts
**Yellow Card**: -1pt
**Red Card**: -3pts
**Own Goal**: -2pts
**Bonus Points**: 1-3pts to best performers in each match

### Transfers
- 1 free transfer per gameweek (max 5 banked with new rules)
- Additional transfers cost -4 points each
- Unlimited transfers on Wildcard

### Chips (one use per season, except Wildcard x2)
- **Wildcard**: Unlimited free transfers, resets team value
- **Free Hit**: Temporary squad for one GW, reverts after
- **Bench Boost**: All 15 players score points
- **Triple Captain**: Captain scores 3x points
"""


@mcp.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts for LLM guidance."""
    return [
        Prompt(
            name="fpl-oracle-system",
            description="System instructions for FPLOracle - use this to understand how to interact with FPL data correctly",
            arguments=[]
        ),
        Prompt(
            name="fpl-rules",
            description="Complete FPL game rules reference - scoring, transfers, chips, squad composition",
            arguments=[]
        ),
        Prompt(
            name="analyze-team",
            description="Structured prompt for analyzing a user's FPL team",
            arguments=[
                PromptArgument(
                    name="team_ids",
                    description="Comma-separated list of player IDs in the user's team",
                    required=True
                ),
                PromptArgument(
                    name="budget_remaining",
                    description="Remaining budget in millions (e.g., 2.5)",
                    required=False
                )
            ]
        ),
        Prompt(
            name="transfer-advice",
            description="Structured prompt for transfer recommendations",
            arguments=[
                PromptArgument(
                    name="player_out",
                    description="Name or ID of player to transfer out",
                    required=True
                ),
                PromptArgument(
                    name="budget",
                    description="Maximum budget for replacement in millions",
                    required=False
                )
            ]
        ),
    ]


@mcp.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None) -> list[PromptMessage]:
    """Return prompt content."""
    args = arguments or {}

    if name == "fpl-oracle-system":
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=SYSTEM_PROMPT)
            )
        ]

    elif name == "fpl-rules":
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=FPL_RULES_PROMPT)
            )
        ]

    elif name == "analyze-team":
        team_ids = args.get("team_ids", "")
        budget = args.get("budget_remaining", "0")
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=f"""Analyze my FPL team comprehensively.

My team player IDs: {team_ids}
Remaining budget: {budget}m

Please:
1. Use analyze_my_team tool to get full analysis
2. Identify any injured/doubtful players using player status
3. Check fixture difficulty for my players' teams
4. Suggest captain picks for the upcoming gameweek
5. Identify any obvious weaknesses or upgrade opportunities
6. Check if any players are overperforming their xG (regression risk)

{SYSTEM_PROMPT}""")
            )
        ]

    elif name == "transfer-advice":
        player_out = args.get("player_out", "")
        budget = args.get("budget", "")
        budget_text = f" with a maximum budget of {budget}m" if budget else ""
        return [
            PromptMessage(
                role="user",
                content=TextContent(type="text", text=f"""I want to transfer out {player_out}{budget_text}.

Please:
1. First check {player_out}'s current stats, form, and upcoming fixtures
2. Use get_transfer_suggestions to find replacements
3. Compare the top 3 alternatives using compare_players
4. Check fixture difficulty for each option
5. Consider ownership % for differential potential
6. Make a final recommendation with reasoning

{SYSTEM_PROMPT}""")
            )
        ]

    return [
        PromptMessage(
            role="user",
            content=TextContent(type="text", text=f"Unknown prompt: {name}")
        )
    ]


# === MCP Tools Definition (41 tools) ===

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

        # ===== ENRICHMENT TOOL (1) =====
        Tool(
            name="fpl_player_enriched",
            description="Get enriched player stats from Understat (xG, xA, shots, key passes) and FBref (SOT, progressive passes, passes into penalty area). Returns advanced analytics not available in FPL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "FPL player ID"},
                    "name": {"type": "string", "description": "Player name (alternative to ID)"}
                }
            }
        ),

        # ===== VISUALIZATION TOOLS (2) =====
        Tool(
            name="fpl_player_radar",
            description="Generate a radar chart comparing players across key metrics (xG, xA, form, points, ownership, ICT). Returns an image. Compare 2-4 players.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of 2-4 player IDs to compare"
                    },
                    "player_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Alternative: list of player names to compare"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["image", "svg", "both"],
                        "default": "image",
                        "description": "Output format: image (PNG), svg, or both"
                    }
                }
            }
        ),
        Tool(
            name="fpl_form_chart",
            description="Generate a line chart showing player form/points over recent gameweeks. Compare 1-4 players.",
            inputSchema={
                "type": "object",
                "properties": {
                    "player_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of 1-4 player IDs"
                    },
                    "player_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Alternative: list of player names"
                    },
                    "metric": {
                        "type": "string",
                        "enum": ["points", "form", "xG", "xA", "minutes"],
                        "default": "points",
                        "description": "Metric to chart over time"
                    },
                    "gameweeks": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of gameweeks to show"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["image", "svg", "both"],
                        "default": "image"
                    }
                }
            }
        ),

        # ===== MY TEAM TOOLS (3) - Uses team_id from SSE URL =====
        Tool(
            name="fpl_my_team",
            description="Get YOUR FPL team info (overall rank, points, transfers, chips used). Uses team_id from SSE connection URL (?team=123456). Can override with team_id param to view other managers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {
                        "type": "integer",
                        "description": "Optional: Override team ID to view a different manager"
                    }
                }
            }
        ),
        Tool(
            name="fpl_my_picks",
            description="Get YOUR FPL picks for a specific gameweek (starting 11, bench, captain, vice-captain, chips played). Uses team_id from SSE connection URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "gameweek": {
                        "type": "integer",
                        "description": "Gameweek number (defaults to current)"
                    },
                    "team_id": {
                        "type": "integer",
                        "description": "Optional: Override team ID"
                    }
                }
            }
        ),
        Tool(
            name="fpl_my_history",
            description="Get YOUR FPL season history (gameweek-by-gameweek points, rank, transfers, bench points). Uses team_id from SSE connection URL.",
            inputSchema={
                "type": "object",
                "properties": {
                    "team_id": {
                        "type": "integer",
                        "description": "Optional: Override team ID"
                    }
                }
            }
        ),
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    try:
        # Visualization tools return mixed content directly
        if name in ("fpl_player_radar", "fpl_form_chart"):
            return await handle_visualization_tool(name, arguments)

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

    # === Enrichment Tool ===
    elif name == "fpl_player_enriched":
        return await tool_fpl_player_enriched(args)

    # === My Team Tools (FPL API) ===
    elif name == "fpl_my_team":
        return await tool_fpl_my_team(args)
    elif name == "fpl_my_picks":
        return await tool_fpl_my_picks(args)
    elif name == "fpl_my_history":
        return await tool_fpl_my_history(args)

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


# ===== ENRICHMENT TOOL IMPLEMENTATION =====

async def tool_fpl_player_enriched(args: dict) -> dict:
    """Get enriched player stats from Understat and FBref."""
    player_id = args.get("player_id")
    name = args.get("name")

    # Resolve player
    if not player_id and name:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        player_id = row["id"] if row else None

    if not player_id and not name:
        return {"error": "Provide either player_id or name"}

    # Get player info for enrichment lookup
    if player_id:
        player = await fetch_one("""
            SELECT p.web_name, p.first_name, p.second_name, t.name as team_name,
                   p.element_type, p.now_cost, p.form, p.total_points,
                   p.goals_scored, p.assists, p.expected_goals, p.expected_assists
            FROM players p JOIN teams t ON p.team_id = t.id WHERE p.id = $1
        """, player_id)
    else:
        player = await fetch_one("""
            SELECT p.id, p.web_name, p.first_name, p.second_name, t.name as team_name,
                   p.element_type, p.now_cost, p.form, p.total_points,
                   p.goals_scored, p.assists, p.expected_goals, p.expected_assists
            FROM players p JOIN teams t ON p.team_id = t.id
            WHERE p.web_name ILIKE $1 LIMIT 1
        """, f"%{name}%")
        if player:
            player_id = player["id"]

    if not player:
        return {"error": f"Player not found: {player_id or name}"}

    # Try different name variants for matching
    full_name = f"{player['first_name']} {player['second_name']}"
    web_name = player["web_name"]
    team_name = player["team_name"]

    # Try full name first, then web_name
    enriched = await enrich_player_async(full_name, team_name)
    if not enriched.get("enrichment_available"):
        enriched = await enrich_player_async(web_name, team_name)

    return {
        "player": {
            "id": player_id,
            "name": web_name,
            "full_name": full_name,
            "team": team_name,
            "position": POSITION_MAP.get(player["element_type"], "?"),
            "price": player["now_cost"] / 10 if player["now_cost"] else None,
        },
        "fpl_stats": {
            "form": float(player["form"]) if player["form"] else 0,
            "total_points": player["total_points"],
            "goals": player["goals_scored"],
            "assists": player["assists"],
            "xG": float(player["expected_goals"]) if player["expected_goals"] else 0,
            "xA": float(player["expected_assists"]) if player["expected_assists"] else 0,
        },
        "enrichment": enriched,
    }


# ===== VISUALIZATION TOOLS IMPLEMENTATION =====

async def handle_visualization_tool(name: str, args: dict) -> list[TextContent | ImageContent]:
    """Handle visualization tools that return images."""
    if name == "fpl_player_radar":
        return await tool_fpl_player_radar(args)
    elif name == "fpl_form_chart":
        return await tool_fpl_form_chart(args)
    return [TextContent(type="text", text=json.dumps({"error": f"Unknown visualization: {name}"}))]


async def _resolve_player_ids(player_ids: list | None, player_names: list | None) -> list[int]:
    """Resolve player names to IDs if needed."""
    if player_ids:
        return player_ids

    if not player_names:
        return []

    resolved = []
    for name in player_names:
        row = await fetch_one("SELECT id FROM players WHERE web_name ILIKE $1 LIMIT 1", f"%{name}%")
        if row:
            resolved.append(row["id"])
    return resolved


async def _get_players_for_chart(player_ids: list[int]) -> list[dict]:
    """Fetch player data for charting."""
    if not player_ids:
        return []

    placeholders = ", ".join(f"${i+1}" for i in range(len(player_ids)))
    players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, p.goals_scored, p.assists, p.minutes,
               p.expected_goals, p.expected_assists, p.ict_index,
               p.selected_by_percent, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
    """, *player_ids)
    return list(players)


def _generate_radar_chart(players: list[dict], output_format: str = "image") -> dict:
    """Generate radar chart comparing players using Plotly."""
    import plotly.graph_objects as go
    import numpy as np
    import base64

    # Define metrics for radar
    categories = ['Form', 'Points/Game', 'xG', 'xA', 'ICT Index', 'Ownership']

    # Normalize data for each player (0-100 scale)
    def normalize(values, max_vals):
        return [min(100, (v / m * 100)) if m > 0 else 0 for v, m in zip(values, max_vals)]

    # Calculate max values for normalization
    max_vals = [
        max(float(p.get("form") or 0) for p in players) or 10,
        max((p.get("total_points") or 0) / max((p.get("minutes") or 1) / 90, 1) for p in players) or 10,
        max(float(p.get("expected_goals") or 0) for p in players) or 5,
        max(float(p.get("expected_assists") or 0) for p in players) or 5,
        max(float(p.get("ict_index") or 0) for p in players) or 100,
        max(float(p.get("selected_by_percent") or 0) for p in players) or 50,
    ]

    # Color palette
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

    fig = go.Figure()

    for i, player in enumerate(players):
        minutes = max(player.get("minutes") or 1, 1)
        nineties = minutes / 90

        raw_values = [
            float(player.get("form") or 0),
            (player.get("total_points") or 0) / max(nineties, 1),
            float(player.get("expected_goals") or 0),
            float(player.get("expected_assists") or 0),
            float(player.get("ict_index") or 0),
            float(player.get("selected_by_percent") or 0),
        ]

        normalized = normalize(raw_values, max_vals)
        # Close the radar by repeating first value
        normalized.append(normalized[0])
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=normalized,
            theta=cats,
            fill='toself',
            fillcolor=f'rgba{tuple(list(int(colors[i % len(colors)].lstrip("#")[j:j+2], 16) for j in (0, 2, 4)) + [0.25])}',
            line=dict(color=colors[i % len(colors)], width=2),
            name=f"{player['web_name']} ({player['team']})",
            hovertemplate=(
                f"<b>{player['web_name']}</b><br>"
                "Form: %{customdata[0]:.1f}<br>"
                "Pts/90: %{customdata[1]:.1f}<br>"
                "xG: %{customdata[2]:.2f}<br>"
                "xA: %{customdata[3]:.2f}<br>"
                "ICT: %{customdata[4]:.1f}<br>"
                "Own%: %{customdata[5]:.1f}%<extra></extra>"
            ),
            customdata=[raw_values] * len(cats)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=False,
                ticks='',
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#333'),
            ),
            bgcolor='rgba(255,255,255,0.9)',
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        title=dict(
            text="Player Comparison Radar",
            font=dict(size=16, color='#333'),
            x=0.5
        ),
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=80, b=80),
        width=600,
        height=500,
    )

    result = {}

    if output_format in ("image", "both"):
        # Generate PNG
        img_bytes = fig.to_image(format="png", scale=2)
        result["image_base64"] = base64.b64encode(img_bytes).decode()

    if output_format in ("svg", "both"):
        # Generate SVG
        result["svg"] = fig.to_image(format="svg").decode()

    return result


def _generate_form_chart(players: list[dict], snapshots_by_player: dict, metric: str, output_format: str = "image") -> dict:
    """Generate line chart showing player performance over gameweeks."""
    import plotly.graph_objects as go
    import base64

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']

    metric_labels = {
        "points": "Points",
        "form": "Form",
        "xG": "Expected Goals",
        "xA": "Expected Assists",
        "minutes": "Minutes"
    }

    fig = go.Figure()

    for i, player in enumerate(players):
        snapshots = snapshots_by_player.get(player["id"], [])
        if not snapshots:
            continue

        # Sort by gameweek
        snapshots = sorted(snapshots, key=lambda x: x.get("gameweek", 0))

        gws = [s.get("gameweek") for s in snapshots]

        # Extract metric values
        if metric == "points":
            values = [s.get("total_points", 0) for s in snapshots]
        elif metric == "form":
            values = [float(s.get("form") or 0) for s in snapshots]
        elif metric == "xG":
            values = [float(s.get("expected_goals") or 0) for s in snapshots]
        elif metric == "xA":
            values = [float(s.get("expected_assists") or 0) for s in snapshots]
        elif metric == "minutes":
            values = [s.get("minutes", 0) for s in snapshots]
        else:
            values = [s.get("total_points", 0) for s in snapshots]

        fig.add_trace(go.Scatter(
            x=gws,
            y=values,
            mode='lines+markers',
            name=f"{player['web_name']} ({player['team']})",
            line=dict(color=colors[i % len(colors)], width=3),
            marker=dict(size=8, color=colors[i % len(colors)]),
            hovertemplate=f"<b>{player['web_name']}</b><br>GW%{{x}}: %{{y:.1f}}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(
            text=f"Player {metric_labels.get(metric, metric)} Over Time",
            font=dict(size=16, color='#333'),
            x=0.5
        ),
        xaxis=dict(
            title="Gameweek",
            tickmode='linear',
            tick0=1,
            dtick=1,
            gridcolor='rgba(0,0,0,0.1)',
        ),
        yaxis=dict(
            title=metric_labels.get(metric, metric),
            gridcolor='rgba(0,0,0,0.1)',
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        paper_bgcolor='white',
        plot_bgcolor='rgba(250,250,250,1)',
        margin=dict(l=60, r=40, t=60, b=80),
        width=700,
        height=400,
    )

    result = {}

    if output_format in ("image", "both"):
        img_bytes = fig.to_image(format="png", scale=2)
        result["image_base64"] = base64.b64encode(img_bytes).decode()

    if output_format in ("svg", "both"):
        result["svg"] = fig.to_image(format="svg").decode()

    return result


async def tool_fpl_player_radar(args: dict) -> list[TextContent | ImageContent]:
    """Generate radar chart comparing players."""
    player_ids = await _resolve_player_ids(args.get("player_ids"), args.get("player_names"))

    if len(player_ids) < 2:
        return [TextContent(type="text", text=json.dumps({"error": "Need at least 2 players to compare"}))]
    if len(player_ids) > 4:
        player_ids = player_ids[:4]  # Limit to 4 players

    players = await _get_players_for_chart(player_ids)
    if len(players) < 2:
        return [TextContent(type="text", text=json.dumps({"error": "Could not find enough players"}))]

    output_format = args.get("format", "image")

    try:
        chart_data = _generate_radar_chart(players, output_format)
    except Exception as e:
        logger.error(f"Radar chart generation failed: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": f"Chart generation failed: {e}"}))]

    result = []

    # Build text summary
    summary_lines = ["**Player Comparison:**"]
    for p in players:
        minutes = max(p.get("minutes") or 1, 1)
        pts_per_90 = (p.get("total_points") or 0) / (minutes / 90)
        summary_lines.append(
            f"- **{p['web_name']}** ({p['team']}): Form {p.get('form', 0)}, "
            f"Pts/90: {pts_per_90:.1f}, xG: {float(p.get('expected_goals') or 0):.2f}, "
            f"xA: {float(p.get('expected_assists') or 0):.2f}"
        )

    # Add image if generated
    if "image_base64" in chart_data:
        result.append(ImageContent(
            type="image",
            data=chart_data["image_base64"],
            mimeType="image/png"
        ))

    # Add SVG as text if generated
    if "svg" in chart_data:
        summary_lines.append("\n**SVG Chart:**")
        summary_lines.append(chart_data["svg"])

    result.append(TextContent(type="text", text="\n".join(summary_lines)))

    return result


async def tool_fpl_form_chart(args: dict) -> list[TextContent | ImageContent]:
    """Generate line chart showing player performance over gameweeks."""
    player_ids = await _resolve_player_ids(args.get("player_ids"), args.get("player_names"))

    if not player_ids:
        return [TextContent(type="text", text=json.dumps({"error": "No players specified"}))]
    if len(player_ids) > 4:
        player_ids = player_ids[:4]

    players = await _get_players_for_chart(player_ids)
    if not players:
        return [TextContent(type="text", text=json.dumps({"error": "Could not find players"}))]

    metric = args.get("metric", "points")
    num_gws = args.get("gameweeks", 10)
    output_format = args.get("format", "image")

    # Fetch snapshots for each player
    snapshots_by_player = {}
    for player in players:
        snapshots = await fetch_all("""
            SELECT gameweek, form, total_points, expected_goals, expected_assists, minutes
            FROM player_snapshots
            WHERE player_id = $1
            ORDER BY gameweek DESC
            LIMIT $2
        """, player["id"], num_gws)
        snapshots_by_player[player["id"]] = list(snapshots)

    try:
        chart_data = _generate_form_chart(players, snapshots_by_player, metric, output_format)
    except Exception as e:
        logger.error(f"Form chart generation failed: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": f"Chart generation failed: {e}"}))]

    result = []

    # Build text summary
    metric_labels = {"points": "Points", "form": "Form", "xG": "xG", "xA": "xA", "minutes": "Minutes"}
    summary_lines = [f"**{metric_labels.get(metric, metric)} Trend:**"]
    for p in players:
        snaps = snapshots_by_player.get(p["id"], [])
        if snaps:
            latest = snaps[0] if snaps else {}
            summary_lines.append(
                f"- **{p['web_name']}**: Current {metric_labels.get(metric, metric)}: "
                f"{latest.get(metric, latest.get('total_points', 'N/A'))}"
            )

    if "image_base64" in chart_data:
        result.append(ImageContent(
            type="image",
            data=chart_data["image_base64"],
            mimeType="image/png"
        ))

    if "svg" in chart_data:
        summary_lines.append("\n**SVG Chart:**")
        summary_lines.append(chart_data["svg"])

    result.append(TextContent(type="text", text="\n".join(summary_lines)))

    return result


# ===== MY TEAM TOOLS IMPLEMENTATION (FPL API) =====

def _get_team_id(args: dict) -> int | None:
    """Get team_id from args or session context."""
    # Explicit override takes priority
    if args.get("team_id"):
        return args["team_id"]
    # Try context variable first
    ctx_team_id = session_team_id.get()
    if ctx_team_id:
        return ctx_team_id
    # Fall back to module-level storage (in case context doesn't propagate)
    if _session_team_ids:
        # Return most recent session's team_id
        return list(_session_team_ids.values())[-1] if _session_team_ids else None
    return None


async def _fetch_fpl_api(endpoint: str) -> dict | None:
    """Fetch data from FPL API."""
    url = f"{FPL_API_BASE}/{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, headers={
                "User-Agent": "FPLOracle/1.0"
            })
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"FPL API error: {e.response.status_code} for {url}")
        return None
    except Exception as e:
        logger.error(f"FPL API request failed: {e}")
        return None


async def _get_current_gameweek() -> int | None:
    """Get the current gameweek number."""
    data = await _fetch_fpl_api("bootstrap-static/")
    if not data:
        return None
    for event in data.get("events", []):
        if event.get("is_current"):
            return event.get("id")
    return None


async def tool_fpl_my_team(args: dict) -> dict:
    """Get FPL manager info from the official API."""
    team_id = _get_team_id(args)

    if not team_id:
        return {
            "error": "No team ID configured",
            "hint": "Add ?team=YOUR_TEAM_ID to your SSE connection URL, e.g., https://your-server/sse?team=2866423",
            "how_to_find": "Your team ID is in your FPL URL: fantasy.premierleague.com/entry/YOUR_ID/event/1"
        }

    data = await _fetch_fpl_api(f"entry/{team_id}/")
    if not data:
        return {"error": f"Could not fetch team {team_id}. Check the team ID is correct."}

    # Extract useful info
    return {
        "team_id": team_id,
        "manager": f"{data.get('player_first_name', '')} {data.get('player_last_name', '')}",
        "team_name": data.get("name"),
        "region": data.get("player_region_name"),
        "started_event": data.get("started_event"),
        "current_event": data.get("current_event"),
        "summary_overall_points": data.get("summary_overall_points"),
        "summary_overall_rank": data.get("summary_overall_rank"),
        "summary_event_points": data.get("summary_event_points"),
        "summary_event_rank": data.get("summary_event_rank"),
        "last_deadline_bank": data.get("last_deadline_bank", 0) / 10,  # Convert to millions
        "last_deadline_value": data.get("last_deadline_value", 0) / 10,
        "last_deadline_total_transfers": data.get("last_deadline_total_transfers"),
        "leagues": {
            "classic": [
                {"name": l.get("name"), "rank": l.get("entry_rank")}
                for l in data.get("leagues", {}).get("classic", [])[:5]
            ]
        }
    }


async def tool_fpl_my_picks(args: dict) -> dict:
    """Get FPL picks for a specific gameweek."""
    team_id = _get_team_id(args)

    if not team_id:
        return {
            "error": "No team ID configured",
            "hint": "Add ?team=YOUR_TEAM_ID to your SSE connection URL"
        }

    gameweek = args.get("gameweek")
    if not gameweek:
        gameweek = await _get_current_gameweek()
        if not gameweek:
            return {"error": "Could not determine current gameweek"}

    data = await _fetch_fpl_api(f"entry/{team_id}/event/{gameweek}/picks/")
    if not data:
        return {"error": f"Could not fetch picks for team {team_id} GW{gameweek}"}

    picks = data.get("picks", [])
    entry_history = data.get("entry_history", {})

    # Get player details from our database
    player_ids = [p["element"] for p in picks]
    placeholders = ", ".join(f"${i+1}" for i in range(len(player_ids)))
    players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, t.short_name as team
        FROM players p JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
    """, *player_ids)
    player_map = {p["id"]: p for p in players}

    # Build picks with details
    starting_11 = []
    bench = []
    captain_id = None
    vice_captain_id = None

    for pick in picks:
        player = player_map.get(pick["element"], {})
        pick_info = {
            "id": pick["element"],
            "name": player.get("web_name", f"Player {pick['element']}"),
            "team": player.get("team", "?"),
            "position": POSITION_MAP.get(player.get("element_type"), "?"),
            "multiplier": pick["multiplier"],
            "is_captain": pick["is_captain"],
            "is_vice_captain": pick["is_vice_captain"],
        }

        if pick["is_captain"]:
            captain_id = pick["element"]
        if pick["is_vice_captain"]:
            vice_captain_id = pick["element"]

        if pick["position"] <= 11:
            starting_11.append(pick_info)
        else:
            bench.append(pick_info)

    return {
        "team_id": team_id,
        "gameweek": gameweek,
        "active_chip": data.get("active_chip"),
        "starting_11": starting_11,
        "bench": bench,
        "captain": player_map.get(captain_id, {}).get("web_name") if captain_id else None,
        "vice_captain": player_map.get(vice_captain_id, {}).get("web_name") if vice_captain_id else None,
        "event_transfers": entry_history.get("event_transfers", 0),
        "event_transfers_cost": entry_history.get("event_transfers_cost", 0),
        "points": entry_history.get("points"),
        "total_points": entry_history.get("total_points"),
        "overall_rank": entry_history.get("overall_rank"),
        "bank": entry_history.get("bank", 0) / 10,
        "value": entry_history.get("value", 0) / 10,
    }


async def tool_fpl_my_history(args: dict) -> dict:
    """Get FPL season history for a manager."""
    team_id = _get_team_id(args)

    if not team_id:
        return {
            "error": "No team ID configured",
            "hint": "Add ?team=YOUR_TEAM_ID to your SSE connection URL"
        }

    data = await _fetch_fpl_api(f"entry/{team_id}/history/")
    if not data:
        return {"error": f"Could not fetch history for team {team_id}"}

    current_season = data.get("current", [])
    past_seasons = data.get("past", [])
    chips = data.get("chips", [])

    # Format current season GW-by-GW
    gw_history = []
    for gw in current_season:
        gw_history.append({
            "gameweek": gw.get("event"),
            "points": gw.get("points"),
            "total_points": gw.get("total_points"),
            "rank": gw.get("rank"),
            "overall_rank": gw.get("overall_rank"),
            "bank": gw.get("bank", 0) / 10,
            "value": gw.get("value", 0) / 10,
            "transfers": gw.get("event_transfers"),
            "transfers_cost": gw.get("event_transfers_cost"),
            "points_on_bench": gw.get("points_on_bench"),
        })

    # Chips used
    chips_used = [
        {"name": c.get("name"), "gameweek": c.get("event")}
        for c in chips
    ]

    return {
        "team_id": team_id,
        "gameweek_history": gw_history,
        "chips_used": chips_used,
        "chips_remaining": _get_remaining_chips(chips_used),
        "past_seasons": [
            {
                "season": s.get("season_name"),
                "total_points": s.get("total_points"),
                "rank": s.get("rank"),
            }
            for s in past_seasons
        ],
    }


def _get_remaining_chips(chips_used: list) -> list:
    """Calculate which chips are still available."""
    all_chips = {"wildcard", "freehit", "bboost", "3xc"}  # FPL chip names
    used = {c["name"] for c in chips_used}
    # Wildcard can be used twice (once per half of season)
    wildcard_count = sum(1 for c in chips_used if c["name"] == "wildcard")
    remaining = list(all_chips - used)
    if wildcard_count < 2:
        remaining.append("wildcard")
    return remaining


# === HTTP Routes ===

class SSEHandler:
    """ASGI handler for SSE that properly parses query params."""

    async def __call__(self, scope, receive, send):
        # Parse team_id from query params: /sse?team=123456
        team_id = None
        query_string = scope.get("query_string", b"").decode()
        if query_string:
            params = parse_qs(query_string)
            team_param = params.get("team", [None])[0]
            if team_param:
                try:
                    team_id = int(team_param)
                    logger.info(f"Session team_id set to {team_id}")
                except ValueError:
                    logger.warning(f"Invalid team_id in query: {team_param}")

        # Store in module-level dict and context var
        session_id = id(scope)
        if team_id:
            _session_team_ids[session_id] = team_id

        token = session_team_id.set(team_id)
        try:
            async with sse.connect_sse(scope, receive, send) as streams:
                await mcp.run(streams[0], streams[1], mcp.create_initialization_options())
        finally:
            session_team_id.reset(token)
            _session_team_ids.pop(session_id, None)


class MessagesHandler:
    """ASGI handler for MCP messages."""

    async def __call__(self, scope, receive, send):
        await sse.handle_post_message(scope, receive, send)


# Create handler instances
handle_sse_endpoint = SSEHandler()
handle_messages_endpoint = MessagesHandler()


async def health_check(request: Request):
    return JSONResponse({"status": "ok", "server": "FPLOracle", "tools": 41})


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
    logger.info("Starting FPLOracle MCP Server...")
    logger.info("41 tools available for ultimate FPL domination")
    logger.info("=" * 60)
    await init_db()
    if await init_cache():
        await populate_cache_from_db()


async def shutdown():
    logger.info("Shutting down FPLOracle...")
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
        Route("/sse", endpoint=handle_sse_endpoint),
        Route("/messages/", endpoint=handle_messages_endpoint, methods=["POST"]),
    ],
    on_startup=[startup],
    on_shutdown=[shutdown],
)

app = APIKeyMiddleware(app)
