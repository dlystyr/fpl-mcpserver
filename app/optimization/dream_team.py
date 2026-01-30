"""Dream team optimization using linear programming (PuLP)."""

from dataclasses import dataclass
import logging

try:
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logging.warning("PuLP not installed. Optimization will use fallback method.")

from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_LIMITS = {"GK": 1, "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}


@dataclass
class OptimalSquad:
    """Optimal 15-player squad."""
    players: list[dict]
    total_cost: float
    total_xp: float
    formation: str
    starting_11: list[dict]
    bench: list[dict]
    captain: dict | None
    vice_captain: dict | None


async def _get_player_pool(
    exclude_players: list[int] | None = None,
    must_include: list[int] | None = None
) -> list[dict]:
    """Get all available players for optimization."""
    # Import here to avoid circular dependency
    from analytics.expected_points import calculate_expected_points

    players = await fetch_all("""
        SELECT p.id, p.web_name, p.team_id, p.element_type, p.now_cost,
               p.form, p.total_points, p.expected_goals, p.expected_assists,
               p.minutes, p.status, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.status = 'a' AND p.minutes > 90
        ORDER BY p.form DESC
        LIMIT 300
    """)

    exclude_set = set(exclude_players or [])
    must_include_set = set(must_include or [])

    pool = []
    for p in players:
        if p["id"] in exclude_set:
            continue

        # Calculate xP for each player
        xp_data = await calculate_expected_points(p["id"], num_fixtures=5)
        xp = xp_data.final_xp if xp_data else float(p["form"] or 0)

        pool.append({
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "team_id": p["team_id"],
            "position": POSITION_MAP.get(p["element_type"], "MID"),
            "position_id": p["element_type"],
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "cost": p["now_cost"] or 0,  # In tenths
            "xp": xp,
            "form": float(p["form"] or 0),
            "points": p["total_points"] or 0,
            "must_include": p["id"] in must_include_set
        })

    return pool


async def build_optimal_squad(
    budget: float = 100.0,
    strategy: str = "balanced",
    exclude_players: list[int] | None = None,
    must_include: list[int] | None = None
) -> OptimalSquad | dict:
    """
    Build optimal 15-player squad using linear programming.

    Constraints:
    - Budget <= specified (default £100m)
    - 2 GK, 5 DEF, 5 MID, 3 FWD
    - Max 3 from any team
    - Maximize expected points
    """
    pool = await _get_player_pool(exclude_players, must_include)

    if not pool:
        return {"error": "No players available for optimization"}

    budget_tenths = int(budget * 10)

    if PULP_AVAILABLE:
        return await _optimize_with_pulp(pool, budget_tenths, strategy)
    else:
        return await _optimize_fallback(pool, budget_tenths, strategy)


async def _optimize_with_pulp(
    pool: list[dict],
    budget: int,
    strategy: str
) -> OptimalSquad:
    """Optimize using PuLP linear programming."""
    prob = LpProblem("FPL_Squad", LpMaximize)

    # Decision variables: 1 if player selected, 0 otherwise
    player_vars = {p["id"]: LpVariable(f"player_{p['id']}", cat="Binary") for p in pool}

    # Objective: maximize xP (with strategy adjustments)
    if strategy == "attacking":
        weights = {"GK": 0.8, "DEF": 0.9, "MID": 1.1, "FWD": 1.2}
    elif strategy == "defensive":
        weights = {"GK": 1.2, "DEF": 1.1, "MID": 1.0, "FWD": 0.9}
    else:  # balanced
        weights = {"GK": 1.0, "DEF": 1.0, "MID": 1.0, "FWD": 1.0}

    prob += lpSum(
        player_vars[p["id"]] * p["xp"] * weights.get(p["position"], 1.0)
        for p in pool
    )

    # Constraint: Budget
    prob += lpSum(player_vars[p["id"]] * p["cost"] for p in pool) <= budget

    # Constraint: Position limits (2 GK, 5 DEF, 5 MID, 3 FWD)
    for pos, limit in POSITION_LIMITS.items():
        pos_players = [p for p in pool if p["position"] == pos]
        prob += lpSum(player_vars[p["id"]] for p in pos_players) == limit

    # Constraint: Max 3 from any team
    teams = set(p["team_id"] for p in pool)
    for team_id in teams:
        team_players = [p for p in pool if p["team_id"] == team_id]
        prob += lpSum(player_vars[p["id"]] for p in team_players) <= 3

    # Constraint: Must include specified players
    for p in pool:
        if p["must_include"]:
            prob += player_vars[p["id"]] == 1

    # Solve
    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        return {"error": f"Optimization failed: {LpStatus[prob.status]}"}

    # Extract selected players
    selected = [p for p in pool if player_vars[p["id"]].value() == 1]

    return await _format_squad(selected)


async def _optimize_fallback(
    pool: list[dict],
    budget: int,
    strategy: str
) -> OptimalSquad:
    """Fallback optimization using greedy algorithm."""
    # Sort by xP per cost (value)
    for p in pool:
        p["value"] = p["xp"] / p["cost"] if p["cost"] > 0 else 0

    # Greedy selection by position
    selected = []
    remaining_budget = budget
    team_counts = {}

    for pos, limit in POSITION_LIMITS.items():
        pos_players = sorted(
            [p for p in pool if p["position"] == pos and p["id"] not in [s["id"] for s in selected]],
            key=lambda x: (-x["must_include"], -x["xp"])  # Must include first, then by xP
        )

        pos_selected = 0
        for p in pos_players:
            if pos_selected >= limit:
                break
            if p["cost"] > remaining_budget:
                continue
            if team_counts.get(p["team_id"], 0) >= 3:
                continue

            selected.append(p)
            remaining_budget -= p["cost"]
            team_counts[p["team_id"]] = team_counts.get(p["team_id"], 0) + 1
            pos_selected += 1

    return await _format_squad(selected)


async def _format_squad(selected: list[dict]) -> OptimalSquad:
    """Format selected players into OptimalSquad."""
    total_cost = sum(p["cost"] for p in selected) / 10
    total_xp = sum(p["xp"] for p in selected)

    # Sort by position for display
    pos_order = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    selected.sort(key=lambda x: (pos_order.get(x["position"], 5), -x["xp"]))

    # Pick starting 11 (best by xP within position constraints)
    starting_11, bench = await _pick_starting_11(selected)

    # Determine formation
    def_count = sum(1 for p in starting_11 if p["position"] == "DEF")
    mid_count = sum(1 for p in starting_11 if p["position"] == "MID")
    fwd_count = sum(1 for p in starting_11 if p["position"] == "FWD")
    formation = f"{def_count}-{mid_count}-{fwd_count}"

    # Captain picks (highest xP from starting 11)
    outfield_starters = [p for p in starting_11 if p["position"] != "GK"]
    outfield_starters.sort(key=lambda x: -x["xp"])
    captain = outfield_starters[0] if outfield_starters else None
    vice_captain = outfield_starters[1] if len(outfield_starters) > 1 else None

    players_formatted = [{
        "id": p["id"],
        "name": p["name"],
        "team": p["team"],
        "position": p["position"],
        "price": p["price"],
        "xp": round(p["xp"], 2)
    } for p in selected]

    starting_formatted = [{
        "id": p["id"],
        "name": p["name"],
        "team": p["team"],
        "position": p["position"],
        "price": p["price"],
        "xp": round(p["xp"], 2)
    } for p in starting_11]

    bench_formatted = [{
        "id": p["id"],
        "name": p["name"],
        "team": p["team"],
        "position": p["position"],
        "price": p["price"],
        "xp": round(p["xp"], 2)
    } for p in bench]

    return OptimalSquad(
        players=players_formatted,
        total_cost=round(total_cost, 1),
        total_xp=round(total_xp, 2),
        formation=formation,
        starting_11=starting_formatted,
        bench=bench_formatted,
        captain={"id": captain["id"], "name": captain["name"]} if captain else None,
        vice_captain={"id": vice_captain["id"], "name": vice_captain["name"]} if vice_captain else None
    )


async def _pick_starting_11(squad: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pick optimal starting 11 from 15-player squad."""
    # Must have: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD, total 11
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in squad:
        by_pos[p["position"]].append(p)

    # Sort each position by xP
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: -x["xp"])

    starting = []
    bench = []

    # Must start: 1 GK, 3 DEF, 2 MID, 1 FWD = 7 players
    starting.append(by_pos["GK"][0])
    bench.append(by_pos["GK"][1]) if len(by_pos["GK"]) > 1 else None

    for p in by_pos["DEF"][:3]:
        starting.append(p)
    for p in by_pos["MID"][:2]:
        starting.append(p)
    for p in by_pos["FWD"][:1]:
        starting.append(p)

    # Remaining 4 spots: pick best from remaining players
    remaining = []
    remaining.extend(by_pos["DEF"][3:])
    remaining.extend(by_pos["MID"][2:])
    remaining.extend(by_pos["FWD"][1:])
    remaining.sort(key=lambda x: -x["xp"])

    for p in remaining[:4]:
        starting.append(p)
    for p in remaining[4:]:
        bench.append(p)

    return starting, bench


async def build_free_hit_team(
    gameweek: int,
    budget: float = 100.0
) -> OptimalSquad | dict:
    """Build optimal team for a single gameweek (Free Hit)."""
    # For Free Hit, we optimize purely for the specific gameweek
    # Import here to avoid circular dependency
    from analytics.fixtures import get_fixture_difficulty

    pool = await _get_player_pool()

    # Adjust xP based on specific GW fixtures
    for p in pool:
        fd = await get_fixture_difficulty(p["team_id"], 1)
        if fd and fd.fixtures:
            gw_fix = next((f for f in fd.fixtures if f["gameweek"] == gameweek), None)
            if gw_fix:
                # Boost for easy fixtures, penalty for hard
                diff_factor = 1.0 + (3 - gw_fix["difficulty"]) * 0.15
                p["xp"] = p["xp"] * diff_factor

    return await build_optimal_squad(budget, "balanced")


async def optimize_starting_11(squad_ids: list[int]) -> dict:
    """Optimize starting 11 from a given squad."""
    if len(squad_ids) < 11:
        return {"error": "Need at least 11 players"}

    # Get player data
    placeholders = ", ".join(f"${i+1}" for i in range(len(squad_ids)))
    players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
    """, *squad_ids)

    # Import xP calculation
    from analytics.expected_points import calculate_expected_points

    squad = []
    for p in players:
        xp_data = await calculate_expected_points(p["id"], num_fixtures=1)
        xp = xp_data.final_xp if xp_data else float(p["form"] or 0)
        squad.append({
            "id": p["id"],
            "name": p["web_name"],
            "position": POSITION_MAP.get(p["element_type"], "MID"),
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "xp": xp,
            "team": p["team"]
        })

    starting, bench = await _pick_starting_11(squad)

    # Determine formation
    def_count = sum(1 for p in starting if p["position"] == "DEF")
    mid_count = sum(1 for p in starting if p["position"] == "MID")
    fwd_count = sum(1 for p in starting if p["position"] == "FWD")

    return {
        "starting_11": starting,
        "bench": bench,
        "formation": f"{def_count}-{mid_count}-{fwd_count}",
        "total_xp": round(sum(p["xp"] for p in starting), 2)
    }


async def suggest_wildcard_team(
    budget: float = 100.0,
    template: bool = False
) -> OptimalSquad | dict:
    """Suggest a wildcard team based on current data."""
    # For wildcard, consider longer horizon
    from analytics.fixtures import get_easiest_fixtures

    easy_fixtures = await get_easiest_fixtures(num_gameweeks=6, limit=10)
    good_fixture_teams = [t["team_id"] for t in easy_fixtures[:6]]

    pool = await _get_player_pool()

    # Boost players from teams with good fixtures
    for p in pool:
        if p["team_id"] in good_fixture_teams:
            p["xp"] *= 1.1

    # If template requested, boost high ownership players
    if template:
        high_ownership = await fetch_all("""
            SELECT id FROM players WHERE selected_by_percent > 20
        """)
        high_own_ids = {p["id"] for p in high_ownership}
        for p in pool:
            if p["id"] in high_own_ids:
                p["xp"] *= 1.05

    return await build_optimal_squad(budget, "balanced")
