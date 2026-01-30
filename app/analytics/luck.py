"""Luck analysis: xG over/underperformers and regression candidates."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class LuckAnalysis:
    """Luck analysis for a player."""
    player_id: int
    player_name: str
    team: str
    position: str

    # Stats
    goals: int
    expected_goals: float
    assists: int
    expected_assists: float

    # xG analysis
    goals_minus_xg: float  # Positive = overperforming
    assists_minus_xa: float
    total_overperformance: float

    # Assessment
    luck_rating: str  # "lucky", "neutral", "unlucky"
    regression_risk: str  # "high", "medium", "low", "none"
    upside_potential: str  # "high", "medium", "low", "none"
    summary: str


async def analyze_luck(player_id: int) -> LuckAnalysis | None:
    """Analyze if a player is over/underperforming their xG/xA."""
    player = await fetch_one("""
        SELECT p.*, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id = $1
    """, player_id)

    if not player:
        return None

    goals = player.get("goals_scored", 0) or 0
    xg = float(player.get("expected_goals", 0) or 0)
    assists = player.get("assists", 0) or 0
    xa = float(player.get("expected_assists", 0) or 0)

    goals_diff = goals - xg
    assists_diff = assists - xa
    total_diff = goals_diff + assists_diff

    # Determine luck rating
    if total_diff > 3:
        luck_rating = "very lucky"
        regression_risk = "high"
        upside_potential = "none"
        summary = f"Significantly overperforming xG/xA by {total_diff:.1f}. High regression risk."
    elif total_diff > 1.5:
        luck_rating = "lucky"
        regression_risk = "medium"
        upside_potential = "low"
        summary = f"Overperforming xG/xA by {total_diff:.1f}. Some regression expected."
    elif total_diff > -1.5:
        luck_rating = "neutral"
        regression_risk = "low"
        upside_potential = "low"
        summary = "Performing roughly in line with underlying numbers."
    elif total_diff > -3:
        luck_rating = "unlucky"
        regression_risk = "none"
        upside_potential = "medium"
        summary = f"Underperforming xG/xA by {abs(total_diff):.1f}. Positive regression expected."
    else:
        luck_rating = "very unlucky"
        regression_risk = "none"
        upside_potential = "high"
        summary = f"Significantly underperforming xG/xA by {abs(total_diff):.1f}. Strong upside potential."

    return LuckAnalysis(
        player_id=player_id,
        player_name=player["web_name"],
        team=player["team"],
        position=POSITION_MAP.get(player["element_type"], "?"),
        goals=goals,
        expected_goals=round(xg, 2),
        assists=assists,
        expected_assists=round(xa, 2),
        goals_minus_xg=round(goals_diff, 2),
        assists_minus_xa=round(assists_diff, 2),
        total_overperformance=round(total_diff, 2),
        luck_rating=luck_rating,
        regression_risk=regression_risk,
        upside_potential=upside_potential,
        summary=summary
    )


async def get_overperformers(
    position: str | None = None,
    limit: int = 15,
    min_minutes: int = 450
) -> list[dict]:
    """Get players overperforming their xG/xA (regression candidates)."""
    conditions = ["p.status = 'a'", f"p.minutes >= {min_minutes}"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        if position.upper() in pos_map:
            conditions.append(f"p.element_type = ${idx}")
            params.append(pos_map[position.upper()])
            idx += 1

    params.append(limit)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost,
               p.goals_scored, p.assists, p.expected_goals, p.expected_assists,
               p.form, p.minutes, t.short_name as team,
               (p.goals_scored - COALESCE(p.expected_goals, 0) +
                p.assists - COALESCE(p.expected_assists, 0)) as overperformance
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
          AND (p.goals_scored - COALESCE(p.expected_goals, 0) +
               p.assists - COALESCE(p.expected_assists, 0)) > 1
        ORDER BY overperformance DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    results = []
    for p in players:
        overperf = float(p["overperformance"] or 0)
        if overperf > 4:
            risk = "high"
        elif overperf > 2:
            risk = "medium"
        else:
            risk = "low"

        results.append({
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "goals": p["goals_scored"] or 0,
            "xG": round(float(p["expected_goals"] or 0), 2),
            "assists": p["assists"] or 0,
            "xA": round(float(p["expected_assists"] or 0), 2),
            "overperformance": round(overperf, 2),
            "regression_risk": risk,
            "form": float(p["form"] or 0)
        })

    return results


async def get_underperformers(
    position: str | None = None,
    limit: int = 15,
    min_minutes: int = 450
) -> list[dict]:
    """Get players underperforming their xG/xA (upside candidates)."""
    conditions = ["p.status = 'a'", f"p.minutes >= {min_minutes}"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        if position.upper() in pos_map:
            conditions.append(f"p.element_type = ${idx}")
            params.append(pos_map[position.upper()])
            idx += 1

    params.append(limit)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost,
               p.goals_scored, p.assists, p.expected_goals, p.expected_assists,
               p.form, p.minutes, t.short_name as team,
               (p.goals_scored - COALESCE(p.expected_goals, 0) +
                p.assists - COALESCE(p.expected_assists, 0)) as overperformance
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
          AND (p.goals_scored - COALESCE(p.expected_goals, 0) +
               p.assists - COALESCE(p.expected_assists, 0)) < -1
        ORDER BY overperformance ASC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    results = []
    for p in players:
        underperf = abs(float(p["overperformance"] or 0))
        if underperf > 4:
            upside = "high"
        elif underperf > 2:
            upside = "medium"
        else:
            upside = "low"

        results.append({
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "goals": p["goals_scored"] or 0,
            "xG": round(float(p["expected_goals"] or 0), 2),
            "assists": p["assists"] or 0,
            "xA": round(float(p["expected_assists"] or 0), 2),
            "underperformance": round(underperf, 2),
            "upside_potential": upside,
            "form": float(p["form"] or 0)
        })

    return results


async def get_xg_analysis_summary(position: str | None = None) -> dict:
    """Get summary of xG analysis across positions."""
    overperformers = await get_overperformers(position, limit=10)
    underperformers = await get_underperformers(position, limit=10)

    return {
        "overperformers": {
            "count": len(overperformers),
            "players": overperformers,
            "description": "Players scoring/assisting above their xG/xA - may regress"
        },
        "underperformers": {
            "count": len(underperformers),
            "players": underperformers,
            "description": "Players scoring/assisting below their xG/xA - may improve"
        }
    }
