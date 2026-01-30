"""Minutes prediction and rotation risk analysis."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class MinutesPrediction:
    """Minutes prediction for a player."""
    player_id: int
    player_name: str
    team: str
    position: str

    # Stats
    total_minutes: int
    total_starts: int
    games_available: int  # Games where status was 'a'
    minutes_per_game: float
    start_rate: float  # % of games started

    # Prediction
    predicted_minutes: float  # Expected minutes next GW
    nailed_score: float  # 0-100, higher = more nailed
    rotation_risk: str  # "low", "medium", "high"
    risk_factors: list[str]


async def predict_minutes(player_id: int) -> MinutesPrediction | None:
    """Predict minutes and rotation risk for a player."""
    player = await fetch_one("""
        SELECT p.*, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id = $1
    """, player_id)

    if not player:
        return None

    total_minutes = player.get("minutes", 0) or 0
    starts = player.get("starts", 0) or 0
    status = player.get("status", "a")
    chance_next = player.get("chance_of_playing_next_round")

    # Get gameweek count from fixtures
    current_gw = await fetch_one("""
        SELECT id FROM events WHERE is_current = true LIMIT 1
    """)
    gw = current_gw["id"] if current_gw else 10

    # Estimate games available (approximation)
    games_available = max(1, gw - 1)

    minutes_per_game = total_minutes / games_available if games_available > 0 else 0
    start_rate = (starts / games_available * 100) if games_available > 0 else 0

    # Calculate nailed score
    risk_factors = []

    # Base nailed score from minutes
    if minutes_per_game >= 85:
        nailed_score = 95
    elif minutes_per_game >= 75:
        nailed_score = 85
    elif minutes_per_game >= 60:
        nailed_score = 70
    elif minutes_per_game >= 45:
        nailed_score = 50
    else:
        nailed_score = 30

    # Adjust for start rate
    if start_rate >= 90:
        nailed_score += 5
    elif start_rate < 70:
        nailed_score -= 10
        risk_factors.append("Low start rate")

    # Adjust for status/availability
    if status != "a":
        nailed_score -= 30
        risk_factors.append(f"Flagged status: {status}")

    if chance_next is not None and chance_next < 100:
        penalty = (100 - chance_next) / 2
        nailed_score -= penalty
        risk_factors.append(f"{chance_next}% chance of playing")

    # Check for recent low minutes (rotation indicator)
    recent_history = await fetch_all("""
        SELECT minutes FROM player_history
        WHERE player_id = $1
        ORDER BY event DESC
        LIMIT 5
    """, player_id)

    if recent_history:
        recent_mins = [h["minutes"] or 0 for h in recent_history]
        if any(m < 60 for m in recent_mins[-3:]):
            nailed_score -= 10
            risk_factors.append("Recent sub appearances")

        if recent_mins and recent_mins[0] == 0:
            nailed_score -= 15
            risk_factors.append("Benched last game")

    nailed_score = max(0, min(100, nailed_score))

    # Determine rotation risk
    if nailed_score >= 80:
        rotation_risk = "low"
    elif nailed_score >= 50:
        rotation_risk = "medium"
    else:
        rotation_risk = "high"

    # Predict next GW minutes
    if status != "a":
        predicted_minutes = 0
    elif chance_next is not None:
        predicted_minutes = minutes_per_game * (chance_next / 100)
    else:
        predicted_minutes = minutes_per_game * (nailed_score / 100)

    return MinutesPrediction(
        player_id=player_id,
        player_name=player["web_name"],
        team=player["team"],
        position=POSITION_MAP.get(player["element_type"], "?"),
        total_minutes=total_minutes,
        total_starts=starts,
        games_available=games_available,
        minutes_per_game=round(minutes_per_game, 1),
        start_rate=round(start_rate, 1),
        predicted_minutes=round(predicted_minutes, 1),
        nailed_score=round(nailed_score, 1),
        rotation_risk=rotation_risk,
        risk_factors=risk_factors
    )


async def get_rotation_risks(
    position: str | None = None,
    limit: int = 15
) -> list[dict]:
    """Get players with high rotation risk."""
    conditions = ["p.status = 'a'", "p.minutes > 90", "p.now_cost >= 45"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    params.append(limit * 3)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.minutes, p.starts, p.selected_by_percent, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.selected_by_percent DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    results = []
    for p in players:
        pred = await predict_minutes(p["id"])
        if pred and pred.rotation_risk in ["medium", "high"]:
            results.append({
                "id": p["id"],
                "name": p["web_name"],
                "team": p["team"],
                "position": POSITION_MAP.get(p["element_type"], "?"),
                "price": p["now_cost"] / 10 if p["now_cost"] else 0,
                "ownership": float(p["selected_by_percent"] or 0),
                "minutes_per_game": pred.minutes_per_game,
                "start_rate": pred.start_rate,
                "nailed_score": pred.nailed_score,
                "rotation_risk": pred.rotation_risk,
                "risk_factors": pred.risk_factors
            })

    # Sort by highest ownership (most impactful rotation risks)
    results.sort(key=lambda x: x["ownership"], reverse=True)
    return results[:limit]


async def get_nailed_players(
    position: str | None = None,
    min_nailed_score: float = 80,
    limit: int = 20
) -> list[dict]:
    """Get nailed-on players (low rotation risk)."""
    conditions = ["p.status = 'a'", "p.minutes > 450"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    params.append(limit * 2)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.minutes, p.starts, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.minutes DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    results = []
    for p in players:
        pred = await predict_minutes(p["id"])
        if pred and pred.nailed_score >= min_nailed_score:
            results.append({
                "id": p["id"],
                "name": p["web_name"],
                "team": p["team"],
                "position": POSITION_MAP.get(p["element_type"], "?"),
                "price": p["now_cost"] / 10 if p["now_cost"] else 0,
                "form": float(p["form"] or 0),
                "minutes_per_game": pred.minutes_per_game,
                "start_rate": pred.start_rate,
                "nailed_score": pred.nailed_score
            })

    results.sort(key=lambda x: x["nailed_score"], reverse=True)
    return results[:limit]


async def get_benching_risks(player_ids: list[int]) -> list[dict]:
    """Analyze benching risk for a list of players (e.g., squad)."""
    if not player_ids:
        return []

    results = []
    for pid in player_ids:
        pred = await predict_minutes(pid)
        if pred:
            results.append({
                "id": pred.player_id,
                "name": pred.player_name,
                "team": pred.team,
                "position": pred.position,
                "predicted_minutes": pred.predicted_minutes,
                "nailed_score": pred.nailed_score,
                "rotation_risk": pred.rotation_risk,
                "risk_factors": pred.risk_factors
            })

    # Sort by rotation risk (high risk first)
    risk_order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda x: risk_order.get(x["rotation_risk"], 3))

    return results
