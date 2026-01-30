"""Expected Points (xP) model with xG, xA, CS probability, and bonus estimation."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Points per action by position
POINTS_CONFIG = {
    "GK": {"goal": 6, "assist": 3, "cs": 4, "saves_per_point": 3},
    "DEF": {"goal": 6, "assist": 3, "cs": 4},
    "MID": {"goal": 5, "assist": 3, "cs": 1},
    "FWD": {"goal": 4, "assist": 3, "cs": 0},
}


@dataclass
class ExpectedPoints:
    """Expected points projection for a player."""
    player_id: int
    player_name: str
    team: str
    position: str
    price: float

    # Core xP
    base_xp: float
    fixture_adjusted_xp: float
    final_xp: float
    xp_per_million: float

    # Components
    xg_contribution: float
    xa_contribution: float
    cs_probability: float
    bonus_expected: float
    minutes_probability: float

    # Per-gameweek projections
    gw_projections: list[dict]


async def calculate_clean_sheet_probability(
    team_id: int,
    opponent_id: int,
    is_home: bool
) -> float:
    """Calculate probability of clean sheet based on team/opponent strength."""
    team = await fetch_one("SELECT * FROM teams WHERE id = $1", team_id)
    opponent = await fetch_one("SELECT * FROM teams WHERE id = $1", opponent_id)

    if not team or not opponent:
        return 0.3  # Default probability

    # Get defensive strength of team and attacking strength of opponent
    if is_home:
        def_strength = team.get("strength_defence_home", 1100) / 1000
        opp_att_strength = opponent.get("strength_attack_away", 1100) / 1000
    else:
        def_strength = team.get("strength_defence_away", 1100) / 1000
        opp_att_strength = opponent.get("strength_attack_home", 1100) / 1000

    # Calculate CS probability (higher def strength + lower opp attack = higher CS prob)
    # Base rate is ~30%, adjusted by strength differential
    base_cs_prob = 0.30
    strength_factor = (def_strength - opp_att_strength) * 0.15
    cs_prob = max(0.05, min(0.60, base_cs_prob + strength_factor))

    return round(cs_prob, 3)


async def calculate_bonus_expectation(player_id: int) -> float:
    """Estimate expected bonus points based on historical BPS."""
    player = await fetch_one("""
        SELECT p.bps, p.minutes, p.bonus, p.element_type
        FROM players p WHERE p.id = $1
    """, player_id)

    if not player or player["minutes"] == 0:
        return 0.0

    # BPS per 90 * conversion factor to bonus
    bps_per_90 = (player["bps"] / player["minutes"]) * 90 if player["minutes"] > 0 else 0
    bonus_per_90 = (player["bonus"] / player["minutes"]) * 90 if player["minutes"] > 0 else 0

    # Estimate based on historical bonus rate
    # High BPS players (>25 per game) tend to get ~1.5-2 bonus per game
    if bps_per_90 > 30:
        expected_bonus = min(2.5, bonus_per_90 * 1.1)
    elif bps_per_90 > 20:
        expected_bonus = min(1.5, bonus_per_90)
    else:
        expected_bonus = bonus_per_90 * 0.9

    return round(expected_bonus, 2)


async def _get_upcoming_fixtures(player_id: int, num_fixtures: int = 5) -> list[dict]:
    """Get upcoming fixtures for a player's team."""
    player = await fetch_one("SELECT team_id FROM players WHERE id = $1", player_id)
    if not player:
        return []

    team_id = player["team_id"]

    fixtures = await fetch_all("""
        SELECT
            f.id, f.event, f.team_h, f.team_a,
            f.team_h_difficulty, f.team_a_difficulty,
            th.short_name as home_team,
            ta.short_name as away_team,
            CASE WHEN f.team_h = $1 THEN true ELSE false END as is_home
        FROM fixtures f
        JOIN teams th ON f.team_h = th.id
        JOIN teams ta ON f.team_a = ta.id
        WHERE (f.team_h = $1 OR f.team_a = $1)
          AND f.finished = false
        ORDER BY f.event
        LIMIT $2
    """, team_id, num_fixtures)

    return fixtures


async def calculate_expected_points(
    player_id: int,
    num_fixtures: int = 5
) -> ExpectedPoints | None:
    """Calculate expected points for a player over upcoming fixtures."""
    player = await fetch_one("""
        SELECT p.*, t.short_name as team, t.name as team_name
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id = $1
    """, player_id)

    if not player:
        return None

    position = POSITION_MAP.get(player["element_type"], "MID")
    points_config = POINTS_CONFIG.get(position, POINTS_CONFIG["MID"])

    # Calculate per-90 rates
    minutes = player.get("minutes", 0) or 1
    xg = float(player.get("expected_goals", 0) or 0)
    xa = float(player.get("expected_assists", 0) or 0)

    xg_per_90 = (xg / minutes) * 90 if minutes > 0 else 0
    xa_per_90 = (xa / minutes) * 90 if minutes > 0 else 0

    # Expected goal/assist contribution
    xg_contribution = xg_per_90 * points_config["goal"]
    xa_contribution = xa_per_90 * points_config["assist"]

    # Minutes probability (based on recent starts)
    starts = player.get("starts", 0) or 0
    games_played = (minutes / 90) if minutes > 0 else 0
    minutes_prob = min(1.0, starts / max(1, games_played) if games_played > 0 else 0.8)

    # Get upcoming fixtures and calculate per-GW xP
    fixtures = await _get_upcoming_fixtures(player_id, num_fixtures)
    gw_projections = []
    total_fixture_xp = 0

    for fix in fixtures:
        is_home = fix["is_home"]
        opponent_id = fix["team_a"] if is_home else fix["team_h"]
        difficulty = fix["team_h_difficulty"] if not is_home else fix["team_a_difficulty"]

        # Clean sheet probability for defenders/GKs
        cs_prob = 0.0
        cs_points = 0.0
        if position in ["GK", "DEF"]:
            cs_prob = await calculate_clean_sheet_probability(
                player["team_id"], opponent_id, is_home
            )
            cs_points = cs_prob * points_config["cs"]

        # Difficulty adjustment (2=easy, 5=hard)
        # Scale: 1.2x for difficulty 2, 0.8x for difficulty 5
        diff_factor = 1.0 + (3 - difficulty) * 0.1

        # Home advantage
        home_factor = 1.1 if is_home else 0.95

        # Calculate GW xP
        base_gw_xp = (
            2.0 +  # Appearance points
            xg_contribution +
            xa_contribution +
            cs_points
        )

        bonus_exp = await calculate_bonus_expectation(player_id)
        gw_xp = (base_gw_xp + bonus_exp) * diff_factor * home_factor * minutes_prob

        gw_projections.append({
            "gameweek": fix["event"],
            "opponent": fix["away_team"] if is_home else fix["home_team"],
            "is_home": is_home,
            "difficulty": difficulty,
            "xp": round(gw_xp, 2),
            "cs_probability": round(cs_prob, 2) if position in ["GK", "DEF"] else None
        })

        total_fixture_xp += gw_xp

    # Aggregate calculations
    avg_cs_prob = 0.0
    if position in ["GK", "DEF"] and fixtures:
        cs_probs = [p["cs_probability"] for p in gw_projections if p["cs_probability"]]
        avg_cs_prob = sum(cs_probs) / len(cs_probs) if cs_probs else 0.3

    base_xp = (
        2.0 +  # Appearance
        xg_contribution +
        xa_contribution +
        (avg_cs_prob * points_config.get("cs", 0))
    )

    bonus_expected = await calculate_bonus_expectation(player_id)
    fixture_adjusted_xp = total_fixture_xp / max(1, len(fixtures)) if fixtures else base_xp
    final_xp = fixture_adjusted_xp

    price = (player.get("now_cost", 0) or 0) / 10
    xp_per_million = final_xp / price if price > 0 else 0

    return ExpectedPoints(
        player_id=player_id,
        player_name=player["web_name"],
        team=player["team"],
        position=position,
        price=price,
        base_xp=round(base_xp, 2),
        fixture_adjusted_xp=round(fixture_adjusted_xp, 2),
        final_xp=round(final_xp, 2),
        xp_per_million=round(xp_per_million, 3),
        xg_contribution=round(xg_contribution, 2),
        xa_contribution=round(xa_contribution, 2),
        cs_probability=round(avg_cs_prob, 2),
        bonus_expected=round(bonus_expected, 2),
        minutes_probability=round(minutes_prob, 2),
        gw_projections=gw_projections
    )


async def get_top_xp_players(
    position: str | None = None,
    limit: int = 20,
    min_minutes: int = 90
) -> list[dict]:
    """Get players ranked by expected points."""
    conditions = ["p.status = 'a'", f"p.minutes >= {min_minutes}"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    params.append(limit * 2)  # Get more to filter after xP calculation

    query = f"""
        SELECT p.id, p.web_name, p.team_id, p.element_type, p.now_cost,
               p.form, p.total_points, p.expected_goals, p.expected_assists,
               p.minutes, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.form DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    # Calculate xP for each player
    results = []
    for p in players:
        xp = await calculate_expected_points(p["id"], num_fixtures=5)
        if xp:
            results.append({
                "id": p["id"],
                "name": p["web_name"],
                "team": p["team"],
                "position": POSITION_MAP.get(p["element_type"], "?"),
                "price": p["now_cost"] / 10 if p["now_cost"] else 0,
                "form": float(p["form"]) if p["form"] else 0,
                "xp_next_5": xp.final_xp,
                "xp_per_million": xp.xp_per_million,
                "gw_projections": xp.gw_projections[:3]  # Next 3 GWs
            })

    # Sort by xP and return top
    results.sort(key=lambda x: x["xp_next_5"], reverse=True)
    return results[:limit]


async def get_value_picks(
    position: str | None = None,
    max_price: float | None = None,
    limit: int = 15
) -> list[dict]:
    """Get best value picks (highest xP per million)."""
    conditions = ["p.status = 'a'", "p.minutes >= 90"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    if max_price:
        conditions.append(f"p.now_cost <= ${idx}")
        params.append(int(max_price * 10))
        idx += 1

    params.append(limit * 3)

    query = f"""
        SELECT p.id, p.web_name, p.team_id, p.element_type, p.now_cost,
               p.form, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY (p.total_points::float / NULLIF(p.now_cost, 0)) DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    results = []
    for p in players:
        xp = await calculate_expected_points(p["id"], num_fixtures=5)
        if xp and xp.xp_per_million > 0:
            results.append({
                "id": p["id"],
                "name": p["web_name"],
                "team": p["team"],
                "position": POSITION_MAP.get(p["element_type"], "?"),
                "price": p["now_cost"] / 10 if p["now_cost"] else 0,
                "xp_next_5": xp.final_xp,
                "xp_per_million": xp.xp_per_million,
                "form": float(p["form"]) if p["form"] else 0
            })

    results.sort(key=lambda x: x["xp_per_million"], reverse=True)
    return results[:limit]
