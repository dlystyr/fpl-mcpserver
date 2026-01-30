"""Venue analysis: home/away performance splits."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class VenueSplits:
    """Home/away performance splits for a player."""
    player_id: int
    player_name: str
    team: str
    position: str

    # Home stats
    home_games: int
    home_points: int
    home_goals: int
    home_assists: int
    home_ppg: float

    # Away stats
    away_games: int
    away_points: int
    away_goals: int
    away_assists: int
    away_ppg: float

    # Comparison
    home_advantage: float  # Home PPG - Away PPG
    better_venue: str  # "home", "away", "neutral"


async def get_home_away_splits(player_id: int) -> VenueSplits | None:
    """Get home/away performance splits for a player."""
    player = await fetch_one("""
        SELECT p.*, t.short_name as team FROM players p
        JOIN teams t ON p.team_id = t.id WHERE p.id = $1
    """, player_id)

    if not player:
        return None

    # Get player history with venue info
    history = await fetch_all("""
        SELECT
            ph.*,
            CASE WHEN f.team_h = p.team_id THEN true ELSE false END as was_home
        FROM player_history ph
        JOIN players p ON ph.player_id = p.id
        JOIN fixtures f ON ph.fixture_id = f.id
        WHERE ph.player_id = $1
    """, player_id)

    home_games = 0
    home_points = 0
    home_goals = 0
    home_assists = 0

    away_games = 0
    away_points = 0
    away_goals = 0
    away_assists = 0

    for h in history:
        if h["was_home"]:
            home_games += 1
            home_points += h.get("total_points", 0) or 0
            home_goals += h.get("goals_scored", 0) or 0
            home_assists += h.get("assists", 0) or 0
        else:
            away_games += 1
            away_points += h.get("total_points", 0) or 0
            away_goals += h.get("goals_scored", 0) or 0
            away_assists += h.get("assists", 0) or 0

    home_ppg = home_points / home_games if home_games > 0 else 0
    away_ppg = away_points / away_games if away_games > 0 else 0
    home_advantage = home_ppg - away_ppg

    if home_advantage > 1:
        better_venue = "home"
    elif home_advantage < -1:
        better_venue = "away"
    else:
        better_venue = "neutral"

    return VenueSplits(
        player_id=player_id,
        player_name=player["web_name"],
        team=player["team"],
        position=POSITION_MAP.get(player["element_type"], "?"),
        home_games=home_games,
        home_points=home_points,
        home_goals=home_goals,
        home_assists=home_assists,
        home_ppg=round(home_ppg, 2),
        away_games=away_games,
        away_points=away_points,
        away_goals=away_goals,
        away_assists=away_assists,
        away_ppg=round(away_ppg, 2),
        home_advantage=round(home_advantage, 2),
        better_venue=better_venue
    )


async def get_best_home_performers(
    position: str | None = None,
    min_home_games: int = 3,
    limit: int = 15
) -> list[dict]:
    """Get players who perform best at home."""
    conditions = ["p.status = 'a'", "p.minutes > 270"]
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
               t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.form DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    results = []
    for p in players:
        splits = await get_home_away_splits(p["id"])
        if splits and splits.home_games >= min_home_games and splits.home_advantage > 0.5:
            results.append({
                "id": p["id"],
                "name": p["web_name"],
                "team": p["team"],
                "position": POSITION_MAP.get(p["element_type"], "?"),
                "price": p["now_cost"] / 10 if p["now_cost"] else 0,
                "home_ppg": splits.home_ppg,
                "away_ppg": splits.away_ppg,
                "home_advantage": splits.home_advantage,
                "home_games": splits.home_games,
                "form": float(p["form"] or 0)
            })

    results.sort(key=lambda x: x["home_advantage"], reverse=True)
    return results[:limit]


async def get_best_away_performers(
    position: str | None = None,
    min_away_games: int = 3,
    limit: int = 15
) -> list[dict]:
    """Get players who perform best away from home."""
    conditions = ["p.status = 'a'", "p.minutes > 270"]
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
               t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.form DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    results = []
    for p in players:
        splits = await get_home_away_splits(p["id"])
        if splits and splits.away_games >= min_away_games:
            # Include players who are good away or venue-neutral
            if splits.away_ppg >= 4 or splits.better_venue in ["away", "neutral"]:
                results.append({
                    "id": p["id"],
                    "name": p["web_name"],
                    "team": p["team"],
                    "position": POSITION_MAP.get(p["element_type"], "?"),
                    "price": p["now_cost"] / 10 if p["now_cost"] else 0,
                    "away_ppg": splits.away_ppg,
                    "home_ppg": splits.home_ppg,
                    "home_advantage": splits.home_advantage,  # Will be low or negative
                    "away_games": splits.away_games,
                    "form": float(p["form"] or 0)
                })

    results.sort(key=lambda x: x["away_ppg"], reverse=True)
    return results[:limit]


async def get_venue_adjusted_picks(
    player_ids: list[int],
    is_home: bool
) -> list[dict]:
    """Get venue-adjusted analysis for a list of players."""
    if not player_ids:
        return []

    results = []
    for pid in player_ids:
        splits = await get_home_away_splits(pid)
        if splits:
            relevant_ppg = splits.home_ppg if is_home else splits.away_ppg
            venue = "home" if is_home else "away"
            games = splits.home_games if is_home else splits.away_games

            results.append({
                "id": splits.player_id,
                "name": splits.player_name,
                "team": splits.team,
                "position": splits.position,
                "venue": venue,
                "expected_ppg": relevant_ppg,
                "venue_games": games,
                "home_advantage": splits.home_advantage,
                "better_venue": splits.better_venue,
                "venue_match": (is_home and splits.better_venue == "home") or
                              (not is_home and splits.better_venue == "away")
            })

    # Sort by expected PPG for this venue
    results.sort(key=lambda x: x["expected_ppg"], reverse=True)
    return results


async def get_team_home_away_form(team_id: int) -> dict:
    """Get team's home vs away form."""
    team = await fetch_one("SELECT * FROM teams WHERE id = $1", team_id)
    if not team:
        return {}

    # Get finished fixtures
    home_fixtures = await fetch_all("""
        SELECT team_h_score, team_a_score FROM fixtures
        WHERE team_h = $1 AND finished = true
        ORDER BY event DESC LIMIT 5
    """, team_id)

    away_fixtures = await fetch_all("""
        SELECT team_h_score, team_a_score FROM fixtures
        WHERE team_a = $1 AND finished = true
        ORDER BY event DESC LIMIT 5
    """, team_id)

    def calc_form(fixtures, is_home):
        results = []
        points = 0
        gf = 0
        ga = 0
        for f in fixtures:
            if is_home:
                goals_for = f["team_h_score"] or 0
                goals_against = f["team_a_score"] or 0
            else:
                goals_for = f["team_a_score"] or 0
                goals_against = f["team_h_score"] or 0

            gf += goals_for
            ga += goals_against

            if goals_for > goals_against:
                results.append("W")
                points += 3
            elif goals_for < goals_against:
                results.append("L")
            else:
                results.append("D")
                points += 1

        return {
            "form": "".join(results),
            "points": points,
            "goals_for": gf,
            "goals_against": ga,
            "clean_sheets": sum(1 for f in fixtures if (f["team_a_score"] if is_home else f["team_h_score"]) == 0)
        }

    home_form = calc_form(home_fixtures, True)
    away_form = calc_form(away_fixtures, False)

    return {
        "team_id": team_id,
        "team_name": team["name"],
        "home": home_form,
        "away": away_form,
        "strength": {
            "attack_home": team.get("strength_attack_home"),
            "attack_away": team.get("strength_attack_away"),
            "defence_home": team.get("strength_defence_home"),
            "defence_away": team.get("strength_defence_away")
        }
    }
