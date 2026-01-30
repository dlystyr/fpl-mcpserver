"""Captain recommendation engine."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class CaptainPick:
    """A captain recommendation."""
    player_id: int
    player_name: str
    team: str
    position: str

    # Scores
    xp_next_gw: float
    fixture_difficulty: int
    form: float
    ownership: float

    # Historical
    vs_opponent_ppg: float | None  # Historical PPG vs this opponent

    # Final
    captain_score: float
    rank: int
    reasons: list[str]


async def get_captaincy_picks(
    squad_ids: list[int],
    gameweek: int | None = None,
    limit: int = 5
) -> list[CaptainPick]:
    """Get captain recommendations for a squad."""
    from analytics.expected_points import calculate_expected_points
    from analytics.fixtures import get_fixture_difficulty

    if not squad_ids:
        return []

    placeholders = ", ".join(f"${i+1}" for i in range(len(squad_ids)))
    players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.team_id, p.element_type, p.now_cost,
               p.form, p.total_points, p.selected_by_percent, p.status,
               t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
    """, *squad_ids)

    # Get current gameweek if not specified
    if not gameweek:
        current = await fetch_one("SELECT id FROM events WHERE is_next = true")
        gameweek = current["id"] if current else 1

    results = []
    for p in players:
        if p["status"] != "a":
            continue  # Skip unavailable players

        # Calculate xP for next GW
        xp_data = await calculate_expected_points(p["id"], num_fixtures=1)
        xp = xp_data.final_xp if xp_data else float(p["form"] or 0)

        # Get fixture difficulty
        fd = await get_fixture_difficulty(p["team_id"], 1)
        fix_diff = 3
        is_home = False
        opponent = "?"

        if fd and fd.fixtures:
            gw_fix = fd.fixtures[0]
            fix_diff = gw_fix["difficulty"]
            is_home = gw_fix["is_home"]
            opponent = gw_fix["opponent"]

        # Get historical performance vs opponent
        vs_opponent = await _get_history_vs_opponent(p["id"], opponent)

        form = float(p["form"] or 0)
        ownership = float(p["selected_by_percent"] or 0)

        # Calculate captain score
        # Higher for: high xP, low difficulty, good form, good vs opponent
        score = (
            xp * 15 +                              # xP is primary factor
            (6 - fix_diff) * 3 +                   # Fixture ease
            form * 2 +                             # Current form
            (vs_opponent or 0) * 2 +               # Historical vs opponent
            (2 if is_home else 0)                  # Home advantage
        )

        # Generate reasons
        reasons = []
        if fix_diff <= 2:
            reasons.append(f"Easy fixture ({opponent} {'H' if is_home else 'A'})")
        if form >= 6:
            reasons.append(f"Excellent form ({form})")
        if vs_opponent and vs_opponent > 5:
            reasons.append(f"Good record vs {opponent} ({vs_opponent:.1f} PPG)")
        if is_home:
            reasons.append("Home advantage")
        if ownership > 30:
            reasons.append(f"High ownership ({ownership:.1f}%) - safe pick")

        results.append({
            "player_id": p["id"],
            "player_name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "xp_next_gw": round(xp, 2),
            "fixture_difficulty": fix_diff,
            "opponent": opponent,
            "is_home": is_home,
            "form": form,
            "ownership": round(ownership, 2),
            "vs_opponent_ppg": round(vs_opponent, 2) if vs_opponent else None,
            "captain_score": round(score, 2),
            "reasons": reasons
        })

    # Sort by captain score
    results.sort(key=lambda x: x["captain_score"], reverse=True)

    # Add ranks
    captain_picks = []
    for i, r in enumerate(results[:limit]):
        captain_picks.append(CaptainPick(
            player_id=r["player_id"],
            player_name=r["player_name"],
            team=r["team"],
            position=r["position"],
            xp_next_gw=r["xp_next_gw"],
            fixture_difficulty=r["fixture_difficulty"],
            form=r["form"],
            ownership=r["ownership"],
            vs_opponent_ppg=r["vs_opponent_ppg"],
            captain_score=r["captain_score"],
            rank=i + 1,
            reasons=r["reasons"]
        ))

    return captain_picks


async def _get_history_vs_opponent(player_id: int, opponent_short: str) -> float | None:
    """Get player's historical PPG against a specific opponent."""
    # Get opponent team ID
    opponent = await fetch_one("SELECT id FROM teams WHERE short_name = $1", opponent_short)
    if not opponent:
        return None

    # Get historical games vs this opponent
    history = await fetch_all("""
        SELECT ph.total_points
        FROM player_history ph
        WHERE ph.player_id = $1 AND ph.opponent_team = $2
    """, player_id, opponent["id"])

    if not history:
        return None

    points = [h["total_points"] or 0 for h in history]
    return sum(points) / len(points) if points else None


async def analyze_captain_options(
    squad_ids: list[int],
    gameweek: int | None = None
) -> dict:
    """Get detailed captain analysis with comparisons."""
    picks = await get_captaincy_picks(squad_ids, gameweek, limit=5)

    if not picks:
        return {"error": "No valid captain options"}

    # Compare top 2
    top_pick = picks[0]
    differential_pick = None

    # Find a differential option (lower ownership)
    for pick in picks[1:]:
        if pick.ownership < top_pick.ownership * 0.5:  # Less than half the ownership
            differential_pick = pick
            break

    analysis = {
        "recommended_captain": {
            "id": top_pick.player_id,
            "name": top_pick.player_name,
            "team": top_pick.team,
            "xp": top_pick.xp_next_gw,
            "reasons": top_pick.reasons
        },
        "recommended_vice": {
            "id": picks[1].player_id,
            "name": picks[1].player_name,
            "team": picks[1].team,
            "xp": picks[1].xp_next_gw,
            "reasons": picks[1].reasons
        } if len(picks) > 1 else None,
        "differential_option": {
            "id": differential_pick.player_id,
            "name": differential_pick.player_name,
            "ownership": differential_pick.ownership,
            "xp": differential_pick.xp_next_gw,
            "risk_reward": "High risk, high reward if they haul"
        } if differential_pick else None,
        "all_options": [{
            "rank": p.rank,
            "name": p.player_name,
            "team": p.team,
            "xp": p.xp_next_gw,
            "fixture_difficulty": p.fixture_difficulty,
            "ownership": p.ownership,
            "captain_score": p.captain_score
        } for p in picks]
    }

    return analysis


async def get_differential_captain(
    squad_ids: list[int],
    max_ownership: float = 15.0,
    gameweek: int | None = None
) -> CaptainPick | None:
    """Get best differential captain option (low ownership)."""
    picks = await get_captaincy_picks(squad_ids, gameweek, limit=10)

    for pick in picks:
        if pick.ownership <= max_ownership:
            return pick

    return None


async def compare_captain_scenarios(
    player_ids: list[int],
    gameweek: int | None = None
) -> list[dict]:
    """Compare multiple captain scenarios."""
    from analytics.expected_points import calculate_expected_points
    from analytics.fixtures import get_fixture_difficulty

    if not player_ids:
        return []

    placeholders = ", ".join(f"${i+1}" for i in range(len(player_ids)))
    players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.team_id, p.element_type, p.form,
               p.selected_by_percent, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
    """, *player_ids)

    scenarios = []
    for p in players:
        xp_data = await calculate_expected_points(p["id"], num_fixtures=1)
        xp = xp_data.final_xp if xp_data else float(p["form"] or 0)

        fd = await get_fixture_difficulty(p["team_id"], 1)
        fixture_info = fd.fixtures[0] if fd and fd.fixtures else None

        ownership = float(p["selected_by_percent"] or 0)

        # Calculate effective ownership swing potential
        # If player hauls and you captain while others don't, big gain
        non_captainers = 100 - ownership  # % who don't own
        partial_captainers = ownership * 0.3  # Estimate 30% of owners captain

        swing_potential = (non_captainers + partial_captainers * 0.5) * xp * 2 / 100

        scenarios.append({
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "xp_single": round(xp, 2),
            "xp_captained": round(xp * 2, 2),
            "ownership": round(ownership, 2),
            "fixture": f"{fixture_info['opponent']} ({'H' if fixture_info['is_home'] else 'A'})" if fixture_info else "?",
            "difficulty": fixture_info["difficulty"] if fixture_info else 3,
            "swing_potential": round(swing_potential, 2),
            "risk_level": "low" if ownership > 25 else "medium" if ownership > 10 else "high"
        })

    # Sort by xp_captained
    scenarios.sort(key=lambda x: x["xp_captained"], reverse=True)
    return scenarios
