"""Hit analysis: evaluate -4/-8 hits for transfers."""

from dataclasses import dataclass
from database import fetch_all, fetch_one
from .expected_points import calculate_expected_points

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class HitAnalysis:
    """Analysis of whether a hit is worth taking."""
    player_out_id: int
    player_out_name: str
    player_in_id: int
    player_in_name: str

    # xP comparison
    xp_out: float  # Expected points from player out
    xp_in: float   # Expected points from player in
    xp_differential: float  # xp_in - xp_out

    # Hit analysis
    hit_cost: int  # -4, -8, etc.
    net_gain: float  # xp_differential - hit_cost
    breakeven_weeks: float  # How many weeks until hit pays off

    # Recommendation
    recommendation: str  # "take_hit", "wait", "avoid"
    confidence: str
    reasoning: str


@dataclass
class MultiHitAnalysis:
    """Analysis of multiple hits."""
    moves: list[HitAnalysis]
    total_hit_cost: int
    total_xp_gain: float
    total_net_gain: float
    recommendation: str
    reasoning: str


async def evaluate_hit(
    player_out_id: int,
    player_in_id: int,
    horizon: int = 5,
    hit_cost: int = 4
) -> HitAnalysis | None:
    """Evaluate if a -4 hit transfer is worth it."""
    # Get player info
    player_out = await fetch_one("""
        SELECT p.*, t.short_name as team FROM players p
        JOIN teams t ON p.team_id = t.id WHERE p.id = $1
    """, player_out_id)

    player_in = await fetch_one("""
        SELECT p.*, t.short_name as team FROM players p
        JOIN teams t ON p.team_id = t.id WHERE p.id = $1
    """, player_in_id)

    if not player_out or not player_in:
        return None

    # Calculate expected points over horizon
    xp_out_data = await calculate_expected_points(player_out_id, horizon)
    xp_in_data = await calculate_expected_points(player_in_id, horizon)

    xp_out = xp_out_data.final_xp * horizon if xp_out_data else 0
    xp_in = xp_in_data.final_xp * horizon if xp_in_data else 0

    xp_differential = xp_in - xp_out
    net_gain = xp_differential - hit_cost

    # Calculate breakeven
    per_week_gain = (xp_in_data.final_xp - xp_out_data.final_xp) if xp_in_data and xp_out_data else 0
    breakeven_weeks = hit_cost / per_week_gain if per_week_gain > 0 else float('inf')

    # Determine recommendation
    if net_gain > 5:
        recommendation = "take_hit"
        confidence = "high"
        reasoning = f"Strong positive gain of {net_gain:.1f} pts over {horizon} weeks"
    elif net_gain > 2:
        recommendation = "take_hit"
        confidence = "medium"
        reasoning = f"Moderate gain of {net_gain:.1f} pts, hit pays off in {breakeven_weeks:.1f} weeks"
    elif net_gain > 0:
        recommendation = "wait"
        confidence = "low"
        reasoning = f"Marginal gain of {net_gain:.1f} pts. Consider waiting for free transfer"
    elif breakeven_weeks < horizon:
        recommendation = "wait"
        confidence = "medium"
        reasoning = f"Hit pays off in {breakeven_weeks:.1f} weeks but margin is thin"
    else:
        recommendation = "avoid"
        confidence = "high"
        reasoning = f"Negative expected return. Hit unlikely to pay off"

    # Adjust for injury/availability
    if player_out.get("status") != "a":
        if net_gain > -2:
            recommendation = "take_hit"
            confidence = "medium"
            reasoning = f"Player out is flagged ({player_out['status']}). Hit necessary despite marginal gain"

    if player_in.get("status") != "a":
        recommendation = "avoid"
        confidence = "high"
        reasoning = f"Player in is flagged ({player_in['status']}). Do not take hit"

    return HitAnalysis(
        player_out_id=player_out_id,
        player_out_name=player_out["web_name"],
        player_in_id=player_in_id,
        player_in_name=player_in["web_name"],
        xp_out=round(xp_out, 2),
        xp_in=round(xp_in, 2),
        xp_differential=round(xp_differential, 2),
        hit_cost=hit_cost,
        net_gain=round(net_gain, 2),
        breakeven_weeks=round(breakeven_weeks, 2) if breakeven_weeks != float('inf') else None,
        recommendation=recommendation,
        confidence=confidence,
        reasoning=reasoning
    )


async def evaluate_multiple_hits(
    moves: list[tuple[int, int]],  # [(out_id, in_id), ...]
    horizon: int = 5
) -> MultiHitAnalysis:
    """Evaluate multiple transfer hits."""
    analyses = []
    total_xp_gain = 0
    total_hit_cost = 0

    for out_id, in_id in moves:
        analysis = await evaluate_hit(out_id, in_id, horizon, hit_cost=4)
        if analysis:
            analyses.append(analysis)
            total_xp_gain += analysis.xp_differential
            total_hit_cost += 4

    total_net_gain = total_xp_gain - total_hit_cost

    # Recommendation for multiple hits
    if total_net_gain > len(moves) * 3:
        recommendation = "take_hits"
        reasoning = f"Combined hits have strong positive return of {total_net_gain:.1f} pts"
    elif total_net_gain > 0:
        recommendation = "selective"
        reasoning = "Some hits may be worth it. Consider prioritizing highest value moves"
    else:
        recommendation = "avoid"
        reasoning = f"Combined hits have negative expected return of {total_net_gain:.1f} pts"

    return MultiHitAnalysis(
        moves=analyses,
        total_hit_cost=total_hit_cost,
        total_xp_gain=round(total_xp_gain, 2),
        total_net_gain=round(total_net_gain, 2),
        recommendation=recommendation,
        reasoning=reasoning
    )


async def get_worth_hit_transfers(
    player_out_id: int,
    budget: float | None = None,
    horizon: int = 5,
    limit: int = 10
) -> list[dict]:
    """Find transfers worth taking a hit for."""
    player_out = await fetch_one("""
        SELECT p.*, t.short_name as team FROM players p
        JOIN teams t ON p.team_id = t.id WHERE p.id = $1
    """, player_out_id)

    if not player_out:
        return []

    position = player_out["element_type"]
    out_price = player_out.get("now_cost", 0) or 0

    # Calculate effective budget
    max_price = int((budget or 15.0) * 10) + out_price

    # Get potential replacements
    candidates = await fetch_all("""
        SELECT p.id, p.web_name, p.now_cost, p.form, p.status, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.element_type = $1
          AND p.status = 'a'
          AND p.now_cost <= $2
          AND p.id != $3
          AND p.form >= 3
        ORDER BY p.form DESC
        LIMIT $4
    """, position, max_price, player_out_id, limit * 2)

    results = []
    for candidate in candidates:
        analysis = await evaluate_hit(player_out_id, candidate["id"], horizon)
        if analysis and analysis.net_gain > 0:
            results.append({
                "player_in": {
                    "id": candidate["id"],
                    "name": candidate["web_name"],
                    "team": candidate["team"],
                    "price": candidate["now_cost"] / 10 if candidate["now_cost"] else 0,
                    "form": float(candidate["form"] or 0)
                },
                "xp_gain": analysis.xp_differential,
                "net_gain_after_hit": analysis.net_gain,
                "breakeven_weeks": analysis.breakeven_weeks,
                "recommendation": analysis.recommendation,
                "confidence": analysis.confidence
            })

    # Sort by net gain
    results.sort(key=lambda x: x["net_gain_after_hit"], reverse=True)
    return results[:limit]


async def get_emergency_transfers(
    player_ids: list[int],
    budget_per_transfer: float = 1.0
) -> list[dict]:
    """Find emergency transfers for injured/suspended players."""
    if not player_ids:
        return []

    results = []

    for pid in player_ids:
        player = await fetch_one("""
            SELECT p.*, t.short_name as team FROM players p
            JOIN teams t ON p.team_id = t.id WHERE p.id = $1
        """, pid)

        if not player:
            continue

        if player.get("status") == "a":
            continue  # Skip healthy players

        # Find replacement
        worth_hits = await get_worth_hit_transfers(pid, budget_per_transfer, horizon=3, limit=3)

        results.append({
            "player_out": {
                "id": pid,
                "name": player["web_name"],
                "team": player["team"],
                "status": player.get("status"),
                "news": player.get("news")
            },
            "suggested_replacements": worth_hits,
            "urgency": "high" if player.get("chance_of_playing_next_round", 100) == 0 else "medium"
        })

    return results
