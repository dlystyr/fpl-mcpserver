"""Transfer suggestions and multi-gameweek planning."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class TransferTarget:
    """A potential transfer target."""
    player_id: int
    player_name: str
    team: str
    position: str
    price: float
    form: float
    xp_next_5: float
    fixture_difficulty: float
    score: float
    reasons: list[str]


@dataclass
class TransferPlan:
    """Multi-week transfer plan."""
    current_week: int
    planned_transfers: list[dict]
    total_hits: int
    projected_points_gain: float
    key_targets: list[str]


async def get_transfer_suggestions(
    position: str | None = None,
    budget: float | None = None,
    exclude_players: list[int] | None = None,
    limit: int = 10
) -> list[TransferTarget]:
    """Get transfer suggestions based on form, fixtures, and value."""
    from analytics.expected_points import calculate_expected_points
    from analytics.fixtures import get_fixture_difficulty

    conditions = ["p.status = 'a'", "p.minutes > 180"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    if budget:
        conditions.append(f"p.now_cost <= ${idx}")
        params.append(int(budget * 10))
        idx += 1

    if exclude_players:
        placeholders = ", ".join(f"${i}" for i in range(idx, idx + len(exclude_players)))
        conditions.append(f"p.id NOT IN ({placeholders})")
        params.extend(exclude_players)
        idx += len(exclude_players)

    params.append(limit * 3)  # Get more to filter/rank

    query = f"""
        SELECT p.id, p.web_name, p.team_id, p.element_type, p.now_cost,
               p.form, p.total_points, p.expected_goals, p.expected_assists,
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
        # Calculate expected points
        xp_data = await calculate_expected_points(p["id"], num_fixtures=5)
        xp = xp_data.final_xp if xp_data else float(p["form"] or 0)

        # Get fixture difficulty
        fd = await get_fixture_difficulty(p["team_id"], 5)
        fix_diff = fd.avg_difficulty_next_5 if fd else 3.0

        # Calculate composite score
        form = float(p["form"] or 0)
        price = p["now_cost"] / 10 if p["now_cost"] else 0
        value = xp / price if price > 0 else 0

        # Score formula: weighted combination
        score = (
            form * 10 +                    # Form weight
            (10 - fix_diff) * 5 +          # Fixture weight (inverse)
            value * 15 +                   # Value weight
            xp * 3                         # Raw xP weight
        )

        # Generate reasons
        reasons = []
        if form >= 6:
            reasons.append(f"Excellent form ({form})")
        elif form >= 4:
            reasons.append(f"Good form ({form})")

        if fix_diff <= 2.5:
            reasons.append("Easy upcoming fixtures")
        elif fix_diff >= 3.5:
            reasons.append("Tough fixtures ahead")

        if value >= 0.8:
            reasons.append(f"Great value ({value:.2f} xP/£M)")

        results.append(TransferTarget(
            player_id=p["id"],
            player_name=p["web_name"],
            team=p["team"],
            position=POSITION_MAP.get(p["element_type"], "?"),
            price=price,
            form=form,
            xp_next_5=round(xp, 2),
            fixture_difficulty=round(fix_diff, 2),
            score=round(score, 2),
            reasons=reasons
        ))

    # Sort by score
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


async def plan_transfers(
    current_team: list[int],
    budget: float,
    num_weeks: int = 5,
    free_transfers: int = 1
) -> TransferPlan | dict:
    """Plan transfers over multiple gameweeks."""
    from analytics.expected_points import calculate_expected_points
    from analytics.minutes import predict_minutes

    if not current_team:
        return {"error": "Current team required"}

    # Get current team data
    placeholders = ", ".join(f"${i+1}" for i in range(len(current_team)))
    team_players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.status, p.chance_of_playing_next_round, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
    """, *current_team)

    # Identify players to potentially transfer out
    transfer_out_candidates = []
    for p in team_players:
        xp_data = await calculate_expected_points(p["id"], num_fixtures=num_weeks)
        xp = xp_data.final_xp * num_weeks if xp_data else 0

        mins = await predict_minutes(p["id"])
        rotation_risk = mins.rotation_risk if mins else "medium"

        priority = 0
        reasons = []

        # Priority factors
        if p.get("status") != "a":
            priority += 30
            reasons.append(f"Flagged: {p['status']}")

        if rotation_risk == "high":
            priority += 20
            reasons.append("High rotation risk")

        if float(p.get("form", 0) or 0) < 3:
            priority += 10
            reasons.append("Poor form")

        if xp < 15:  # Low xP projection
            priority += 10
            reasons.append("Low xP projection")

        if priority > 0:
            transfer_out_candidates.append({
                "id": p["id"],
                "name": p["web_name"],
                "position": POSITION_MAP.get(p["element_type"], "?"),
                "price": p["now_cost"] / 10 if p["now_cost"] else 0,
                "priority": priority,
                "reasons": reasons,
                "xp_remaining": round(xp, 2)
            })

    # Sort by priority
    transfer_out_candidates.sort(key=lambda x: -x["priority"])

    # Plan transfers
    planned = []
    hits_needed = 0
    total_gain = 0
    available_ft = free_transfers

    for i, out_player in enumerate(transfer_out_candidates[:3]):  # Max 3 transfers planned
        # Find replacement
        position = out_player["position"]
        max_budget = budget + out_player["price"]

        suggestions = await get_transfer_suggestions(
            position=position,
            budget=max_budget,
            exclude_players=current_team,
            limit=3
        )

        if suggestions:
            best_in = suggestions[0]
            xp_gain = best_in.xp_next_5 * num_weeks - out_player["xp_remaining"]

            week = i + 1
            is_hit = i >= available_ft

            if is_hit:
                hits_needed += 1
                xp_gain -= 4  # Hit cost

            if xp_gain > 0 or out_player["priority"] >= 30:  # Worth it or necessary
                planned.append({
                    "week": week,
                    "out": {"id": out_player["id"], "name": out_player["name"]},
                    "in": {"id": best_in.player_id, "name": best_in.player_name, "price": best_in.price},
                    "is_hit": is_hit,
                    "xp_gain": round(xp_gain, 2),
                    "reasons": out_player["reasons"]
                })
                total_gain += xp_gain

    # Get current gameweek
    current_event = await fetch_one("SELECT id FROM events WHERE is_current = true")
    current_gw = current_event["id"] if current_event else 1

    key_targets = [t["in"]["name"] for t in planned[:2]]

    return TransferPlan(
        current_week=current_gw,
        planned_transfers=planned,
        total_hits=hits_needed,
        projected_points_gain=round(total_gain, 2),
        key_targets=key_targets
    )


async def get_best_transfers_by_position(
    current_team_ids: list[int] | None = None,
    budget: float = 15.0
) -> dict:
    """Get best transfer suggestions for each position."""
    positions = ["GK", "DEF", "MID", "FWD"]
    exclude = current_team_ids or []

    results = {}
    for pos in positions:
        suggestions = await get_transfer_suggestions(
            position=pos,
            budget=budget,
            exclude_players=exclude,
            limit=5
        )
        results[pos] = [{
            "id": s.player_id,
            "name": s.player_name,
            "team": s.team,
            "price": s.price,
            "form": s.form,
            "xp_next_5": s.xp_next_5,
            "score": s.score,
            "reasons": s.reasons
        } for s in suggestions]

    return results


async def find_like_for_like_replacements(
    player_id: int,
    price_tolerance: float = 0.5,
    limit: int = 5
) -> list[dict]:
    """Find similar replacements for a specific player."""
    player = await fetch_one("""
        SELECT p.*, t.short_name as team FROM players p
        JOIN teams t ON p.team_id = t.id WHERE p.id = $1
    """, player_id)

    if not player:
        return []

    position = player["element_type"]
    price = player.get("now_cost", 0) / 10

    min_price = price - price_tolerance
    max_price = price + price_tolerance

    suggestions = await get_transfer_suggestions(
        position=POSITION_MAP.get(position, "MID"),
        budget=max_price,
        exclude_players=[player_id],
        limit=limit * 2
    )

    # Filter by price range
    filtered = [s for s in suggestions if s.price >= min_price]
    return [{
        "id": s.player_id,
        "name": s.player_name,
        "team": s.team,
        "position": s.position,
        "price": s.price,
        "price_diff": round(s.price - price, 1),
        "form": s.form,
        "xp_next_5": s.xp_next_5,
        "score": s.score
    } for s in filtered[:limit]]


async def get_premium_to_budget_options(
    player_out_id: int,
    min_savings: float = 2.0,
    limit: int = 5
) -> list[dict]:
    """Find budget options to free up funds from a premium player."""
    player = await fetch_one("""
        SELECT p.*, t.short_name as team FROM players p
        JOIN teams t ON p.team_id = t.id WHERE p.id = $1
    """, player_out_id)

    if not player:
        return []

    position = player["element_type"]
    price = player.get("now_cost", 0) / 10

    # Find cheaper alternatives
    max_price = price - min_savings

    if max_price < 4:  # Too cheap, no realistic options
        return []

    suggestions = await get_transfer_suggestions(
        position=POSITION_MAP.get(position, "MID"),
        budget=max_price,
        exclude_players=[player_out_id],
        limit=limit
    )

    return [{
        "id": s.player_id,
        "name": s.player_name,
        "team": s.team,
        "position": s.position,
        "price": s.price,
        "savings": round(price - s.price, 1),
        "form": s.form,
        "xp_next_5": s.xp_next_5,
        "value_score": round(s.xp_next_5 / s.price, 2) if s.price > 0 else 0
    } for s in suggestions]
