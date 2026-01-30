"""Price change prediction based on transfer activity and ownership trends."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class PricePrediction:
    """Price change prediction for a player."""
    player_id: int
    player_name: str
    team: str
    position: str
    current_price: float

    # Transfer activity
    transfers_in_event: int
    transfers_out_event: int
    net_transfers: int

    # Ownership
    ownership_percent: float
    ownership_change: float  # Estimated from snapshots

    # Prediction
    predicted_change: int  # -1, 0, +1
    confidence: str  # "high", "medium", "low"
    reason: str


async def predict_price_change(player_id: int) -> PricePrediction | None:
    """Predict if a player's price will rise or fall."""
    player = await fetch_one("""
        SELECT p.*, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id = $1
    """, player_id)

    if not player:
        return None

    transfers_in = player.get("transfers_in_event", 0) or 0
    transfers_out = player.get("transfers_out_event", 0) or 0
    net_transfers = transfers_in - transfers_out
    ownership = float(player.get("selected_by_percent", 0) or 0)

    # Get historical ownership from snapshots
    snapshots = await fetch_all("""
        SELECT gameweek, selected_by_percent, now_cost
        FROM player_snapshots
        WHERE player_id = $1
        ORDER BY gameweek DESC
        LIMIT 3
    """, player_id)

    ownership_change = 0.0
    if len(snapshots) >= 2:
        current_own = float(snapshots[0].get("selected_by_percent", 0) or 0)
        prev_own = float(snapshots[1].get("selected_by_percent", 0) or 0)
        ownership_change = current_own - prev_own

    # Price change prediction logic
    # Based on FPL's price change algorithm (approximately):
    # - Price rises when ~5% of active managers transfer in
    # - Price falls when ~5% of active managers transfer out
    # Active managers ~ 8-10 million, so threshold ~400k-500k net transfers

    predicted_change = 0
    confidence = "low"
    reason = "Stable transfer activity"

    # Aggressive transfer thresholds (simplified)
    RISE_THRESHOLD = 50000
    FALL_THRESHOLD = -50000

    if net_transfers > RISE_THRESHOLD:
        predicted_change = 1
        if net_transfers > RISE_THRESHOLD * 2:
            confidence = "high"
            reason = f"Very high transfer in volume ({net_transfers:,})"
        else:
            confidence = "medium"
            reason = f"Strong transfer in activity ({net_transfers:,})"
    elif net_transfers < FALL_THRESHOLD:
        predicted_change = -1
        if net_transfers < FALL_THRESHOLD * 2:
            confidence = "high"
            reason = f"Very high transfer out volume ({net_transfers:,})"
        else:
            confidence = "medium"
            reason = f"Strong transfer out activity ({net_transfers:,})"
    else:
        # Check if already at extremes
        if ownership > 50:
            reason = "High ownership may limit further rises"
        elif ownership < 1:
            reason = "Low ownership may limit further falls"
        else:
            reason = "Transfer activity within normal range"

    # Consider player status
    status = player.get("status", "a")
    if status != "a":
        if predicted_change >= 0:
            predicted_change = -1
            confidence = "medium"
            reason = f"Player flagged ({status}) - likely to fall"

    return PricePrediction(
        player_id=player_id,
        player_name=player["web_name"],
        team=player["team"],
        position=POSITION_MAP.get(player["element_type"], "?"),
        current_price=player["now_cost"] / 10 if player["now_cost"] else 0,
        transfers_in_event=transfers_in,
        transfers_out_event=transfers_out,
        net_transfers=net_transfers,
        ownership_percent=ownership,
        ownership_change=round(ownership_change, 2),
        predicted_change=predicted_change,
        confidence=confidence,
        reason=reason
    )


async def get_price_risers(limit: int = 15) -> list[dict]:
    """Get players most likely to rise in price."""
    # Get players with highest net transfers in
    players = await fetch_all("""
        SELECT p.id, p.web_name, p.element_type, p.now_cost,
               p.transfers_in_event, p.transfers_out_event,
               (p.transfers_in_event - p.transfers_out_event) as net_transfers,
               p.selected_by_percent, p.form, p.status,
               t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.status = 'a'
          AND (p.transfers_in_event - p.transfers_out_event) > 10000
        ORDER BY (p.transfers_in_event - p.transfers_out_event) DESC
        LIMIT $1
    """, limit)

    results = []
    for p in players:
        net = p["net_transfers"] or 0
        if net > 100000:
            confidence = "high"
        elif net > 50000:
            confidence = "medium"
        else:
            confidence = "low"

        results.append({
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "net_transfers": net,
            "transfers_in": p["transfers_in_event"] or 0,
            "transfers_out": p["transfers_out_event"] or 0,
            "ownership": float(p["selected_by_percent"] or 0),
            "form": float(p["form"] or 0),
            "confidence": confidence,
            "likely_to_rise": True
        })

    return results


async def get_price_fallers(limit: int = 15) -> list[dict]:
    """Get players most likely to fall in price."""
    # Get players with highest net transfers out
    players = await fetch_all("""
        SELECT p.id, p.web_name, p.element_type, p.now_cost,
               p.transfers_in_event, p.transfers_out_event,
               (p.transfers_in_event - p.transfers_out_event) as net_transfers,
               p.selected_by_percent, p.form, p.status, p.news,
               t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE (p.transfers_in_event - p.transfers_out_event) < -10000
        ORDER BY (p.transfers_in_event - p.transfers_out_event) ASC
        LIMIT $1
    """, limit)

    results = []
    for p in players:
        net = p["net_transfers"] or 0
        if net < -100000:
            confidence = "high"
        elif net < -50000:
            confidence = "medium"
        else:
            confidence = "low"

        results.append({
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "net_transfers": net,
            "transfers_in": p["transfers_in_event"] or 0,
            "transfers_out": p["transfers_out_event"] or 0,
            "ownership": float(p["selected_by_percent"] or 0),
            "status": p["status"],
            "news": p["news"],
            "confidence": confidence,
            "likely_to_fall": True
        })

    return results


async def get_price_change_summary() -> dict:
    """Get summary of predicted price changes."""
    risers = await get_price_risers(10)
    fallers = await get_price_fallers(10)

    return {
        "likely_risers": risers,
        "likely_fallers": fallers,
        "summary": {
            "risers_count": len(risers),
            "fallers_count": len(fallers),
            "highest_net_in": risers[0]["net_transfers"] if risers else 0,
            "highest_net_out": fallers[0]["net_transfers"] if fallers else 0
        }
    }
