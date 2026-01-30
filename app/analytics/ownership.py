"""Ownership analysis: template players, differentials, and EO tracking."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class TemplateScore:
    """How template/differential a team is."""
    player_ids: list[int]
    total_players: int
    template_players: int
    differential_players: int
    average_ownership: float
    template_score: float  # 0-100, higher = more template
    classification: str  # "very template", "template", "balanced", "differential"


async def get_template_players(
    position: str | None = None,
    min_ownership: float = 20.0,
    limit: int = 20
) -> list[dict]:
    """Get highly-owned template players."""
    conditions = ["p.status = 'a'", f"p.selected_by_percent >= {min_ownership}"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    params.append(limit)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, p.selected_by_percent, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.selected_by_percent DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    return [{
        "id": p["id"],
        "name": p["web_name"],
        "team": p["team"],
        "position": POSITION_MAP.get(p["element_type"], "?"),
        "price": p["now_cost"] / 10 if p["now_cost"] else 0,
        "ownership": float(p["selected_by_percent"] or 0),
        "form": float(p["form"] or 0),
        "points": p["total_points"] or 0,
        "is_template": True
    } for p in players]


async def get_differentials(
    position: str | None = None,
    max_ownership: float = 10.0,
    min_form: float = 3.0,
    limit: int = 15
) -> list[dict]:
    """Get low-ownership differential players with good form."""
    conditions = [
        "p.status = 'a'",
        f"p.selected_by_percent <= {max_ownership}",
        f"p.form >= {min_form}",
        "p.minutes > 180"
    ]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    params.append(limit)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, p.selected_by_percent, p.expected_goals,
               p.expected_assists, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.form DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    return [{
        "id": p["id"],
        "name": p["web_name"],
        "team": p["team"],
        "position": POSITION_MAP.get(p["element_type"], "?"),
        "price": p["now_cost"] / 10 if p["now_cost"] else 0,
        "ownership": float(p["selected_by_percent"] or 0),
        "form": float(p["form"] or 0),
        "points": p["total_points"] or 0,
        "xG": round(float(p["expected_goals"] or 0), 2),
        "xA": round(float(p["expected_assists"] or 0), 2),
        "is_differential": True
    } for p in players]


async def calculate_template_score(player_ids: list[int]) -> TemplateScore:
    """Calculate how template or differential a squad is."""
    if not player_ids:
        return TemplateScore(
            player_ids=[],
            total_players=0,
            template_players=0,
            differential_players=0,
            average_ownership=0.0,
            template_score=50.0,
            classification="balanced"
        )

    placeholders = ", ".join(f"${i+1}" for i in range(len(player_ids)))
    players = await fetch_all(f"""
        SELECT id, selected_by_percent FROM players WHERE id IN ({placeholders})
    """, *player_ids)

    ownerships = [float(p["selected_by_percent"] or 0) for p in players]
    avg_ownership = sum(ownerships) / len(ownerships) if ownerships else 0

    template_count = sum(1 for o in ownerships if o >= 20)
    differential_count = sum(1 for o in ownerships if o <= 5)

    # Template score: weighted average of ownerships, scaled to 0-100
    template_score = min(100, avg_ownership * 2)

    if template_score >= 70:
        classification = "very template"
    elif template_score >= 50:
        classification = "template"
    elif template_score >= 30:
        classification = "balanced"
    else:
        classification = "differential"

    return TemplateScore(
        player_ids=player_ids,
        total_players=len(players),
        template_players=template_count,
        differential_players=differential_count,
        average_ownership=round(avg_ownership, 2),
        template_score=round(template_score, 2),
        classification=classification
    )


async def get_ownership_bands(position: str | None = None) -> dict:
    """Get players grouped by ownership bands."""
    conditions = ["p.status = 'a'", "p.minutes > 0"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.selected_by_percent, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.selected_by_percent DESC
    """

    players = await fetch_all(query, *params)

    bands = {
        "essential (50%+)": [],
        "template (20-50%)": [],
        "popular (10-20%)": [],
        "differential (5-10%)": [],
        "punt (<5%)": []
    }

    for p in players:
        own = float(p["selected_by_percent"] or 0)
        player_data = {
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "ownership": own,
            "form": float(p["form"] or 0)
        }

        if own >= 50:
            bands["essential (50%+)"].append(player_data)
        elif own >= 20:
            bands["template (20-50%)"].append(player_data)
        elif own >= 10:
            bands["popular (10-20%)"].append(player_data)
        elif own >= 5:
            bands["differential (5-10%)"].append(player_data)
        else:
            if float(p["form"] or 0) >= 3:  # Only include decent form punts
                bands["punt (<5%)"].append(player_data)

    # Limit each band
    for band in bands:
        bands[band] = bands[band][:10]

    return bands


async def get_captaincy_eo(player_ids: list[int]) -> list[dict]:
    """Get effective ownership analysis for captain picks."""
    if not player_ids:
        return []

    placeholders = ", ".join(f"${i+1}" for i in range(len(player_ids)))
    players = await fetch_all(f"""
        SELECT p.id, p.web_name, p.selected_by_percent, p.form,
               p.total_points, t.short_name as team, p.element_type
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id IN ({placeholders})
        ORDER BY p.selected_by_percent DESC
    """, *player_ids)

    # Estimate captain EO (typically ~2x ownership for popular picks)
    results = []
    for p in players:
        ownership = float(p["selected_by_percent"] or 0)
        # Popular players get captained more
        if ownership > 30:
            captain_factor = 0.5  # ~50% of owners captain
        elif ownership > 15:
            captain_factor = 0.35
        else:
            captain_factor = 0.2

        captain_eo = ownership * captain_factor
        effective_ownership = ownership + captain_eo  # Base + captain boost

        results.append({
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "ownership": round(ownership, 2),
            "estimated_captain_eo": round(captain_eo, 2),
            "effective_ownership": round(effective_ownership, 2),
            "form": float(p["form"] or 0)
        })

    return results
