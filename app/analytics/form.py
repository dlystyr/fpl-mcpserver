"""Player and team form analysis with momentum detection."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class PlayerForm:
    """Player form analysis."""
    player_id: int
    player_name: str
    team: str
    position: str

    # Form values
    current_form: float
    form_3gw: float
    form_5gw: float
    form_10gw: float

    # Momentum
    momentum: str  # "rising", "stable", "falling"
    momentum_score: float

    # ICT trends
    ict_trend: str
    influence_trend: float
    creativity_trend: float
    threat_trend: float


@dataclass
class TeamForm:
    """Team form analysis."""
    team_id: int
    team_name: str

    # Results
    recent_form: str  # "WWDLW"
    points_last_5: int
    goals_scored_last_5: int
    goals_conceded_last_5: int
    clean_sheets_last_5: int

    # Strength metrics
    attack_strength: float
    defence_strength: float
    home_form: str
    away_form: str


async def calculate_player_form(
    player_id: int,
    num_gameweeks: int = 10
) -> PlayerForm | None:
    """Calculate detailed form analysis for a player."""
    player = await fetch_one("""
        SELECT p.*, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE p.id = $1
    """, player_id)

    if not player:
        return None

    # Get historical snapshots
    snapshots = await fetch_all("""
        SELECT gameweek, form, ict_index, influence, creativity, threat,
               total_points, minutes
        FROM player_snapshots
        WHERE player_id = $1
        ORDER BY gameweek DESC
        LIMIT $2
    """, player_id, num_gameweeks)

    current_form = float(player.get("form", 0) or 0)

    # Calculate rolling form averages
    def calc_avg_form(snaps: list, n: int) -> float:
        relevant = snaps[:n] if len(snaps) >= n else snaps
        if not relevant:
            return current_form
        forms = [float(s["form"] or 0) for s in relevant]
        return round(sum(forms) / len(forms), 2) if forms else 0

    form_3gw = calc_avg_form(snapshots, 3)
    form_5gw = calc_avg_form(snapshots, 5)
    form_10gw = calc_avg_form(snapshots, 10)

    # Detect momentum
    if len(snapshots) >= 3:
        recent_forms = [float(s["form"] or 0) for s in snapshots[:3]]
        older_forms = [float(s["form"] or 0) for s in snapshots[3:6]] if len(snapshots) >= 6 else []

        recent_avg = sum(recent_forms) / len(recent_forms)
        older_avg = sum(older_forms) / len(older_forms) if older_forms else recent_avg

        momentum_score = recent_avg - older_avg

        if momentum_score > 1.0:
            momentum = "rising"
        elif momentum_score < -1.0:
            momentum = "falling"
        else:
            momentum = "stable"
    else:
        momentum = "stable"
        momentum_score = 0.0

    # ICT trends
    if len(snapshots) >= 3:
        recent_ict = [float(s["ict_index"] or 0) for s in snapshots[:3]]
        older_ict = [float(s["ict_index"] or 0) for s in snapshots[3:6]] if len(snapshots) >= 6 else []

        recent_inf = [float(s["influence"] or 0) for s in snapshots[:3]]
        recent_cre = [float(s["creativity"] or 0) for s in snapshots[:3]]
        recent_thr = [float(s["threat"] or 0) for s in snapshots[:3]]

        older_inf = [float(s["influence"] or 0) for s in snapshots[3:6]] if len(snapshots) >= 6 else recent_inf
        older_cre = [float(s["creativity"] or 0) for s in snapshots[3:6]] if len(snapshots) >= 6 else recent_cre
        older_thr = [float(s["threat"] or 0) for s in snapshots[3:6]] if len(snapshots) >= 6 else recent_thr

        def trend(recent: list, older: list) -> float:
            r_avg = sum(recent) / len(recent) if recent else 0
            o_avg = sum(older) / len(older) if older else r_avg
            return round(r_avg - o_avg, 2)

        influence_trend = trend(recent_inf, older_inf)
        creativity_trend = trend(recent_cre, older_cre)
        threat_trend = trend(recent_thr, older_thr)

        ict_change = trend(recent_ict, older_ict)
        if ict_change > 5:
            ict_trend = "improving"
        elif ict_change < -5:
            ict_trend = "declining"
        else:
            ict_trend = "stable"
    else:
        ict_trend = "stable"
        influence_trend = 0.0
        creativity_trend = 0.0
        threat_trend = 0.0

    return PlayerForm(
        player_id=player_id,
        player_name=player["web_name"],
        team=player["team"],
        position=POSITION_MAP.get(player["element_type"], "?"),
        current_form=current_form,
        form_3gw=form_3gw,
        form_5gw=form_5gw,
        form_10gw=form_10gw,
        momentum=momentum,
        momentum_score=round(momentum_score, 2),
        ict_trend=ict_trend,
        influence_trend=influence_trend,
        creativity_trend=creativity_trend,
        threat_trend=threat_trend
    )


async def calculate_team_form(team_id: int) -> TeamForm | None:
    """Calculate team form from recent fixtures."""
    team = await fetch_one("SELECT * FROM teams WHERE id = $1", team_id)
    if not team:
        return None

    # Get recent finished fixtures
    fixtures = await fetch_all("""
        SELECT
            f.*,
            CASE WHEN f.team_h = $1 THEN f.team_h_score ELSE f.team_a_score END as goals_for,
            CASE WHEN f.team_h = $1 THEN f.team_a_score ELSE f.team_h_score END as goals_against,
            CASE WHEN f.team_h = $1 THEN true ELSE false END as was_home
        FROM fixtures f
        WHERE (f.team_h = $1 OR f.team_a = $1) AND f.finished = true
        ORDER BY f.event DESC
        LIMIT 5
    """, team_id)

    if not fixtures:
        return TeamForm(
            team_id=team_id,
            team_name=team["name"],
            recent_form="",
            points_last_5=0,
            goals_scored_last_5=0,
            goals_conceded_last_5=0,
            clean_sheets_last_5=0,
            attack_strength=1.0,
            defence_strength=1.0,
            home_form="",
            away_form=""
        )

    results = []
    home_results = []
    away_results = []
    total_points = 0
    total_gf = 0
    total_ga = 0
    clean_sheets = 0

    for f in fixtures:
        gf = f["goals_for"] or 0
        ga = f["goals_against"] or 0
        total_gf += gf
        total_ga += ga

        if ga == 0:
            clean_sheets += 1

        if gf > ga:
            result = "W"
            total_points += 3
        elif gf < ga:
            result = "L"
        else:
            result = "D"
            total_points += 1

        results.append(result)
        if f["was_home"]:
            home_results.append(result)
        else:
            away_results.append(result)

    # Calculate strength (normalized to 1.0 baseline)
    attack_strength = (team.get("strength_attack_home", 1100) + team.get("strength_attack_away", 1100)) / 2200
    defence_strength = (team.get("strength_defence_home", 1100) + team.get("strength_defence_away", 1100)) / 2200

    return TeamForm(
        team_id=team_id,
        team_name=team["name"],
        recent_form="".join(results),
        points_last_5=total_points,
        goals_scored_last_5=total_gf,
        goals_conceded_last_5=total_ga,
        clean_sheets_last_5=clean_sheets,
        attack_strength=round(attack_strength, 2),
        defence_strength=round(defence_strength, 2),
        home_form="".join(home_results),
        away_form="".join(away_results)
    )


async def detect_form_momentum(
    position: str | None = None,
    momentum_type: str = "rising",
    limit: int = 10
) -> list[dict]:
    """Find players with specific momentum type."""
    conditions = ["p.status = 'a'", "p.minutes > 0"]
    params = []
    idx = 1

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    params.append(limit * 3)  # Get more to filter

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.form, p.now_cost,
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
        form = await calculate_player_form(p["id"])
        if form and form.momentum == momentum_type:
            results.append({
                "id": p["id"],
                "name": p["web_name"],
                "team": p["team"],
                "position": POSITION_MAP.get(p["element_type"], "?"),
                "price": p["now_cost"] / 10 if p["now_cost"] else 0,
                "current_form": form.current_form,
                "form_3gw": form.form_3gw,
                "momentum": form.momentum,
                "momentum_score": form.momentum_score,
                "ict_trend": form.ict_trend
            })

    # Sort by momentum score
    if momentum_type == "rising":
        results.sort(key=lambda x: x["momentum_score"], reverse=True)
    else:
        results.sort(key=lambda x: x["momentum_score"])

    return results[:limit]
