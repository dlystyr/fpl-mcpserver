"""Fixture analysis: difficulty ratings, DGW/BGW detection, best fixture runs."""

from dataclasses import dataclass
from database import fetch_all, fetch_one

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


@dataclass
class FixtureDifficulty:
    """Fixture difficulty analysis for a team."""
    team_id: int
    team_name: str

    # Aggregate difficulty
    avg_difficulty_next_5: float
    avg_difficulty_next_10: float

    # Breakdown
    fixtures: list[dict]

    # Classification
    run_quality: str  # "excellent", "good", "mixed", "tough"


@dataclass
class DGWInfo:
    """Double Gameweek information."""
    gameweek: int
    teams: list[dict]  # [{team_id, team_name, fixtures: [...]}]


@dataclass
class BGWInfo:
    """Blank Gameweek information."""
    gameweek: int
    teams_with_blanks: list[dict]  # [{team_id, team_name}]


async def get_fixture_difficulty(
    team_id: int,
    num_fixtures: int = 10
) -> FixtureDifficulty | None:
    """Get fixture difficulty analysis for a team."""
    team = await fetch_one("SELECT * FROM teams WHERE id = $1", team_id)
    if not team:
        return None

    fixtures = await fetch_all("""
        SELECT
            f.id, f.event, f.team_h, f.team_a,
            f.team_h_difficulty, f.team_a_difficulty,
            f.kickoff_time,
            th.short_name as home_team,
            ta.short_name as away_team,
            CASE WHEN f.team_h = $1 THEN true ELSE false END as is_home,
            CASE
                WHEN f.team_h = $1 THEN f.team_h_difficulty
                ELSE f.team_a_difficulty
            END as difficulty
        FROM fixtures f
        JOIN teams th ON f.team_h = th.id
        JOIN teams ta ON f.team_a = ta.id
        WHERE (f.team_h = $1 OR f.team_a = $1) AND f.finished = false
        ORDER BY f.event
        LIMIT $2
    """, team_id, num_fixtures)

    if not fixtures:
        return FixtureDifficulty(
            team_id=team_id,
            team_name=team["name"],
            avg_difficulty_next_5=3.0,
            avg_difficulty_next_10=3.0,
            fixtures=[],
            run_quality="mixed"
        )

    fixture_list = []
    for f in fixtures:
        fixture_list.append({
            "gameweek": f["event"],
            "opponent": f["away_team"] if f["is_home"] else f["home_team"],
            "is_home": f["is_home"],
            "difficulty": f["difficulty"],
            "kickoff": str(f["kickoff_time"]) if f["kickoff_time"] else None
        })

    # Calculate averages
    difficulties = [f["difficulty"] or 3 for f in fixture_list]
    avg_5 = sum(difficulties[:5]) / min(5, len(difficulties)) if difficulties else 3.0
    avg_10 = sum(difficulties[:10]) / min(10, len(difficulties)) if difficulties else 3.0

    # Classify run quality
    if avg_5 <= 2.5:
        run_quality = "excellent"
    elif avg_5 <= 3.0:
        run_quality = "good"
    elif avg_5 <= 3.5:
        run_quality = "mixed"
    else:
        run_quality = "tough"

    return FixtureDifficulty(
        team_id=team_id,
        team_name=team["name"],
        avg_difficulty_next_5=round(avg_5, 2),
        avg_difficulty_next_10=round(avg_10, 2),
        fixtures=fixture_list,
        run_quality=run_quality
    )


async def get_easiest_fixtures(
    num_gameweeks: int = 5,
    limit: int = 10
) -> list[dict]:
    """Get teams with easiest upcoming fixtures."""
    teams = await fetch_all("SELECT id, name, short_name FROM teams")

    results = []
    for team in teams:
        fd = await get_fixture_difficulty(team["id"], num_gameweeks)
        if fd:
            results.append({
                "team_id": team["id"],
                "team_name": team["name"],
                "short_name": team["short_name"],
                "avg_difficulty": fd.avg_difficulty_next_5,
                "run_quality": fd.run_quality,
                "fixtures": fd.fixtures[:num_gameweeks]
            })

    results.sort(key=lambda x: x["avg_difficulty"])
    return results[:limit]


async def get_hardest_fixtures(
    num_gameweeks: int = 5,
    limit: int = 10
) -> list[dict]:
    """Get teams with hardest upcoming fixtures (teams to avoid)."""
    teams = await fetch_all("SELECT id, name, short_name FROM teams")

    results = []
    for team in teams:
        fd = await get_fixture_difficulty(team["id"], num_gameweeks)
        if fd:
            results.append({
                "team_id": team["id"],
                "team_name": team["name"],
                "short_name": team["short_name"],
                "avg_difficulty": fd.avg_difficulty_next_5,
                "run_quality": fd.run_quality,
                "fixtures": fd.fixtures[:num_gameweeks]
            })

    results.sort(key=lambda x: x["avg_difficulty"], reverse=True)
    return results[:limit]


async def detect_double_gameweeks() -> list[DGWInfo]:
    """Detect upcoming double gameweeks."""
    # Find gameweeks where a team has multiple fixtures
    dgw_query = await fetch_all("""
        WITH team_fixture_counts AS (
            SELECT
                f.event,
                CASE WHEN f.team_h = t.id THEN t.id ELSE NULL END as team_h_id,
                CASE WHEN f.team_a = t.id THEN t.id ELSE NULL END as team_a_id,
                t.id as team_id,
                t.name as team_name,
                t.short_name
            FROM fixtures f
            CROSS JOIN teams t
            WHERE f.finished = false
              AND (f.team_h = t.id OR f.team_a = t.id)
        ),
        dgw_teams AS (
            SELECT
                event,
                team_id,
                team_name,
                short_name,
                COUNT(*) as fixture_count
            FROM team_fixture_counts
            WHERE team_id IS NOT NULL
            GROUP BY event, team_id, team_name, short_name
            HAVING COUNT(*) > 1
        )
        SELECT * FROM dgw_teams
        ORDER BY event, team_name
    """)

    # Group by gameweek
    dgw_map = {}
    for row in dgw_query:
        gw = row["event"]
        if gw not in dgw_map:
            dgw_map[gw] = []

        # Get specific fixtures for this team in this GW
        team_fixtures = await fetch_all("""
            SELECT
                f.id,
                th.short_name as home_team,
                ta.short_name as away_team,
                CASE WHEN f.team_h = $1 THEN true ELSE false END as is_home
            FROM fixtures f
            JOIN teams th ON f.team_h = th.id
            JOIN teams ta ON f.team_a = ta.id
            WHERE f.event = $2 AND (f.team_h = $1 OR f.team_a = $1)
        """, row["team_id"], gw)

        dgw_map[gw].append({
            "team_id": row["team_id"],
            "team_name": row["team_name"],
            "short_name": row["short_name"],
            "fixture_count": row["fixture_count"],
            "fixtures": [{
                "opponent": tf["away_team"] if tf["is_home"] else tf["home_team"],
                "is_home": tf["is_home"]
            } for tf in team_fixtures]
        })

    return [
        DGWInfo(gameweek=gw, teams=teams)
        for gw, teams in sorted(dgw_map.items())
    ]


async def detect_blank_gameweeks() -> list[BGWInfo]:
    """Detect upcoming blank gameweeks (teams with no fixtures)."""
    # Get all future gameweeks
    events = await fetch_all("""
        SELECT id FROM events
        WHERE finished = false AND id IS NOT NULL
        ORDER BY id
    """)

    if not events:
        return []

    all_teams = await fetch_all("SELECT id, name, short_name FROM teams")
    team_ids = {t["id"]: t for t in all_teams}

    bgw_list = []

    for event in events:
        gw = event["id"]

        # Get teams playing in this GW
        playing_teams = await fetch_all("""
            SELECT DISTINCT team_h as team_id FROM fixtures WHERE event = $1
            UNION
            SELECT DISTINCT team_a as team_id FROM fixtures WHERE event = $1
        """, gw)

        playing_ids = {t["team_id"] for t in playing_teams}

        # Find teams NOT playing
        blank_teams = [
            {"team_id": tid, "team_name": t["name"], "short_name": t["short_name"]}
            for tid, t in team_ids.items()
            if tid not in playing_ids
        ]

        if blank_teams:
            bgw_list.append(BGWInfo(
                gameweek=gw,
                teams_with_blanks=blank_teams
            ))

    return bgw_list


async def get_fixture_ticker(num_gameweeks: int = 6) -> list[dict]:
    """Get fixture difficulty ticker for all teams."""
    teams = await fetch_all("SELECT id, name, short_name FROM teams ORDER BY name")

    results = []
    for team in teams:
        fd = await get_fixture_difficulty(team["id"], num_gameweeks)
        if fd:
            results.append({
                "team_id": team["id"],
                "team_name": team["name"],
                "short_name": team["short_name"],
                "avg_difficulty": fd.avg_difficulty_next_5,
                "run_quality": fd.run_quality,
                "fixtures": [
                    {
                        "gw": f["gameweek"],
                        "opp": f["opponent"],
                        "home": f["is_home"],
                        "fdr": f["difficulty"]
                    }
                    for f in fd.fixtures[:num_gameweeks]
                ]
            })

    return results


async def get_best_fixture_players(
    position: str | None = None,
    num_gameweeks: int = 5,
    limit: int = 15
) -> list[dict]:
    """Get players from teams with best upcoming fixtures."""
    # Get teams sorted by fixture ease
    easy_fixtures = await get_easiest_fixtures(num_gameweeks, limit=8)
    easy_team_ids = [t["team_id"] for t in easy_fixtures]

    if not easy_team_ids:
        return []

    conditions = ["p.status = 'a'", "p.minutes > 90"]
    params = []
    idx = 1

    # Build IN clause for team IDs
    team_placeholders = ", ".join(f"${i}" for i in range(idx, idx + len(easy_team_ids)))
    conditions.append(f"p.team_id IN ({team_placeholders})")
    params.extend(easy_team_ids)
    idx += len(easy_team_ids)

    if position:
        pos_map = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
        conditions.append(f"p.element_type = ${idx}")
        params.append(pos_map.get(position.upper(), 3))
        idx += 1

    params.append(limit)

    query = f"""
        SELECT p.id, p.web_name, p.element_type, p.now_cost, p.form,
               p.total_points, p.team_id, t.short_name as team
        FROM players p
        JOIN teams t ON p.team_id = t.id
        WHERE {' AND '.join(conditions)}
        ORDER BY p.form DESC
        LIMIT ${idx}
    """

    players = await fetch_all(query, *params)

    # Enrich with fixture info
    team_fixture_map = {t["team_id"]: t for t in easy_fixtures}

    results = []
    for p in players:
        team_info = team_fixture_map.get(p["team_id"], {})
        results.append({
            "id": p["id"],
            "name": p["web_name"],
            "team": p["team"],
            "position": POSITION_MAP.get(p["element_type"], "?"),
            "price": p["now_cost"] / 10 if p["now_cost"] else 0,
            "form": float(p["form"]) if p["form"] else 0,
            "fixture_difficulty": team_info.get("avg_difficulty", 3.0),
            "run_quality": team_info.get("run_quality", "mixed"),
            "next_fixtures": [
                f"{f['opponent']} ({'H' if f['is_home'] else 'A'})"
                for f in team_info.get("fixtures", [])[:3]
            ]
        })

    return results
