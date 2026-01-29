"""Sync FPL data from API to PostgreSQL and Redis."""

import asyncio
import logging
import httpx
from datetime import datetime, timezone
from dateutil import parser as dateparser

from config import get_settings


def parse_datetime(value: str | None) -> datetime | None:
    """Parse datetime string from FPL API to naive UTC datetime."""
    if not value:
        return None
    try:
        dt = dateparser.parse(value)
        # Convert to UTC and strip timezone info for PostgreSQL TIMESTAMP columns
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def utc_now() -> datetime:
    """Return current UTC time as naive datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


from database import init_db, close_db, get_pool
from cache import init_cache, close_cache, cache_set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FPL_API = "https://fantasy.premierleague.com/api"


async def fetch_fpl(client: httpx.AsyncClient, endpoint: str) -> dict | list:
    """Fetch from FPL API."""
    resp = await client.get(f"{FPL_API}/{endpoint}/")
    resp.raise_for_status()
    return resp.json()


async def sync_teams(pool, teams: list) -> int:
    """Sync teams to database."""
    async with pool.acquire() as conn:
        for t in teams:
            await conn.execute("""
                INSERT INTO teams (id, name, short_name, code, strength,
                    strength_overall_home, strength_overall_away,
                    strength_attack_home, strength_attack_away,
                    strength_defence_home, strength_defence_away, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, short_name = EXCLUDED.short_name,
                    strength = EXCLUDED.strength,
                    strength_overall_home = EXCLUDED.strength_overall_home,
                    strength_overall_away = EXCLUDED.strength_overall_away,
                    strength_attack_home = EXCLUDED.strength_attack_home,
                    strength_attack_away = EXCLUDED.strength_attack_away,
                    strength_defence_home = EXCLUDED.strength_defence_home,
                    strength_defence_away = EXCLUDED.strength_defence_away,
                    updated_at = EXCLUDED.updated_at
            """, t["id"], t["name"], t["short_name"], t.get("code"),
                t.get("strength"), t.get("strength_overall_home"),
                t.get("strength_overall_away"), t.get("strength_attack_home"),
                t.get("strength_attack_away"), t.get("strength_defence_home"),
                t.get("strength_defence_away"), utc_now())
    return len(teams)


async def sync_events(pool, events: list) -> int:
    """Sync events/gameweeks to database."""
    async with pool.acquire() as conn:
        for e in events:
            await conn.execute("""
                INSERT INTO events (id, name, deadline_time, finished, is_current, is_next,
                    is_previous, average_entry_score, highest_score, most_selected,
                    most_captained, most_vice_captained, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name, deadline_time = EXCLUDED.deadline_time,
                    finished = EXCLUDED.finished, is_current = EXCLUDED.is_current,
                    is_next = EXCLUDED.is_next, is_previous = EXCLUDED.is_previous,
                    average_entry_score = EXCLUDED.average_entry_score,
                    highest_score = EXCLUDED.highest_score,
                    most_selected = EXCLUDED.most_selected,
                    most_captained = EXCLUDED.most_captained,
                    most_vice_captained = EXCLUDED.most_vice_captained,
                    updated_at = EXCLUDED.updated_at
            """, e["id"], e["name"], parse_datetime(e.get("deadline_time")),
                e.get("finished", False), e.get("is_current", False),
                e.get("is_next", False), e.get("is_previous", False),
                e.get("average_entry_score"), e.get("highest_score"),
                e.get("most_selected"), e.get("most_captained"),
                e.get("most_vice_captained"), utc_now())
    return len(events)


async def sync_players(pool, players: list) -> int:
    """Sync players to database."""
    async with pool.acquire() as conn:
        for p in players:
            await conn.execute("""
                INSERT INTO players (id, code, first_name, second_name, web_name, team_id,
                    element_type, now_cost, selected_by_percent, transfers_in_event,
                    transfers_out_event, form, points_per_game, total_points, minutes,
                    goals_scored, assists, clean_sheets, goals_conceded, own_goals,
                    penalties_saved, penalties_missed, yellow_cards, red_cards, saves,
                    bonus, bps, expected_goals, expected_assists, expected_goal_involvements,
                    expected_goals_conceded, influence, creativity, threat, ict_index, starts,
                    status, chance_of_playing_next_round, chance_of_playing_this_round,
                    news, news_added, updated_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
                        $21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38,$39,$40,$41,$42)
                ON CONFLICT (id) DO UPDATE SET
                    web_name = EXCLUDED.web_name, team_id = EXCLUDED.team_id,
                    now_cost = EXCLUDED.now_cost, selected_by_percent = EXCLUDED.selected_by_percent,
                    transfers_in_event = EXCLUDED.transfers_in_event,
                    transfers_out_event = EXCLUDED.transfers_out_event,
                    form = EXCLUDED.form, points_per_game = EXCLUDED.points_per_game,
                    total_points = EXCLUDED.total_points, minutes = EXCLUDED.minutes,
                    goals_scored = EXCLUDED.goals_scored, assists = EXCLUDED.assists,
                    clean_sheets = EXCLUDED.clean_sheets, goals_conceded = EXCLUDED.goals_conceded,
                    own_goals = EXCLUDED.own_goals, penalties_saved = EXCLUDED.penalties_saved,
                    penalties_missed = EXCLUDED.penalties_missed, yellow_cards = EXCLUDED.yellow_cards,
                    red_cards = EXCLUDED.red_cards, saves = EXCLUDED.saves,
                    bonus = EXCLUDED.bonus, bps = EXCLUDED.bps,
                    expected_goals = EXCLUDED.expected_goals, expected_assists = EXCLUDED.expected_assists,
                    expected_goal_involvements = EXCLUDED.expected_goal_involvements,
                    expected_goals_conceded = EXCLUDED.expected_goals_conceded,
                    influence = EXCLUDED.influence, creativity = EXCLUDED.creativity,
                    threat = EXCLUDED.threat, ict_index = EXCLUDED.ict_index,
                    starts = EXCLUDED.starts, status = EXCLUDED.status,
                    chance_of_playing_next_round = EXCLUDED.chance_of_playing_next_round,
                    chance_of_playing_this_round = EXCLUDED.chance_of_playing_this_round,
                    news = EXCLUDED.news, news_added = EXCLUDED.news_added,
                    updated_at = EXCLUDED.updated_at
            """, p["id"], p.get("code"), p.get("first_name"), p.get("second_name"),
                p["web_name"], p["team"], p["element_type"], p.get("now_cost"),
                float(p.get("selected_by_percent") or 0),
                p.get("transfers_in_event", 0), p.get("transfers_out_event", 0),
                float(p.get("form") or 0), float(p.get("points_per_game") or 0),
                p.get("total_points", 0), p.get("minutes", 0),
                p.get("goals_scored", 0), p.get("assists", 0),
                p.get("clean_sheets", 0), p.get("goals_conceded", 0),
                p.get("own_goals", 0), p.get("penalties_saved", 0),
                p.get("penalties_missed", 0), p.get("yellow_cards", 0),
                p.get("red_cards", 0), p.get("saves", 0),
                p.get("bonus", 0), p.get("bps", 0),
                float(p.get("expected_goals") or 0), float(p.get("expected_assists") or 0),
                float(p.get("expected_goal_involvements") or 0),
                float(p.get("expected_goals_conceded") or 0),
                float(p.get("influence") or 0), float(p.get("creativity") or 0),
                float(p.get("threat") or 0), float(p.get("ict_index") or 0),
                p.get("starts", 0), p.get("status", "a"),
                p.get("chance_of_playing_next_round"),
                p.get("chance_of_playing_this_round"),
                p.get("news"), parse_datetime(p.get("news_added")), utc_now())
    return len(players)


async def sync_fixtures(pool, fixtures: list) -> int:
    """Sync fixtures to database."""
    async with pool.acquire() as conn:
        for f in fixtures:
            await conn.execute("""
                INSERT INTO fixtures (id, code, event, team_h, team_a, team_h_score,
                    team_a_score, finished, kickoff_time, team_h_difficulty,
                    team_a_difficulty, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    event = EXCLUDED.event, team_h_score = EXCLUDED.team_h_score,
                    team_a_score = EXCLUDED.team_a_score, finished = EXCLUDED.finished,
                    kickoff_time = EXCLUDED.kickoff_time,
                    team_h_difficulty = EXCLUDED.team_h_difficulty,
                    team_a_difficulty = EXCLUDED.team_a_difficulty,
                    updated_at = EXCLUDED.updated_at
            """, f["id"], f.get("code"), f.get("event"), f["team_h"], f["team_a"],
                f.get("team_h_score"), f.get("team_a_score"),
                f.get("finished", False), parse_datetime(f.get("kickoff_time")),
                f.get("team_h_difficulty"), f.get("team_a_difficulty"),
                utc_now())
    return len(fixtures)


async def sync_player_snapshots(pool, players: list, current_gw: int) -> int:
    """Sync player snapshots for current gameweek (upsert)."""
    if not current_gw:
        logger.warning("No current gameweek, skipping snapshots")
        return 0

    async with pool.acquire() as conn:
        for p in players:
            await conn.execute("""
                INSERT INTO player_snapshots (
                    player_id, gameweek, now_cost, selected_by_percent, form,
                    points_per_game, total_points, minutes, goals_scored, assists,
                    clean_sheets, goals_conceded, bonus, bps, expected_goals,
                    expected_assists, expected_goal_involvements, influence,
                    creativity, threat, ict_index, transfers_in_event, transfers_out_event
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23)
                ON CONFLICT (player_id, gameweek) DO UPDATE SET
                    now_cost = EXCLUDED.now_cost,
                    selected_by_percent = EXCLUDED.selected_by_percent,
                    form = EXCLUDED.form,
                    points_per_game = EXCLUDED.points_per_game,
                    total_points = EXCLUDED.total_points,
                    minutes = EXCLUDED.minutes,
                    goals_scored = EXCLUDED.goals_scored,
                    assists = EXCLUDED.assists,
                    clean_sheets = EXCLUDED.clean_sheets,
                    goals_conceded = EXCLUDED.goals_conceded,
                    bonus = EXCLUDED.bonus,
                    bps = EXCLUDED.bps,
                    expected_goals = EXCLUDED.expected_goals,
                    expected_assists = EXCLUDED.expected_assists,
                    expected_goal_involvements = EXCLUDED.expected_goal_involvements,
                    influence = EXCLUDED.influence,
                    creativity = EXCLUDED.creativity,
                    threat = EXCLUDED.threat,
                    ict_index = EXCLUDED.ict_index,
                    transfers_in_event = EXCLUDED.transfers_in_event,
                    transfers_out_event = EXCLUDED.transfers_out_event
            """, p["id"], current_gw, p.get("now_cost"),
                float(p.get("selected_by_percent") or 0),
                float(p.get("form") or 0), float(p.get("points_per_game") or 0),
                p.get("total_points", 0), p.get("minutes", 0),
                p.get("goals_scored", 0), p.get("assists", 0),
                p.get("clean_sheets", 0), p.get("goals_conceded", 0),
                p.get("bonus", 0), p.get("bps", 0),
                float(p.get("expected_goals") or 0), float(p.get("expected_assists") or 0),
                float(p.get("expected_goal_involvements") or 0),
                float(p.get("influence") or 0), float(p.get("creativity") or 0),
                float(p.get("threat") or 0), float(p.get("ict_index") or 0),
                p.get("transfers_in_event", 0), p.get("transfers_out_event", 0))
    return len(players)


async def populate_redis(teams: list, players: list, fixtures: list, events: list):
    """Populate Redis cache."""
    # Teams
    await cache_set("fpl:teams:all", teams)
    for t in teams:
        await cache_set(f"fpl:team:{t['id']}", t)

    # Players with team info
    team_map = {t["id"]: t for t in teams}
    for p in players:
        t = team_map.get(p["team"])
        if t:
            p["team_name"] = t["name"]
            p["team_short"] = t["short_name"]

    await cache_set("fpl:players:all", players)
    for p in players:
        await cache_set(f"fpl:player:{p['id']}", p)

    # Fixtures
    await cache_set("fpl:fixtures:all", fixtures)
    gw_map = {}
    for f in fixtures:
        gw = f.get("event")
        if gw:
            gw_map.setdefault(gw, []).append(f)
    for gw, gw_fix in gw_map.items():
        await cache_set(f"fpl:fixtures:gw:{gw}", gw_fix)

    # Current GW
    current = next((e for e in events if e.get("is_current")), None)
    if current:
        await cache_set("fpl:current_gw", current["id"])

    await cache_set("fpl:last_updated", utc_now().isoformat())


async def main():
    logger.info("=" * 50)
    logger.info("FPL Data Sync")
    logger.info("=" * 50)

    await init_db()
    pool = await get_pool()
    redis_ok = await init_cache()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch bootstrap
        logger.info("Fetching bootstrap data...")
        bootstrap = await fetch_fpl(client, "bootstrap-static")
        teams = bootstrap.get("teams", [])
        events = bootstrap.get("events", [])
        players = bootstrap.get("elements", [])

        logger.info(f"Found {len(teams)} teams, {len(players)} players")

        # Get current gameweek
        current_event = next((e for e in events if e.get("is_current")), None)
        current_gw = current_event["id"] if current_event else None
        logger.info(f"Current gameweek: {current_gw}")

        # Sync to PostgreSQL
        logger.info("Syncing to PostgreSQL...")
        await sync_teams(pool, teams)
        await sync_events(pool, events)
        await sync_players(pool, players)

        # Fetch and sync fixtures
        logger.info("Fetching fixtures...")
        fixtures = await fetch_fpl(client, "fixtures")
        await sync_fixtures(pool, fixtures)
        logger.info(f"Synced {len(fixtures)} fixtures")

        # Save player snapshots for current gameweek
        logger.info(f"Saving player snapshots for GW{current_gw}...")
        await sync_player_snapshots(pool, players, current_gw)
        logger.info(f"Saved {len(players)} player snapshots")

        # Populate Redis
        if redis_ok:
            logger.info("Populating Redis cache...")
            await populate_redis(teams, players, fixtures, events)
            logger.info("Redis cache populated")

    await close_cache()
    await close_db()

    logger.info("=" * 50)
    logger.info("Sync complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
