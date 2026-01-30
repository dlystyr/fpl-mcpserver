"""Chip timing optimization: TC, BB, FH, WC analysis."""

from dataclasses import dataclass
from database import fetch_all, fetch_one
from .fixtures import detect_double_gameweeks, detect_blank_gameweeks, get_fixture_ticker


@dataclass
class ChipTiming:
    """Recommendation for when to use a chip."""
    chip_name: str
    recommended_gw: int | None
    confidence: str  # "high", "medium", "low"
    reasoning: str
    alternative_gws: list[int]
    key_factors: list[str]


@dataclass
class ChipCalendar:
    """Season-long chip planning."""
    triple_captain: ChipTiming
    bench_boost: ChipTiming
    free_hit: ChipTiming
    wildcard: ChipTiming
    overall_strategy: str


async def optimize_triple_captain(remaining_gws: int = 10) -> ChipTiming:
    """Find optimal gameweek for Triple Captain chip."""
    dgws = await detect_double_gameweeks()
    fixture_ticker = await get_fixture_ticker(remaining_gws)

    recommended_gw = None
    confidence = "low"
    reasoning = "No clear DGW opportunity found"
    alternatives = []
    factors = []

    # Priority 1: DGW with premium players having easy fixtures
    if dgws:
        for dgw in dgws:
            gw = dgw.gameweek
            teams_in_dgw = {t["team_id"] for t in dgw.teams}

            # Check if top 6 teams have DGW
            top_teams = {"MCI", "ARS", "LIV", "CHE", "MUN", "TOT"}
            dgw_team_names = {t["short_name"] for t in dgw.teams}

            if dgw_team_names & top_teams:
                if not recommended_gw:
                    recommended_gw = gw
                    confidence = "high"
                    reasoning = f"DGW{gw} includes premium assets from {dgw_team_names & top_teams}"
                    factors.append(f"DGW with top teams: {dgw_team_names & top_teams}")
                else:
                    alternatives.append(gw)

    # Priority 2: Best single GW fixtures for premium players
    if not recommended_gw:
        # Find GW with best fixtures for top teams
        best_gw = None
        best_score = 0

        for team in fixture_ticker:
            if team["short_name"] in {"MCI", "ARS", "LIV"}:
                for fix in team["fixtures"]:
                    if fix["fdr"] <= 2:  # Easy fixture
                        score = (3 - fix["fdr"]) * (2 if fix["home"] else 1)
                        if score > best_score:
                            best_score = score
                            best_gw = fix["gw"]

        if best_gw:
            recommended_gw = best_gw
            confidence = "medium"
            reasoning = f"GW{best_gw} has favorable fixtures for premium assets"
            factors.append("Premium player fixtures align")

    return ChipTiming(
        chip_name="Triple Captain",
        recommended_gw=recommended_gw,
        confidence=confidence,
        reasoning=reasoning,
        alternative_gws=alternatives[:3],
        key_factors=factors
    )


async def optimize_bench_boost(remaining_gws: int = 10) -> ChipTiming:
    """Find optimal gameweek for Bench Boost chip."""
    dgws = await detect_double_gameweeks()

    recommended_gw = None
    confidence = "low"
    reasoning = "No optimal DGW found for Bench Boost"
    alternatives = []
    factors = []

    # Bench Boost is best in DGWs where 15 players have 2 games
    if dgws:
        for dgw in dgws:
            gw = dgw.gameweek
            num_dgw_teams = len(dgw.teams)

            # Ideal BB: 8+ teams in DGW (allows full 15 players to have doubles)
            if num_dgw_teams >= 8:
                if not recommended_gw:
                    recommended_gw = gw
                    confidence = "high"
                    reasoning = f"DGW{gw} has {num_dgw_teams} teams with doubles - ideal for Bench Boost"
                    factors.append(f"{num_dgw_teams} teams in DGW")
                else:
                    alternatives.append(gw)
            elif num_dgw_teams >= 5:
                if not recommended_gw:
                    recommended_gw = gw
                    confidence = "medium"
                    reasoning = f"DGW{gw} has {num_dgw_teams} teams - usable for Bench Boost"
                    factors.append(f"{num_dgw_teams} teams in DGW")
                else:
                    alternatives.append(gw)

    # Factors for good BB
    factors.append("Requires healthy squad of 15 playing players")
    factors.append("All bench players should have favorable fixtures")

    return ChipTiming(
        chip_name="Bench Boost",
        recommended_gw=recommended_gw,
        confidence=confidence,
        reasoning=reasoning,
        alternative_gws=alternatives[:3],
        key_factors=factors
    )


async def optimize_free_hit(remaining_gws: int = 10) -> ChipTiming:
    """Find optimal gameweek for Free Hit chip."""
    bgws = await detect_blank_gameweeks()
    dgws = await detect_double_gameweeks()

    recommended_gw = None
    confidence = "low"
    reasoning = "No significant BGW found for Free Hit"
    alternatives = []
    factors = []

    # Free Hit is best for BGWs with many blanks
    if bgws:
        for bgw in bgws:
            gw = bgw.gameweek
            num_blanks = len(bgw.teams_with_blanks)

            # Significant BGW: 5+ teams blank
            if num_blanks >= 5:
                if not recommended_gw:
                    recommended_gw = gw
                    confidence = "high"
                    reasoning = f"BGW{gw} has {num_blanks} teams blanking - ideal for Free Hit"
                    factors.append(f"{num_blanks} teams without fixtures")
                    factors.append("Free Hit allows picking from playing teams only")
                else:
                    alternatives.append(gw)
            elif num_blanks >= 3:
                if not recommended_gw:
                    recommended_gw = gw
                    confidence = "medium"
                    reasoning = f"BGW{gw} has {num_blanks} teams blanking"
                else:
                    alternatives.append(gw)

    # Alternative: Use FH on a big DGW to load up
    if not recommended_gw and dgws:
        largest_dgw = max(dgws, key=lambda x: len(x.teams))
        if len(largest_dgw.teams) >= 8:
            recommended_gw = largest_dgw.gameweek
            confidence = "medium"
            reasoning = f"DGW{largest_dgw.gameweek} - Free Hit to maximize DGW coverage"
            factors.append(f"{len(largest_dgw.teams)} teams in DGW")

    return ChipTiming(
        chip_name="Free Hit",
        recommended_gw=recommended_gw,
        confidence=confidence,
        reasoning=reasoning,
        alternative_gws=alternatives[:3],
        key_factors=factors
    )


async def optimize_wildcard(remaining_gws: int = 10) -> ChipTiming:
    """Find optimal gameweek for Wildcard chip."""
    dgws = await detect_double_gameweeks()
    fixture_ticker = await get_fixture_ticker(remaining_gws)

    recommended_gw = None
    confidence = "medium"
    reasoning = "Use wildcard when squad needs significant restructuring"
    alternatives = []
    factors = []

    # Wildcard best used before DGWs or fixture swings
    if dgws:
        for dgw in dgws:
            # Recommend WC 1-2 weeks before a big DGW
            wc_gw = dgw.gameweek - 1
            if wc_gw > 0:
                if not recommended_gw:
                    recommended_gw = wc_gw
                    confidence = "high"
                    reasoning = f"GW{wc_gw} - Wildcard before DGW{dgw.gameweek} to set up optimal team"
                    factors.append(f"Prepare for DGW{dgw.gameweek}")
                else:
                    alternatives.append(wc_gw)

    # Look for fixture swings
    if not recommended_gw:
        # Find teams whose fixtures improve dramatically
        for team in fixture_ticker:
            fixtures = team["fixtures"]
            if len(fixtures) >= 4:
                early_avg = sum(f["fdr"] for f in fixtures[:2]) / 2
                later_avg = sum(f["fdr"] for f in fixtures[2:4]) / 2

                if early_avg - later_avg > 1.5:  # Fixtures get harder
                    factors.append(f"{team['short_name']}: fixtures worsen")
                elif later_avg - early_avg > 1.5:  # Fixtures improve
                    factors.append(f"{team['short_name']}: fixtures improve")

    factors.append("Consider using when 4+ transfers needed")
    factors.append("Best before a favorable fixture swing")

    return ChipTiming(
        chip_name="Wildcard",
        recommended_gw=recommended_gw,
        confidence=confidence,
        reasoning=reasoning,
        alternative_gws=alternatives[:3],
        key_factors=factors
    )


async def get_chip_calendar(remaining_gws: int = 15) -> ChipCalendar:
    """Get full season chip planning calendar."""
    tc = await optimize_triple_captain(remaining_gws)
    bb = await optimize_bench_boost(remaining_gws)
    fh = await optimize_free_hit(remaining_gws)
    wc = await optimize_wildcard(remaining_gws)

    # Build overall strategy
    chips_by_gw = []
    if tc.recommended_gw:
        chips_by_gw.append((tc.recommended_gw, "TC"))
    if bb.recommended_gw:
        chips_by_gw.append((bb.recommended_gw, "BB"))
    if fh.recommended_gw:
        chips_by_gw.append((fh.recommended_gw, "FH"))
    if wc.recommended_gw:
        chips_by_gw.append((wc.recommended_gw, "WC"))

    chips_by_gw.sort(key=lambda x: x[0])

    if chips_by_gw:
        strategy_parts = [f"GW{gw}: {chip}" for gw, chip in chips_by_gw]
        overall_strategy = "Recommended order: " + " → ".join(strategy_parts)
    else:
        overall_strategy = "No clear chip opportunities identified. Monitor fixture announcements."

    return ChipCalendar(
        triple_captain=tc,
        bench_boost=bb,
        free_hit=fh,
        wildcard=wc,
        overall_strategy=overall_strategy
    )


async def analyze_dgw_bgw() -> dict:
    """Get comprehensive DGW/BGW analysis."""
    dgws = await detect_double_gameweeks()
    bgws = await detect_blank_gameweeks()

    dgw_summary = []
    for dgw in dgws:
        dgw_summary.append({
            "gameweek": dgw.gameweek,
            "teams": [{"name": t["team_name"], "short": t["short_name"]} for t in dgw.teams],
            "num_teams": len(dgw.teams)
        })

    bgw_summary = []
    for bgw in bgws:
        bgw_summary.append({
            "gameweek": bgw.gameweek,
            "teams_blanking": [{"name": t["team_name"], "short": t["short_name"]} for t in bgw.teams_with_blanks],
            "num_blanks": len(bgw.teams_with_blanks)
        })

    return {
        "double_gameweeks": dgw_summary,
        "blank_gameweeks": bgw_summary,
        "chip_implications": {
            "triple_captain": "Best used in DGW with premium players having doubles",
            "bench_boost": "Best in DGW where all 15 players have doubles",
            "free_hit": "Best in BGW to field 11 playing players",
            "wildcard": "Use before DGW to set up optimal squad"
        }
    }
