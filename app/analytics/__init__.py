"""Analytics modules for FPL data analysis."""

from .expected_points import (
    calculate_expected_points,
    get_top_xp_players,
    calculate_clean_sheet_probability,
)
from .form import (
    calculate_player_form,
    calculate_team_form,
    detect_form_momentum,
)
from .fixtures import (
    get_fixture_difficulty,
    get_easiest_fixtures,
    get_hardest_fixtures,
    detect_double_gameweeks,
    detect_blank_gameweeks,
)
from .price_prediction import (
    predict_price_change,
    get_price_risers,
    get_price_fallers,
)
from .luck import (
    analyze_luck,
    get_overperformers,
    get_underperformers,
)
from .ownership import (
    get_template_players,
    get_differentials,
    calculate_template_score,
)
from .minutes import (
    predict_minutes,
    get_rotation_risks,
    get_nailed_players,
)
from .hits import (
    evaluate_hit,
    evaluate_multiple_hits,
    get_worth_hit_transfers,
)
from .chips import (
    optimize_triple_captain,
    optimize_bench_boost,
    optimize_free_hit,
    get_chip_calendar,
)
from .venue import (
    get_home_away_splits,
    get_best_home_performers,
    get_best_away_performers,
)

__all__ = [
    # Expected Points
    "calculate_expected_points",
    "get_top_xp_players",
    "calculate_clean_sheet_probability",
    # Form
    "calculate_player_form",
    "calculate_team_form",
    "detect_form_momentum",
    # Fixtures
    "get_fixture_difficulty",
    "get_easiest_fixtures",
    "get_hardest_fixtures",
    "detect_double_gameweeks",
    "detect_blank_gameweeks",
    # Price Prediction
    "predict_price_change",
    "get_price_risers",
    "get_price_fallers",
    # Luck
    "analyze_luck",
    "get_overperformers",
    "get_underperformers",
    # Ownership
    "get_template_players",
    "get_differentials",
    "calculate_template_score",
    # Minutes
    "predict_minutes",
    "get_rotation_risks",
    "get_nailed_players",
    # Hits
    "evaluate_hit",
    "evaluate_multiple_hits",
    "get_worth_hit_transfers",
    # Chips
    "optimize_triple_captain",
    "optimize_bench_boost",
    "optimize_free_hit",
    "get_chip_calendar",
    # Venue
    "get_home_away_splits",
    "get_best_home_performers",
    "get_best_away_performers",
]
