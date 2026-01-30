"""Optimization modules using linear programming."""

from .dream_team import (
    build_optimal_squad,
    build_free_hit_team,
    optimize_starting_11,
    suggest_wildcard_team,
)
from .transfers import (
    get_transfer_suggestions,
    plan_transfers,
    get_best_transfers_by_position,
)
from .captaincy import (
    get_captaincy_picks,
    analyze_captain_options,
)

__all__ = [
    # Dream Team
    "build_optimal_squad",
    "build_free_hit_team",
    "optimize_starting_11",
    "suggest_wildcard_team",
    # Transfers
    "get_transfer_suggestions",
    "plan_transfers",
    "get_best_transfers_by_position",
    # Captaincy
    "get_captaincy_picks",
    "analyze_captain_options",
]
