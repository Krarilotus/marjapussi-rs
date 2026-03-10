from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorEvalSummary:
    point_diff: float
    pass_game_rate: float
    avg_bid: float
    contract_made_rate: float
    trump_call_rate: float


@dataclass(frozen=True)
class BehaviorScoreBreakdown:
    total: float
    point_term: float
    pass_penalty: float
    minimal_bid_penalty: float
    contract_bonus: float
    trump_bonus: float


def compute_behavior_score(
    summary: BehaviorEvalSummary,
    *,
    pass_penalty_weight: float = 80.0,
    minimal_bid_floor: float = 130.0,
    minimal_bid_penalty_weight: float = 0.60,
    contract_bonus_weight: float = 20.0,
    trump_bonus_weight: float = 8.0,
) -> BehaviorScoreBreakdown:
    pass_penalty = pass_penalty_weight * max(0.0, min(1.0, summary.pass_game_rate))
    minimal_bid_penalty = minimal_bid_penalty_weight * max(0.0, minimal_bid_floor - summary.avg_bid)
    contract_bonus = contract_bonus_weight * max(0.0, min(1.0, summary.contract_made_rate))
    trump_bonus = trump_bonus_weight * max(0.0, min(1.0, summary.trump_call_rate))
    total = summary.point_diff - pass_penalty - minimal_bid_penalty + contract_bonus + trump_bonus
    return BehaviorScoreBreakdown(
        total=total,
        point_term=summary.point_diff,
        pass_penalty=pass_penalty,
        minimal_bid_penalty=minimal_bid_penalty,
        contract_bonus=contract_bonus,
        trump_bonus=trump_bonus,
    )
