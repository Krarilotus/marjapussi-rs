from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from ml.neurosymbolic_state import CanonicalState
except ModuleNotFoundError:
    from neurosymbolic_state import CanonicalState


BIDDING_RULE_TARGET_NAMES = (
    "makeable_bid_floor",
    "makeable_bid_ceiling",
    "team_has_ace_signal",
    "allow_over_140",
    "recommended_bid_step",
    "estimated_bid_value",
    "partner_first_step",
    "team_bid_count",
    "own_unmatched_halves",
    "own_big_pair_count",
)


@dataclass(frozen=True)
class BiddingTeacherTargets:
    aux_targets: torch.Tensor
    teacher_policy: torch.Tensor


def _normalize_bid(value: int) -> float:
    return max(0.0, min(1.0, value / 420.0))


def _normalize_step(value: int) -> float:
    return max(0.0, min(1.0, value / 15.0))


def build_bidding_teacher_targets(record: dict, state: CanonicalState) -> BiddingTeacherTargets:
    strategy = state.strategy
    legal_actions = list(record["obs"].get("legal_actions", []))

    aux_targets = torch.tensor(
        [
            _normalize_bid(strategy.makeable_bid_floor),
            _normalize_bid(strategy.makeable_bid_ceiling),
            float(strategy.bidding_team_has_ace_signal),
            float(strategy.bidding_allow_over_140),
            _normalize_step(strategy.bidding_recommended_step),
            _normalize_bid(strategy.bidding_estimated_value),
            _normalize_step(strategy.bidding_partner_first_step),
            max(0.0, min(1.0, strategy.bidding_team_bid_count / 4.0)),
            max(0.0, min(1.0, strategy.bidding_own_unmatched_halves / 4.0)),
            max(0.0, min(1.0, strategy.bidding_own_big_pair_count / 2.0)),
        ],
        dtype=torch.float32,
    )

    teacher_policy = torch.zeros(len(legal_actions), dtype=torch.float32)
    target_bid = None
    if strategy.bidding_recommended_step > 0:
        candidate_bid = strategy.bidding_current_highest_bid + strategy.bidding_recommended_step
        if candidate_bid <= strategy.bidding_estimated_value:
            target_bid = candidate_bid

    preferred_idx = None
    for idx, legal_action in enumerate(legal_actions):
        bid_value = legal_action.get("bid_value")
        if bid_value is None:
            continue
        if target_bid is not None and int(bid_value) == int(target_bid):
            preferred_idx = idx
            break

    if preferred_idx is None:
        for idx, legal_action in enumerate(legal_actions):
            action_token = int(legal_action.get("action_token", -1))
            if action_token == 42:
                preferred_idx = idx
                break

    if preferred_idx is not None:
        teacher_policy[preferred_idx] = 1.0
    elif len(legal_actions) > 0:
        teacher_policy.fill_(1.0 / len(legal_actions))

    return BiddingTeacherTargets(aux_targets=aux_targets, teacher_policy=teacher_policy)
