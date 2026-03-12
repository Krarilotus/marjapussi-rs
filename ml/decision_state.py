from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from ml.four_model_phase import task_from_phase_name
    from ml.env import obs_to_tensors
    from ml.neurosymbolic_dataset import (
        NUM_OWNER_CLASSES,
        OWNER_NAME_TO_INDEX,
        build_belief_features,
        build_belief_targets,
    )
    from ml.neurosymbolic_state import CanonicalState
except ModuleNotFoundError:
    from four_model_phase import task_from_phase_name
    from env import obs_to_tensors
    from neurosymbolic_dataset import (
        NUM_OWNER_CLASSES,
        OWNER_NAME_TO_INDEX,
        build_belief_features,
        build_belief_targets,
    )
    from neurosymbolic_state import CanonicalState


TASK_TO_PHASE_INDEX = {"bidding": 0, "passing": 1, "playing": 2}

TASK_AUX_TARGET_NAMES = {
    "bidding": ("makeable_bid_floor", "makeable_bid_ceiling", "win_signal"),
    "passing": ("standing_cards", "pair_points_ceiling", "point_diff"),
    "playing": ("standing_cards", "secured_point_floor", "point_diff"),
}


@dataclass(frozen=True)
class DecisionFeatures:
    task: str
    card_features: torch.Tensor
    player_features: torch.Tensor
    global_features: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    action_idx: int


@dataclass(frozen=True)
class DecisionTargets:
    task: str
    policy_idx: int
    value_target: float
    aux_targets: torch.Tensor
    sample_weight: float


def _teacher_belief_onehot(state: CanonicalState) -> torch.Tensor:
    targets = build_belief_targets(state)
    rows = []
    for owner_idx in targets.card_owner_targets:
        row = [0.0] * NUM_OWNER_CLASSES
        if owner_idx >= 0:
            row[owner_idx] = 1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)
def build_decision_features_from_record(
    record: dict,
    use_teacher_belief: bool = True,
    belief_owner_override: torch.Tensor | None = None,
) -> DecisionFeatures:
    state = CanonicalState.from_record(record)
    obs = record["obs"]
    obs_tensors = obs_to_tensors(obs)
    base = build_belief_features(state)

    if belief_owner_override is not None:
        belief_owner = belief_owner_override.to(dtype=torch.float32)
    elif use_teacher_belief and state.belief_targets is not None:
        belief_owner = _teacher_belief_onehot(state)
    else:
        belief_owner = torch.zeros((36, NUM_OWNER_CLASSES), dtype=torch.float32)

    task = task_from_phase_name(state.global_state.phase)
    phase_oh = torch.zeros(3, dtype=torch.float32)
    phase_oh[{"bidding": 0, "passing": 1, "playing": 2}[task]] = 1.0

    global_features = torch.cat([base.global_features, phase_oh], dim=0)
    card_features = torch.cat([base.card_features, belief_owner], dim=1)

    return DecisionFeatures(
        task=task,
        card_features=card_features,
        player_features=base.player_features,
        global_features=global_features,
        action_features=obs_tensors["action_feats"].squeeze(0),
        action_mask=obs_tensors["action_mask"].squeeze(0),
        action_idx=int(record.get("action_taken", 0)),
    )


def _normalize_points(value: float) -> float:
    return max(-1.0, min(1.0, value / 420.0))


def _normalize_bid(value: int) -> float:
    return max(0.0, min(1.0, value / 420.0))


def _quality_weight_from_record(record: dict) -> float:
    winrate = float(record.get("pov_player_winrate", 0.5))
    winrate = max(0.0, min(1.0, winrate))
    weight = 0.5 + winrate
    if winrate >= 0.55:
        weight += 0.5
    return weight


def build_decision_targets_from_record(record: dict) -> DecisionTargets:
    state = CanonicalState.from_record(record)
    task = task_from_phase_name(state.global_state.phase)
    my_points = float(record.get("outcome_pts_my_team", 0.0))
    opp_points = float(record.get("outcome_pts_opp", 0.0))
    point_diff = my_points - opp_points
    win_signal = 1.0 if point_diff > 0.0 else 0.0
    standing_norm = len(state.strategy.standing_card_indices) / 36.0
    secured_floor_norm = _normalize_points(state.teams[0].secured_point_floor)
    pair_ceiling_norm = max(
        0.0,
        min(1.0, state.strategy.visible_pair_points_ceiling / 120.0),
    )

    if task == "bidding":
        aux_values = (
            _normalize_bid(state.strategy.makeable_bid_floor),
            _normalize_bid(state.strategy.makeable_bid_ceiling),
            win_signal,
        )
    elif task == "passing":
        aux_values = (
            standing_norm,
            pair_ceiling_norm,
            _normalize_points(point_diff),
        )
    else:
        aux_values = (
            standing_norm,
            secured_floor_norm,
            _normalize_points(point_diff),
        )

    return DecisionTargets(
        task=task,
        policy_idx=int(record.get("action_taken", 0)),
        value_target=_normalize_points(point_diff),
        aux_targets=torch.tensor(aux_values, dtype=torch.float32),
        sample_weight=_quality_weight_from_record(record),
    )
