from ml.decision_state import (
    build_decision_features_from_record,
    build_decision_targets_from_record,
    task_from_phase_name,
)
from ml.tests.test_neurosymbolic_state import sample_payload


def sample_record() -> dict:
    return {
        "canonical_state": sample_payload(),
        "obs": {
            "schema_version": 1,
            "my_hand_bitmask": [True] + [False] * 35,
            "possible_bitmasks": [[False] * 36, [False] * 36, [False] * 36],
            "confirmed_bitmasks": [[False] * 36, [False] * 36, [False] * 36],
            "current_trick_indices": [],
            "cards_remaining": [1, 12, 12, 11],
            "trump": None,
            "trump_announced": [False, False, False, False],
            "trump_possibilities": 0,
            "my_role": 4,
            "trick_position": 0,
            "trick_number": 1,
            "points_my_team": 20,
            "points_opp_team": 10,
            "last_trick_bonus_live": False,
            "active_player": 1,
            "phase": "Bidding",
            "event_tokens": [1, 2, 3],
            "legal_actions": [
                {"action_token": 41, "bid_value": 140, "card_idx": None, "suit_idx": None},
                {"action_token": 42, "bid_value": None, "card_idx": None, "suit_idx": None},
            ],
        },
        "action_taken": 0,
        "outcome_pts_my_team": 96,
        "outcome_pts_opp": 44,
        "pov_player_winrate": 0.62,
    }


def test_build_decision_features_from_record_shapes():
    features = build_decision_features_from_record(sample_record(), use_teacher_belief=True)
    assert features.task == "bidding"
    assert tuple(features.card_features.shape) == (36, 32)
    assert tuple(features.player_features.shape) == (4, 18)
    assert tuple(features.global_features.shape) == (15,)
    assert tuple(features.action_features.shape) == (2, 87)
    assert tuple(features.action_mask.shape) == (2,)


def test_build_decision_targets_from_record_uses_winrate_weighting():
    targets = build_decision_targets_from_record(sample_record())
    assert targets.task == "bidding"
    assert targets.policy_idx == 0
    assert tuple(targets.aux_targets.shape) == (3,)
    assert targets.sample_weight > 1.5


def test_task_from_phase_name_normalizes_parameterized_answer_phases():
    assert task_from_phase_name("AnsweringHalf(Acorns)") == "playing"
    assert task_from_phase_name("AnsweringPair") == "playing"
