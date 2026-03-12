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
                {"action_token": 41, "bid_value": 145, "card_idx": None, "suit_idx": None},
                {"action_token": 42, "bid_value": None, "card_idx": None, "suit_idx": None},
            ],
        },
        "action_taken": 0,
        "outcome_pts_my_team": 96,
        "outcome_pts_opp": 44,
        "pov_player_winrate": 0.62,
    }


def sample_passing_record() -> dict:
    payload = sample_payload()
    my_hand = [0, 5, 6, 8, 10, 14, 15, 17, 26]
    for card in payload["cards"]:
        if card["card_idx"] in my_hand:
            card["exact_location"] = "MyHand"
            card["symbolically_resolved"] = True
            card["possible_hidden_rel"] = []
            card["impossible_hidden_rel"] = [1, 2, 3]
        elif card["card_idx"] not in (1,):
            card["exact_location"] = None
            card["symbolically_resolved"] = False
            card["possible_hidden_rel"] = [1, 2, 3]
            card["impossible_hidden_rel"] = []
    payload["global"]["phase"] = "PassingForth"
    payload["global"]["player_at_turn_cards_remaining"] = len(my_hand)
    payload["players"][0]["cards_remaining"] = len(my_hand)
    payload["players"][0]["confirmed_cards"] = my_hand
    payload["players"][0]["possible_cards"] = my_hand
    payload["players"][0]["void_suits"] = [False, False, False, False]
    return {
        "canonical_state": payload,
        "obs": {
            "schema_version": 1,
            "my_hand_bitmask": [idx in set(my_hand) for idx in range(36)],
            "possible_bitmasks": [[False] * 36, [False] * 36, [False] * 36],
            "confirmed_bitmasks": [[False] * 36, [False] * 36, [False] * 36],
            "current_trick_indices": [],
            "cards_remaining": [len(my_hand), 9, 9, 9],
            "trump": None,
            "trump_announced": [False, False, False, False],
            "trump_possibilities": 0,
            "my_role": 4,
            "trick_position": 0,
            "trick_number": 1,
            "points_my_team": 20,
            "points_opp_team": 10,
            "last_trick_bonus_live": False,
            "active_player": 0,
            "phase": "PassingForth",
            "event_tokens": [1, 2, 3],
            "legal_actions": [
                {"action_token": 43, "pass_cards": [0, 8, 10, 14], "card_idx": None, "suit_idx": None},
                {"action_token": 43, "pass_cards": [5, 6, 15, 17], "card_idx": None, "suit_idx": None},
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
    assert tuple(features.global_features.shape) == (30,)
    assert tuple(features.action_features.shape) == (2, 87)
    assert tuple(features.action_mask.shape) == (2,)


def test_build_decision_targets_from_record_uses_winrate_weighting():
    targets = build_decision_targets_from_record(sample_record())
    assert targets.task == "bidding"
    assert targets.policy_idx == 0
    assert tuple(targets.aux_targets.shape) == (10,)
    assert targets.sample_weight > 1.5
    assert targets.teacher_policy is not None
    assert abs(float(targets.teacher_policy.sum().item()) - 1.0) < 1e-6


def test_build_passing_targets_uses_teacher_policy_and_aux_targets():
    targets = build_decision_targets_from_record(sample_passing_record())
    assert targets.task == "passing"
    assert tuple(targets.aux_targets.shape) == (8,)
    assert targets.teacher_policy is not None
    assert abs(float(targets.teacher_policy.sum().item()) - 1.0) < 1e-6
    assert int(targets.teacher_policy.argmax().item()) == 0


def test_build_decision_targets_prefers_contract_evaluated_points_when_present():
    record = sample_record()
    record["canonical_state"]["global"]["phase"] = "Playing"
    record["obs"]["phase"] = "Playing"
    record["outcome_pts_my_team"] = 158
    record["outcome_pts_opp"] = 62
    record["outcome_eval_pts_my_team"] = -140
    record["outcome_eval_pts_opp"] = 62
    targets = build_decision_targets_from_record(record)
    assert targets.task == "playing"
    assert targets.value_target < 0.0
    assert float(targets.aux_targets[-1].item()) < 0.0


def test_task_from_phase_name_normalizes_parameterized_answer_phases():
    assert task_from_phase_name("AnsweringHalf(Acorns)") == "playing"
    assert task_from_phase_name("AnsweringPair") == "playing"
