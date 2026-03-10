import torch

from ml.belief_decoder import decode_hidden_card_assignment
from ml.neurosymbolic_dataset import OWNER_LEFT_HAND, OWNER_PARTNER_HAND, OWNER_RIGHT_HAND
from ml.neurosymbolic_state import CanonicalState


def decoder_payload() -> dict:
    cards = []
    cards.append(
        {
            "card_idx": 0,
            "suit_idx": 0,
            "value_idx": 0,
            "point_value": 0,
            "exact_location": "MyHand",
            "possible_hidden_rel": [],
            "confirmed_hidden_rel": None,
            "impossible_hidden_rel": [1, 2, 3],
            "symbolically_resolved": True,
            "standing_for_my_team": None,
        }
    )
    for idx in range(1, 4):
        cards.append(
            {
                "card_idx": idx,
                "suit_idx": 0,
                "value_idx": idx,
                "point_value": 0,
                "exact_location": None,
                "possible_hidden_rel": [1, 2, 3],
                "confirmed_hidden_rel": None,
                "impossible_hidden_rel": [],
                "symbolically_resolved": False,
                "standing_for_my_team": None,
            }
        )
    for idx in range(4, 36):
        cards.append(
            {
                "card_idx": idx,
                "suit_idx": idx // 9,
                "value_idx": idx % 9,
                "point_value": 0,
                "exact_location": "Played",
                "possible_hidden_rel": [],
                "confirmed_hidden_rel": None,
                "impossible_hidden_rel": [1, 2, 3],
                "symbolically_resolved": True,
                "standing_for_my_team": None,
            }
        )
    return {
        "schema_version": 1,
        "global": {
            "pov_abs_seat": 0,
            "active_abs_seat": 1,
            "active_rel_seat": 1,
            "phase": "Trick",
            "trick_number": 8,
            "trick_position": 0,
            "trump": None,
            "trump_announced": [False, False, False, False],
            "current_contract_value": 120,
            "legal_action_count": 2,
            "player_at_turn_cards_remaining": 1,
        },
        "cards": cards,
        "players": [
            {
                "relative_seat": 0,
                "absolute_seat": 0,
                "is_self": True,
                "is_partner": False,
                "cards_remaining": 1,
                "confirmed_cards": [0],
                "possible_cards": [0],
                "void_suits": [False, True, True, True],
                "required_half_suits": [False, False, False, False],
                "required_pair_suits": [False, False, False, False],
            },
            {
                "relative_seat": 1,
                "absolute_seat": 1,
                "is_self": False,
                "is_partner": False,
                "cards_remaining": 1,
                "confirmed_cards": [],
                "possible_cards": [1, 2, 3],
                "void_suits": [False, False, False, False],
                "required_half_suits": [False, False, False, False],
                "required_pair_suits": [False, False, False, False],
            },
            {
                "relative_seat": 2,
                "absolute_seat": 2,
                "is_self": False,
                "is_partner": True,
                "cards_remaining": 1,
                "confirmed_cards": [],
                "possible_cards": [1, 2, 3],
                "void_suits": [False, False, False, False],
                "required_half_suits": [False, False, False, False],
                "required_pair_suits": [False, False, False, False],
            },
            {
                "relative_seat": 3,
                "absolute_seat": 3,
                "is_self": False,
                "is_partner": False,
                "cards_remaining": 1,
                "confirmed_cards": [],
                "possible_cards": [1, 2, 3],
                "void_suits": [False, False, False, False],
                "required_half_suits": [False, False, False, False],
                "required_pair_suits": [False, False, False, False],
            },
        ],
        "teams": [
            {"team_idx": 0, "points": 0, "secured_point_floor": 0, "max_reachable_points": 30},
            {"team_idx": 1, "points": 0, "secured_point_floor": 0, "max_reachable_points": 30},
        ],
        "strategy": {
            "standing_card_indices": [],
            "exhausted_pair_suits": [False, False, False, False],
            "current_trump_suit": None,
            "trump_called_count": 0,
            "visible_pair_points_floor": 0,
            "visible_pair_points_ceiling": 0,
            "makeable_bid_floor": 120,
            "makeable_bid_ceiling": 120,
        },
        "belief_targets": {
            "schema_version": 1,
            "card_owner_classes": ["SelfHand", "LeftHand", "PartnerHand", "RightHand"] + ["Played"] * 32,
            "hidden_card_indices": [1, 2, 3],
            "player_void_suits": [[False, True, True, True]] * 4,
            "player_has_half_suits": [[False, False, False, False]] * 4,
            "player_has_pair_suits": [[False, False, False, False]] * 4,
        },
    }


def test_decode_hidden_card_assignment_respects_global_capacities():
    state = CanonicalState.from_dict(decoder_payload())
    logits = torch.zeros(36, 9)
    logits[1, OWNER_LEFT_HAND] = 10.0
    logits[2, OWNER_PARTNER_HAND] = 10.0
    logits[3, OWNER_RIGHT_HAND] = 10.0

    decoded = decode_hidden_card_assignment(logits, state)
    assert decoded.card_owner_classes[1] == OWNER_LEFT_HAND
    assert decoded.card_owner_classes[2] == OWNER_PARTNER_HAND
    assert decoded.card_owner_classes[3] == OWNER_RIGHT_HAND


def test_decode_hidden_card_assignment_falls_back_to_unknown_when_caps_infeasible():
    payload = decoder_payload()
    payload["players"][1]["cards_remaining"] = 0
    payload["players"][2]["cards_remaining"] = 0
    payload["players"][3]["cards_remaining"] = 0
    state = CanonicalState.from_dict(payload)
    logits = torch.zeros(36, 9)
    decoded = decode_hidden_card_assignment(logits, state)
    assert decoded.card_owner_classes[1] == -1
    assert decoded.card_owner_classes[2] == -1
    assert decoded.card_owner_classes[3] == -1
