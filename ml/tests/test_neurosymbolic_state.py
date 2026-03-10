from ml.neurosymbolic_state import (
    CANONICAL_STATE_SCHEMA_VERSION,
    CanonicalState,
)


def sample_payload() -> dict:
    return {
        "schema_version": CANONICAL_STATE_SCHEMA_VERSION,
        "global": {
            "pov_abs_seat": 0,
            "active_abs_seat": 1,
            "active_rel_seat": 1,
            "phase": "Bidding",
            "trick_number": 1,
            "trick_position": 0,
            "trump": None,
            "trump_announced": [False, False, False, False],
            "current_contract_value": 140,
            "legal_action_count": 3,
            "player_at_turn_cards_remaining": 9,
        },
        "cards": [
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
            },
            {
                "card_idx": 1,
                "suit_idx": 0,
                "value_idx": 1,
                "point_value": 0,
                "exact_location": None,
                "possible_hidden_rel": [1, 2],
                "confirmed_hidden_rel": None,
                "impossible_hidden_rel": [3],
                "symbolically_resolved": False,
                "standing_for_my_team": None,
            },
        ]
        + [
            {
                "card_idx": idx,
                "suit_idx": idx // 9,
                "value_idx": idx % 9,
                "point_value": 0,
                "exact_location": None,
                "possible_hidden_rel": [1, 2, 3],
                "confirmed_hidden_rel": None,
                "impossible_hidden_rel": [],
                "symbolically_resolved": False,
                "standing_for_my_team": None,
            }
            for idx in range(2, 36)
        ],
        "players": [
            {
                "relative_seat": 0,
                "absolute_seat": 0,
                "is_self": True,
                "is_partner": False,
                "cards_remaining": 9,
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
                "cards_remaining": 9,
                "confirmed_cards": [],
                "possible_cards": [1],
                "void_suits": [False, False, False, False],
                "required_half_suits": [False, False, False, False],
                "required_pair_suits": [False, False, False, False],
            },
            {
                "relative_seat": 2,
                "absolute_seat": 2,
                "is_self": False,
                "is_partner": True,
                "cards_remaining": 9,
                "confirmed_cards": [1],
                "possible_cards": [1],
                "void_suits": [False, False, False, False],
                "required_half_suits": [True, False, False, False],
                "required_pair_suits": [False, False, False, False],
            },
            {
                "relative_seat": 3,
                "absolute_seat": 3,
                "is_self": False,
                "is_partner": False,
                "cards_remaining": 9,
                "confirmed_cards": [],
                "possible_cards": [1],
                "void_suits": [True, False, False, False],
                "required_half_suits": [False, False, False, False],
                "required_pair_suits": [False, False, False, False],
            },
        ],
        "teams": [
            {
                "team_idx": 0,
                "points": 20,
                "secured_point_floor": 20,
                "max_reachable_points": 120,
            },
            {
                "team_idx": 1,
                "points": 10,
                "secured_point_floor": 10,
                "max_reachable_points": 110,
            },
        ],
        "strategy": {
            "standing_card_indices": [0],
            "exhausted_pair_suits": [False, False, False, False],
            "current_trump_suit": None,
            "trump_called_count": 0,
            "visible_pair_points_floor": 0,
            "visible_pair_points_ceiling": 40,
            "makeable_bid_floor": 120,
            "makeable_bid_ceiling": 145,
        },
        "belief_targets": {
            "schema_version": CANONICAL_STATE_SCHEMA_VERSION,
            "card_owner_classes": ["SelfHand", "PartnerHand"] + ["LeftHand"] * 34,
            "hidden_card_indices": [1] + list(range(2, 36)),
            "player_void_suits": [
                [False, True, True, True],
                [False, False, False, False],
                [False, False, False, False],
                [True, False, False, False],
            ],
            "player_has_half_suits": [
                [False, False, False, False],
                [False, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ],
            "player_has_pair_suits": [
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
        },
    }


def test_canonical_state_from_dict_parses_sections():
    state = CanonicalState.from_dict(sample_payload())
    assert state.schema_version == CANONICAL_STATE_SCHEMA_VERSION
    assert state.global_state.phase == "Bidding"
    assert len(state.cards) == 36
    assert len(state.players) == 4
    assert len(state.teams) == 2


def test_canonical_state_helpers_expose_unknown_and_partner_cards():
    state = CanonicalState.from_dict(sample_payload())
    assert 1 in state.unknown_card_indices
    assert state.confirmed_partner_cards == (1,)
    assert state.belief_targets is not None
    assert state.belief_targets.card_owner_classes[1] == "PartnerHand"


def test_from_record_merges_top_level_belief_targets():
    payload = sample_payload()
    belief_targets = payload.pop("belief_targets")
    state = CanonicalState.from_record(
        {
            "canonical_state": payload,
            "belief_targets": belief_targets,
        }
    )
    assert state.belief_targets is not None
    assert state.belief_targets.hidden_card_indices == tuple(belief_targets["hidden_card_indices"])
