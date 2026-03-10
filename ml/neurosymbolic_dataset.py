from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

try:
    from ml.neurosymbolic_state import CanonicalState
except ModuleNotFoundError:
    from neurosymbolic_state import CanonicalState


OWNER_SELF_HAND = 0
OWNER_LEFT_HAND = 1
OWNER_PARTNER_HAND = 2
OWNER_RIGHT_HAND = 3
OWNER_TRICK_SELF = 4
OWNER_TRICK_LEFT = 5
OWNER_TRICK_PARTNER = 6
OWNER_TRICK_RIGHT = 7
OWNER_PLAYED = 8
OWNER_UNKNOWN = -1
NUM_OWNER_CLASSES = 9

OWNER_NAME_TO_INDEX = {
    "SelfHand": OWNER_SELF_HAND,
    "LeftHand": OWNER_LEFT_HAND,
    "PartnerHand": OWNER_PARTNER_HAND,
    "RightHand": OWNER_RIGHT_HAND,
    "TrickSelf": OWNER_TRICK_SELF,
    "TrickLeft": OWNER_TRICK_LEFT,
    "TrickPartner": OWNER_TRICK_PARTNER,
    "TrickRight": OWNER_TRICK_RIGHT,
    "Played": OWNER_PLAYED,
}


@dataclass(frozen=True)
class BeliefTargets:
    card_owner_targets: tuple[int, ...]
    hidden_card_mask: tuple[bool, ...]
    owner_candidate_mask: tuple[tuple[bool, ...], ...]
    player_void_targets: tuple[tuple[bool, bool, bool, bool], ...]
    player_has_half_targets: tuple[tuple[bool, bool, bool, bool], ...]
    player_has_pair_targets: tuple[tuple[bool, bool, bool, bool], ...]


@dataclass(frozen=True)
class BeliefFeatures:
    card_features: torch.Tensor
    player_features: torch.Tensor
    global_features: torch.Tensor


def _candidate_mask_from_state(state: CanonicalState, card_idx: int) -> tuple[bool, ...]:
    card = state.cards[card_idx]
    exact = card.exact_location
    if exact == "MyHand":
        return (True, False, False, False, False, False, False, False, False)
    if exact == "HiddenLeft":
        return (False, True, False, False, False, False, False, False, False)
    if exact == "HiddenPartner":
        return (False, False, True, False, False, False, False, False, False)
    if exact == "HiddenRight":
        return (False, False, False, True, False, False, False, False, False)
    if isinstance(exact, Mapping) and "CurrentTrick" in exact:
        rel = int(exact["CurrentTrick"]["relative_seat"])
        trick_slot = {
            0: OWNER_TRICK_SELF,
            1: OWNER_TRICK_LEFT,
            2: OWNER_TRICK_PARTNER,
            3: OWNER_TRICK_RIGHT,
        }[rel]
        return tuple(idx == trick_slot for idx in range(NUM_OWNER_CLASSES))
    if exact == "Played":
        return (False, False, False, False, False, False, False, False, True)

    mask = [False] * NUM_OWNER_CLASSES
    for rel in card.possible_hidden_rel:
        if rel == 1:
            mask[OWNER_LEFT_HAND] = True
        elif rel == 2:
            mask[OWNER_PARTNER_HAND] = True
        elif rel == 3:
            mask[OWNER_RIGHT_HAND] = True
    return tuple(mask)


def build_belief_targets(state: CanonicalState) -> BeliefTargets:
    if state.belief_targets is None:
        raise ValueError("canonical_state record does not contain belief_targets")

    target_owner = tuple(
        OWNER_NAME_TO_INDEX[name] for name in state.belief_targets.card_owner_classes
    )
    hidden_mask = tuple(card_idx in set(state.belief_targets.hidden_card_indices) for card_idx in range(36))
    candidate_mask = tuple(_candidate_mask_from_state(state, card_idx) for card_idx in range(36))

    return BeliefTargets(
        card_owner_targets=target_owner,
        hidden_card_mask=hidden_mask,
        owner_candidate_mask=candidate_mask,
        player_void_targets=tuple(tuple(row) for row in state.belief_targets.player_void_suits),
        player_has_half_targets=tuple(tuple(row) for row in state.belief_targets.player_has_half_suits),
        player_has_pair_targets=tuple(tuple(row) for row in state.belief_targets.player_has_pair_suits),
    )


def build_belief_features(state: CanonicalState) -> BeliefFeatures:
    card_rows: list[list[float]] = []
    for card in state.cards:
        exact = card.exact_location
        exact_oh = [0.0] * NUM_OWNER_CLASSES
        if exact == "MyHand":
            exact_oh[OWNER_SELF_HAND] = 1.0
        elif exact == "HiddenLeft":
            exact_oh[OWNER_LEFT_HAND] = 1.0
        elif exact == "HiddenPartner":
            exact_oh[OWNER_PARTNER_HAND] = 1.0
        elif exact == "HiddenRight":
            exact_oh[OWNER_RIGHT_HAND] = 1.0
        elif isinstance(exact, Mapping) and "CurrentTrick" in exact:
            rel = int(exact["CurrentTrick"]["relative_seat"])
            exact_oh[{0: OWNER_TRICK_SELF, 1: OWNER_TRICK_LEFT, 2: OWNER_TRICK_PARTNER, 3: OWNER_TRICK_RIGHT}[rel]] = 1.0
        elif exact == "Played":
            exact_oh[OWNER_PLAYED] = 1.0

        possible_hidden = [0.0, 0.0, 0.0]
        impossible_hidden = [0.0, 0.0, 0.0]
        for rel in card.possible_hidden_rel:
            possible_hidden[rel - 1] = 1.0
        for rel in card.impossible_hidden_rel:
            impossible_hidden[rel - 1] = 1.0
        confirmed_hidden = [0.0, 0.0, 0.0]
        if card.confirmed_hidden_rel is not None:
            confirmed_hidden[card.confirmed_hidden_rel - 1] = 1.0
        standing = 0.0 if card.standing_for_my_team is None else (1.0 if card.standing_for_my_team else -1.0)

        card_rows.append(
            [
                card.suit_idx / 3.0,
                card.value_idx / 8.0,
                card.point_value / 11.0,
                float(card.symbolically_resolved),
                standing,
                *exact_oh,
                *possible_hidden,
                *impossible_hidden,
                *confirmed_hidden,
            ]
        )

    player_rows: list[list[float]] = []
    for player in state.players:
        player_rows.append(
            [
                player.relative_seat / 3.0,
                float(player.is_self),
                float(player.is_partner),
                player.cards_remaining / 13.0,
                len(player.confirmed_cards) / 13.0,
                len(player.possible_cards) / 13.0,
                *[float(v) for v in player.void_suits],
                *[float(v) for v in player.required_half_suits],
                *[float(v) for v in player.required_pair_suits],
            ]
        )

    global_features = torch.tensor(
        [
            state.global_state.active_rel_seat / 3.0,
            state.global_state.trick_number / 9.0,
            state.global_state.trick_position / 3.0,
            -1.0 if state.global_state.trump is None else state.global_state.trump / 3.0,
            state.global_state.current_contract_value / 420.0,
            state.global_state.legal_action_count / 64.0,
            state.strategy.visible_pair_points_floor / 120.0,
            state.strategy.visible_pair_points_ceiling / 120.0,
            state.strategy.makeable_bid_floor / 420.0,
            state.strategy.makeable_bid_ceiling / 420.0,
            state.strategy.trump_called_count / 4.0,
            len(state.strategy.standing_card_indices) / 36.0,
        ],
        dtype=torch.float32,
    )

    return BeliefFeatures(
        card_features=torch.tensor(card_rows, dtype=torch.float32),
        player_features=torch.tensor(player_rows, dtype=torch.float32),
        global_features=global_features,
    )


def load_canonical_state(record: Mapping[str, object]) -> CanonicalState:
    return CanonicalState.from_record(record)


def iter_hidden_card_indices(targets: BeliefTargets) -> Iterable[int]:
    return (idx for idx, is_hidden in enumerate(targets.hidden_card_mask) if is_hidden)
