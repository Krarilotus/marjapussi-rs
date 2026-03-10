from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

try:
    from ml.neurosymbolic_dataset import (
        OWNER_LEFT_HAND,
        OWNER_PARTNER_HAND,
        OWNER_RIGHT_HAND,
        OWNER_UNKNOWN,
    )
    from ml.neurosymbolic_state import CanonicalState
except ModuleNotFoundError:
    from neurosymbolic_dataset import (
        OWNER_LEFT_HAND,
        OWNER_PARTNER_HAND,
        OWNER_RIGHT_HAND,
        OWNER_UNKNOWN,
    )
    from neurosymbolic_state import CanonicalState


@dataclass(frozen=True)
class DecodedBeliefState:
    card_owner_classes: tuple[int, ...]
    hidden_capacities: tuple[int, int, int]


def hidden_capacities_from_state(state: CanonicalState) -> tuple[int, int, int]:
    caps = []
    for player in state.players[1:]:
        caps.append(max(0, player.cards_remaining - len(player.confirmed_cards)))
    return tuple(caps)  # left, partner, right


def decode_hidden_card_assignment(
    owner_logits: torch.Tensor,
    state: CanonicalState,
) -> DecodedBeliefState:
    if owner_logits.ndim != 2:
        raise ValueError(f"owner_logits must be [36, num_owner_classes], got {tuple(owner_logits.shape)}")
    if owner_logits.shape[0] != len(state.cards):
        raise ValueError("owner_logits/card count mismatch")

    capacities = hidden_capacities_from_state(state)
    base_assignment = [OWNER_UNKNOWN] * len(state.cards)
    unknown_cards: list[int] = []
    scores: list[tuple[float, float, float]] = []

    for card in state.cards:
        exact = card.exact_location
        if exact == "MyHand":
            base_assignment[card.card_idx] = 0
        elif exact == "HiddenLeft":
            base_assignment[card.card_idx] = OWNER_LEFT_HAND
        elif exact == "HiddenPartner":
            base_assignment[card.card_idx] = OWNER_PARTNER_HAND
        elif exact == "HiddenRight":
            base_assignment[card.card_idx] = OWNER_RIGHT_HAND
        elif exact == "Played":
            base_assignment[card.card_idx] = 8
        elif isinstance(exact, dict) and "CurrentTrick" in exact:
            rel = int(exact["CurrentTrick"]["relative_seat"])
            base_assignment[card.card_idx] = 4 + rel
        else:
            unknown_cards.append(card.card_idx)
            row = owner_logits[card.card_idx]
            scores.append(
                (
                    float(row[OWNER_LEFT_HAND].item()) if 1 in card.possible_hidden_rel else float("-inf"),
                    float(row[OWNER_PARTNER_HAND].item()) if 2 in card.possible_hidden_rel else float("-inf"),
                    float(row[OWNER_RIGHT_HAND].item()) if 3 in card.possible_hidden_rel else float("-inf"),
                )
            )

    @lru_cache(maxsize=None)
    def solve(card_pos: int, cap_left: int, cap_partner: int, cap_right: int) -> tuple[float, tuple[int, ...]]:
        if card_pos == len(unknown_cards):
            if cap_left == 0 and cap_partner == 0 and cap_right == 0:
                return 0.0, ()
            return float("-inf"), ()

        best_score = float("-inf")
        best_assign: tuple[int, ...] = ()
        left_score, partner_score, right_score = scores[card_pos]

        if cap_left > 0 and left_score != float("-inf"):
            score, assign = solve(card_pos + 1, cap_left - 1, cap_partner, cap_right)
            score += left_score
            if score > best_score:
                best_score = score
                best_assign = (OWNER_LEFT_HAND,) + assign
        if cap_partner > 0 and partner_score != float("-inf"):
            score, assign = solve(card_pos + 1, cap_left, cap_partner - 1, cap_right)
            score += partner_score
            if score > best_score:
                best_score = score
                best_assign = (OWNER_PARTNER_HAND,) + assign
        if cap_right > 0 and right_score != float("-inf"):
            score, assign = solve(card_pos + 1, cap_left, cap_partner, cap_right - 1)
            score += right_score
            if score > best_score:
                best_score = score
                best_assign = (OWNER_RIGHT_HAND,) + assign

        return best_score, best_assign

    _, hidden_assignment = solve(0, capacities[0], capacities[1], capacities[2])
    if len(hidden_assignment) != len(unknown_cards):
        remaining_caps = [capacities[0], capacities[1], capacities[2]]
        greedy_assignment: list[int] = []
        for left_score, partner_score, right_score in scores:
            candidates = [
                (OWNER_LEFT_HAND, left_score, 0),
                (OWNER_PARTNER_HAND, partner_score, 1),
                (OWNER_RIGHT_HAND, right_score, 2),
            ]
            candidates.sort(key=lambda item: item[1], reverse=True)
            assigned = OWNER_UNKNOWN
            for owner_class, score, cap_idx in candidates:
                if score == float("-inf"):
                    continue
                if remaining_caps[cap_idx] <= 0:
                    continue
                remaining_caps[cap_idx] -= 1
                assigned = owner_class
                break
            greedy_assignment.append(assigned)
        hidden_assignment = tuple(greedy_assignment)

    for card_idx, owner_class in zip(unknown_cards, hidden_assignment):
        base_assignment[card_idx] = owner_class

    return DecodedBeliefState(
        card_owner_classes=tuple(base_assignment),
        hidden_capacities=capacities,
    )
