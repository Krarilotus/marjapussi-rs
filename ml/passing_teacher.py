from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from ml.neurosymbolic_state import CanonicalState
except ModuleNotFoundError:
    from neurosymbolic_state import CanonicalState


PASSING_RULE_TARGET_NAMES = (
    "target_retained_suits",
    "prefer_blank_suits",
    "preserve_pairs",
    "preserve_aces",
    "preserve_trump",
    "standing_cards",
    "pair_points_ceiling",
    "point_diff",
)


@dataclass(frozen=True)
class PassingTeacherTargets:
    aux_targets: torch.Tensor
    teacher_policy: torch.Tensor


def _normalize_points(value: float) -> float:
    return max(-1.0, min(1.0, value / 420.0))


def _suit_idx(card_idx: int) -> int:
    return int(card_idx) // 9


def _value_idx(card_idx: int) -> int:
    return int(card_idx) % 9


def _count_aces(cards: list[int]) -> int:
    return sum(1 for card_idx in cards if _value_idx(card_idx) == 8)


def _count_trump(cards: list[int], trump_suit: int | None) -> int:
    if trump_suit is None:
        return 0
    return sum(1 for card_idx in cards if _suit_idx(card_idx) == trump_suit)


def _distinct_suits(cards: list[int]) -> set[int]:
    return {_suit_idx(card_idx) for card_idx in cards}


def _pair_suits(cards: list[int]) -> set[int]:
    suits: dict[int, set[int]] = {}
    for card_idx in cards:
        value_idx = _value_idx(card_idx)
        if value_idx not in (5, 6):
            continue
        suit_idx = _suit_idx(card_idx)
        suits.setdefault(suit_idx, set()).add(value_idx)
    return {suit_idx for suit_idx, values in suits.items() if values == {5, 6}}


def _half_suits(cards: list[int]) -> set[int]:
    return {_suit_idx(card_idx) for card_idx in cards if _value_idx(card_idx) in (5, 6)}


def _score_passing_action(
    *,
    my_hand: list[int],
    pass_cards: list[int],
    phase_name: str,
    trump_suit: int | None,
) -> float:
    pass_set = set(int(card_idx) for card_idx in pass_cards)
    if len(pass_set) != 4 or any(card_idx not in my_hand for card_idx in pass_set):
        return -1e6

    remaining = [card_idx for card_idx in my_hand if card_idx not in pass_set]
    before_suits = _distinct_suits(my_hand)
    after_suits = _distinct_suits(remaining)
    blanked_suits = before_suits - after_suits
    blanked_non_trump = {
        suit_idx for suit_idx in blanked_suits if trump_suit is None or suit_idx != trump_suit
    }

    orig_pair_suits = _pair_suits(my_hand)
    rem_pair_suits = _pair_suits(remaining)
    broken_pairs = len(orig_pair_suits - rem_pair_suits)

    orig_half_suits = _half_suits(my_hand)
    rem_half_suits = _half_suits(remaining)
    broken_halves = len(orig_half_suits - rem_half_suits)

    ace_count = _count_aces(my_hand)
    kept_aces = _count_aces(remaining)
    passed_aces = _count_aces(list(pass_set))
    helpful_passed_aces = sum(
        1
        for card_idx in pass_set
        if _value_idx(card_idx) == 8 and _suit_idx(card_idx) in blanked_non_trump
    )
    passed_trump = _count_trump(list(pass_set), trump_suit)
    remaining_suits = len(after_suits)
    target_retained_suits = 2
    if phase_name == "PassingBack" and (len(orig_pair_suits) >= 2 or ace_count >= 2):
        target_retained_suits = 3
    suit_target_score = 1.0 - min(abs(remaining_suits - target_retained_suits), 2) / 2.0
    blank_gain = len(blanked_non_trump) / max(1, len(before_suits))
    pair_pres = 1.0 - (broken_pairs / max(1, len(orig_pair_suits))) if orig_pair_suits else 1.0
    half_pres = 1.0 - (broken_halves / max(1, len(orig_half_suits))) if orig_half_suits else 1.0
    ace_pres = kept_aces / max(1, ace_count) if ace_count else 1.0
    helpful_ace_ratio = helpful_passed_aces / max(1, passed_aces) if passed_aces else 0.0
    low_card_return = sum(1 for card_idx in pass_set if _value_idx(card_idx) <= 3) / 4.0

    if phase_name == "PassingForth":
        return (
            2.2 * suit_target_score
            + 1.8 * blank_gain
            + 1.5 * pair_pres
            + 0.8 * half_pres
            + 0.8 * helpful_ace_ratio
            + 0.3 * low_card_return
            + 0.2 * ace_pres
            - 0.9 * passed_trump
        )
    return (
        1.0 * suit_target_score
        + 0.4 * blank_gain
        + 2.0 * pair_pres
        + 1.2 * half_pres
        + 1.4 * ace_pres
        + 0.5 * low_card_return
        - 1.1 * passed_trump
    )


def build_passing_teacher_targets(record: dict, state: CanonicalState) -> PassingTeacherTargets:
    legal_actions = list(record["obs"].get("legal_actions", []))
    phase_name = str(record["obs"].get("phase", state.global_state.phase))
    my_hand = sorted(card.card_idx for card in state.cards if card.exact_location == "MyHand")
    current_suits = _distinct_suits(my_hand)
    pair_suits = _pair_suits(my_hand)
    ace_count = _count_aces(my_hand)
    trump_suit = state.strategy.current_trump_suit
    target_retained_suits = 2
    if phase_name == "PassingBack" and (len(pair_suits) >= 2 or ace_count >= 2):
        target_retained_suits = 3

    aux_targets = torch.tensor(
        [
            target_retained_suits / 4.0,
            float(len(current_suits) > target_retained_suits),
            min(1.0, len(pair_suits) / 2.0),
            float(ace_count > 0),
            float(trump_suit is not None and _count_trump(my_hand, trump_suit) > 0),
            len(state.strategy.standing_card_indices) / 36.0,
            max(0.0, min(1.0, state.strategy.visible_pair_points_ceiling / 120.0)),
            _normalize_points(
                float(record.get("outcome_pts_my_team", 0.0)) - float(record.get("outcome_pts_opp", 0.0))
            ),
        ],
        dtype=torch.float32,
    )

    teacher_policy = torch.zeros(len(legal_actions), dtype=torch.float32)
    scores: list[float] = []
    valid_indices: list[int] = []
    for idx, legal_action in enumerate(legal_actions):
        pass_cards = legal_action.get("pass_cards")
        if not pass_cards:
            continue
        score = _score_passing_action(
            my_hand=my_hand,
            pass_cards=[int(card_idx) for card_idx in pass_cards],
            phase_name=phase_name,
            trump_suit=trump_suit,
        )
        if score <= -1e5:
            continue
        valid_indices.append(idx)
        scores.append(score)

    if valid_indices:
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(score_tensor / 0.35, dim=0)
        for idx, prob in zip(valid_indices, probs.tolist()):
            teacher_policy[idx] = float(prob)
    elif len(legal_actions) > 0:
        teacher_policy.fill_(1.0 / len(legal_actions))

    return PassingTeacherTargets(aux_targets=aux_targets, teacher_policy=teacher_policy)
