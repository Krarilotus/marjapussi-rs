from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CANONICAL_STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CanonicalGlobalState:
    pov_abs_seat: int
    active_abs_seat: int
    active_rel_seat: int
    phase: str
    trick_number: int
    trick_position: int
    trump: int | None
    trump_announced: tuple[bool, bool, bool, bool]
    current_contract_value: int
    legal_action_count: int
    player_at_turn_cards_remaining: int


@dataclass(frozen=True)
class CanonicalCardState:
    card_idx: int
    suit_idx: int
    value_idx: int
    point_value: int
    exact_location: Mapping[str, Any] | str | None
    possible_hidden_rel: tuple[int, ...]
    confirmed_hidden_rel: int | None
    impossible_hidden_rel: tuple[int, ...]
    symbolically_resolved: bool
    standing_for_my_team: bool | None


@dataclass(frozen=True)
class CanonicalPlayerState:
    relative_seat: int
    absolute_seat: int
    is_self: bool
    is_partner: bool
    cards_remaining: int
    confirmed_cards: tuple[int, ...]
    possible_cards: tuple[int, ...]
    void_suits: tuple[bool, bool, bool, bool]
    required_half_suits: tuple[bool, bool, bool, bool]
    required_pair_suits: tuple[bool, bool, bool, bool]


@dataclass(frozen=True)
class CanonicalTeamState:
    team_idx: int
    points: int
    secured_point_floor: int
    max_reachable_points: int


@dataclass(frozen=True)
class CanonicalStrategyState:
    standing_card_indices: tuple[int, ...]
    exhausted_pair_suits: tuple[bool, bool, bool, bool]
    current_trump_suit: int | None
    trump_called_count: int
    visible_pair_points_floor: int
    visible_pair_points_ceiling: int
    makeable_bid_floor: int
    makeable_bid_ceiling: int


@dataclass(frozen=True)
class CanonicalBeliefTargets:
    schema_version: int
    card_owner_classes: tuple[str, ...]
    hidden_card_indices: tuple[int, ...]
    player_void_suits: tuple[tuple[bool, ...], ...]
    player_has_half_suits: tuple[tuple[bool, ...], ...]
    player_has_pair_suits: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True)
class CanonicalState:
    schema_version: int
    global_state: CanonicalGlobalState
    cards: tuple[CanonicalCardState, ...]
    players: tuple[CanonicalPlayerState, ...]
    teams: tuple[CanonicalTeamState, ...]
    strategy: CanonicalStrategyState
    belief_targets: CanonicalBeliefTargets | None = None

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "CanonicalState":
        schema_version = int(data["schema_version"])
        if schema_version != CANONICAL_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported canonical state schema_version={schema_version}, "
                f"expected {CANONICAL_STATE_SCHEMA_VERSION}"
            )

        global_state = CanonicalGlobalState(
            pov_abs_seat=int(data["global"]["pov_abs_seat"]),
            active_abs_seat=int(data["global"]["active_abs_seat"]),
            active_rel_seat=int(data["global"]["active_rel_seat"]),
            phase=str(data["global"]["phase"]),
            trick_number=int(data["global"]["trick_number"]),
            trick_position=int(data["global"]["trick_position"]),
            trump=None if data["global"]["trump"] is None else int(data["global"]["trump"]),
            trump_announced=tuple(bool(v) for v in data["global"]["trump_announced"]),
            current_contract_value=int(data["global"]["current_contract_value"]),
            legal_action_count=int(data["global"]["legal_action_count"]),
            player_at_turn_cards_remaining=int(data["global"]["player_at_turn_cards_remaining"]),
        )

        cards = tuple(
            CanonicalCardState(
                card_idx=int(card["card_idx"]),
                suit_idx=int(card["suit_idx"]),
                value_idx=int(card["value_idx"]),
                point_value=int(card["point_value"]),
                exact_location=card.get("exact_location"),
                possible_hidden_rel=tuple(int(v) for v in card.get("possible_hidden_rel", ())),
                confirmed_hidden_rel=(
                    None
                    if card.get("confirmed_hidden_rel") is None
                    else int(card["confirmed_hidden_rel"])
                ),
                impossible_hidden_rel=tuple(int(v) for v in card.get("impossible_hidden_rel", ())),
                symbolically_resolved=bool(card["symbolically_resolved"]),
                standing_for_my_team=(
                    None
                    if card.get("standing_for_my_team") is None
                    else bool(card["standing_for_my_team"])
                ),
            )
            for card in data["cards"]
        )
        players = tuple(
            CanonicalPlayerState(
                relative_seat=int(player["relative_seat"]),
                absolute_seat=int(player["absolute_seat"]),
                is_self=bool(player["is_self"]),
                is_partner=bool(player["is_partner"]),
                cards_remaining=int(player["cards_remaining"]),
                confirmed_cards=tuple(int(v) for v in player.get("confirmed_cards", ())),
                possible_cards=tuple(int(v) for v in player.get("possible_cards", ())),
                void_suits=tuple(bool(v) for v in player["void_suits"]),
                required_half_suits=tuple(bool(v) for v in player["required_half_suits"]),
                required_pair_suits=tuple(bool(v) for v in player["required_pair_suits"]),
            )
            for player in data["players"]
        )
        teams = tuple(
            CanonicalTeamState(
                team_idx=int(team["team_idx"]),
                points=int(team["points"]),
                secured_point_floor=int(team["secured_point_floor"]),
                max_reachable_points=int(team["max_reachable_points"]),
            )
            for team in data["teams"]
        )
        strategy = CanonicalStrategyState(
            standing_card_indices=tuple(int(v) for v in data["strategy"]["standing_card_indices"]),
            exhausted_pair_suits=tuple(bool(v) for v in data["strategy"]["exhausted_pair_suits"]),
            current_trump_suit=(
                None
                if data["strategy"]["current_trump_suit"] is None
                else int(data["strategy"]["current_trump_suit"])
            ),
            trump_called_count=int(data["strategy"]["trump_called_count"]),
            visible_pair_points_floor=int(data["strategy"]["visible_pair_points_floor"]),
            visible_pair_points_ceiling=int(data["strategy"]["visible_pair_points_ceiling"]),
            makeable_bid_floor=int(data["strategy"]["makeable_bid_floor"]),
            makeable_bid_ceiling=int(data["strategy"]["makeable_bid_ceiling"]),
        )
        belief_targets = None
        if data.get("belief_targets") is not None:
            bt = data["belief_targets"]
            belief_targets = CanonicalBeliefTargets(
                schema_version=int(bt["schema_version"]),
                card_owner_classes=tuple(str(v) for v in bt["card_owner_classes"]),
                hidden_card_indices=tuple(int(v) for v in bt["hidden_card_indices"]),
                player_void_suits=tuple(tuple(bool(v) for v in row) for row in bt["player_void_suits"]),
                player_has_half_suits=tuple(
                    tuple(bool(v) for v in row) for row in bt["player_has_half_suits"]
                ),
                player_has_pair_suits=tuple(
                    tuple(bool(v) for v in row) for row in bt["player_has_pair_suits"]
                ),
            )
        return CanonicalState(
            schema_version=schema_version,
            global_state=global_state,
            cards=cards,
            players=players,
            teams=teams,
            strategy=strategy,
            belief_targets=belief_targets,
        )

    @staticmethod
    def from_record(record: Mapping[str, Any]) -> "CanonicalState":
        payload = record.get("canonical_state")
        if payload is None:
            raise KeyError("record does not contain canonical_state")
        if record.get("belief_targets") is not None and isinstance(payload, Mapping):
            merged = dict(payload)
            merged.setdefault("belief_targets", record["belief_targets"])
            payload = merged
        return CanonicalState.from_dict(payload)

    @property
    def unknown_card_indices(self) -> tuple[int, ...]:
        return tuple(card.card_idx for card in self.cards if not card.symbolically_resolved)

    @property
    def confirmed_partner_cards(self) -> tuple[int, ...]:
        partner = next(player for player in self.players if player.is_partner)
        return partner.confirmed_cards
