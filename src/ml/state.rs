use serde::{Deserialize, Serialize};

use crate::game::cards::{higher_cards, Suit};
use crate::game::player::PlaceAtTable;
use crate::game::points::{points_card, points_pair};
use crate::game::Game;
use crate::ml::inference::{build_inference_state_from_observation, HalfConstraint, InferenceState};
use crate::ml::observation::{
    build_observation, card_from_index, card_index, suit_index, Observation,
};

pub const CANONICAL_STATE_SCHEMA_VERSION: u32 = 1;
const HIDDEN_TO_RELATIVE: [usize; 3] = [1, 2, 3];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalState {
    pub schema_version: u32,
    pub global: CanonicalGlobalState,
    pub cards: Vec<CanonicalCardState>,
    pub players: Vec<CanonicalPlayerState>,
    pub teams: Vec<CanonicalTeamState>,
    pub strategy: CanonicalStrategyState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalBeliefTargets {
    pub schema_version: u32,
    pub card_owner_classes: Vec<CardOwnerClass>,
    pub hidden_card_indices: Vec<usize>,
    pub player_void_suits: Vec<[bool; 4]>,
    pub player_has_half_suits: Vec<[bool; 4]>,
    pub player_has_pair_suits: Vec<[bool; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalGlobalState {
    pub pov_abs_seat: u8,
    pub active_abs_seat: u8,
    pub active_rel_seat: usize,
    pub phase: String,
    pub trick_number: usize,
    pub trick_position: usize,
    pub trump: Option<usize>,
    pub trump_announced: [bool; 4],
    pub current_contract_value: i32,
    pub legal_action_count: usize,
    pub player_at_turn_cards_remaining: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExactCardLocation {
    MyHand,
    HiddenLeft,
    HiddenPartner,
    HiddenRight,
    CurrentTrick { relative_seat: usize },
    Played,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CardOwnerClass {
    SelfHand,
    LeftHand,
    PartnerHand,
    RightHand,
    TrickSelf,
    TrickLeft,
    TrickPartner,
    TrickRight,
    Played,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalCardState {
    pub card_idx: usize,
    pub suit_idx: usize,
    pub value_idx: usize,
    pub point_value: i32,
    pub exact_location: Option<ExactCardLocation>,
    pub possible_hidden_rel: Vec<usize>,
    pub confirmed_hidden_rel: Option<usize>,
    pub impossible_hidden_rel: Vec<usize>,
    pub symbolically_resolved: bool,
    pub standing_for_my_team: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalPlayerState {
    pub relative_seat: usize,
    pub absolute_seat: u8,
    pub is_self: bool,
    pub is_partner: bool,
    pub cards_remaining: usize,
    pub confirmed_cards: Vec<usize>,
    pub possible_cards: Vec<usize>,
    pub void_suits: [bool; 4],
    pub required_half_suits: [bool; 4],
    pub required_pair_suits: [bool; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalTeamState {
    pub team_idx: usize,
    pub points: i32,
    pub secured_point_floor: i32,
    pub max_reachable_points: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalStrategyState {
    pub standing_card_indices: Vec<usize>,
    pub exhausted_pair_suits: [bool; 4],
    pub current_trump_suit: Option<usize>,
    pub trump_called_count: usize,
    pub visible_pair_points_floor: i32,
    pub visible_pair_points_ceiling: i32,
    pub makeable_bid_floor: i32,
    pub makeable_bid_ceiling: i32,
}

pub fn build_canonical_state(game: &Game, pov: PlaceAtTable) -> CanonicalState {
    let obs = build_observation(game, pov.clone());
    build_canonical_state_from_observation(game, &pov, &obs)
}

pub fn build_canonical_belief_targets(
    game: &Game,
    pov: PlaceAtTable,
    obs: Option<&Observation>,
) -> CanonicalBeliefTargets {
    let owned_obs;
    let obs = match obs {
        Some(obs) => obs,
        None => {
            owned_obs = build_observation(game, pov.clone());
            &owned_obs
        }
    };
    let current_trick_lookup = current_trick_lookup(obs);

    let mut card_owner_classes = Vec::with_capacity(36);
    let mut hidden_card_indices = Vec::new();
    for card_idx in 0..36 {
        let owner = true_card_owner_class(game, pov.clone(), card_idx, &current_trick_lookup);
        if matches!(
            owner,
            CardOwnerClass::LeftHand | CardOwnerClass::PartnerHand | CardOwnerClass::RightHand
        ) {
            hidden_card_indices.push(card_idx);
        }
        card_owner_classes.push(owner);
    }

    let mut player_void_suits = Vec::with_capacity(4);
    let mut player_has_half_suits = Vec::with_capacity(4);
    let mut player_has_pair_suits = Vec::with_capacity(4);
    for relative_seat in 0..4 {
        let place = PlaceAtTable(((pov.0 as usize + relative_seat) % 4) as u8);
        let player = game.state.player_at_place(place);
        let mut void_suits = [true; 4];
        let mut half_suits = [false; 4];
        let mut pair_suits = [false; 4];
        let mut counts = [[false; 2]; 4];
        for card in &player.cards {
            let suit_idx = suit_index(card.suit);
            void_suits[suit_idx] = false;
            match card.value {
                crate::game::cards::Value::Ober => counts[suit_idx][0] = true,
                crate::game::cards::Value::King => counts[suit_idx][1] = true,
                _ => {}
            }
        }
        for suit_idx in 0..4 {
            half_suits[suit_idx] = counts[suit_idx][0] || counts[suit_idx][1];
            pair_suits[suit_idx] = counts[suit_idx][0] && counts[suit_idx][1];
        }
        player_void_suits.push(void_suits);
        player_has_half_suits.push(half_suits);
        player_has_pair_suits.push(pair_suits);
    }

    CanonicalBeliefTargets {
        schema_version: CANONICAL_STATE_SCHEMA_VERSION,
        card_owner_classes,
        hidden_card_indices,
        player_void_suits,
        player_has_half_suits,
        player_has_pair_suits,
    }
}

pub fn build_canonical_state_from_observation(
    game: &Game,
    pov: &PlaceAtTable,
    obs: &Observation,
) -> CanonicalState {
    let current_trick_lookup = current_trick_lookup(obs);
    let inference = build_inference_state_from_observation(game, pov, obs);
    let cards = build_card_states(obs, &current_trick_lookup, &inference);
    let players = build_player_states(game, pov, obs, &inference);
    let teams = build_team_states(obs);
    let strategy = build_strategy_state(game, obs, &cards, &players);
    let active_rel = relative_from_pov(pov.clone(), game.state.player_at_turn.clone());

    CanonicalState {
        schema_version: CANONICAL_STATE_SCHEMA_VERSION,
        global: CanonicalGlobalState {
            pov_abs_seat: pov.0,
            active_abs_seat: game.state.player_at_turn.0,
            active_rel_seat: active_rel,
            phase: obs.phase.clone(),
            trick_number: obs.trick_number,
            trick_position: obs.trick_position,
            trump: obs.trump,
            trump_announced: obs.trump_announced,
            current_contract_value: game.state.value.0,
            legal_action_count: obs.legal_actions.len(),
            player_at_turn_cards_remaining: game.state.player_at_turn().cards.len(),
        },
        cards,
        players,
        teams,
        strategy,
    }
}

fn build_card_states(
    obs: &Observation,
    current_trick_lookup: &[Option<usize>; 36],
    inference: &InferenceState,
) -> Vec<CanonicalCardState> {
    (0..36)
        .map(|card_idx| {
            let exact_location = if obs.my_hand_bitmask[card_idx] {
                Some(ExactCardLocation::MyHand)
            } else if let Some(relative_seat) = current_trick_lookup[card_idx] {
                Some(ExactCardLocation::CurrentTrick { relative_seat })
            } else if obs.played_bitmask[card_idx] {
                Some(ExactCardLocation::Played)
            } else if let Some(hidden_rel) = inference.confirmed_hidden_rel(card_idx) {
                Some(match hidden_rel {
                    1 => ExactCardLocation::HiddenLeft,
                    2 => ExactCardLocation::HiddenPartner,
                    3 => ExactCardLocation::HiddenRight,
                    _ => unreachable!("invalid hidden relative seat"),
                })
            } else {
                None
            };

            let possible_hidden_rel = inference.possible_hidden_rel(card_idx);
            let impossible_hidden_rel = HIDDEN_TO_RELATIVE
                .iter()
                .copied()
                .filter(|rel| !possible_hidden_rel.contains(rel))
                .collect::<Vec<_>>();

            let standing_for_my_team = exact_location
                .as_ref()
                .and_then(|loc| {
                    standing_status_for_location(card_idx, loc, obs, current_trick_lookup, inference)
                });

            CanonicalCardState {
                card_idx,
                suit_idx: card_idx / 9,
                value_idx: card_idx % 9,
                point_value: points_card(card_from_index(card_idx)).0,
                confirmed_hidden_rel: inference.confirmed_hidden_rel(card_idx),
                exact_location,
                possible_hidden_rel,
                impossible_hidden_rel,
                symbolically_resolved: obs.my_hand_bitmask[card_idx]
                    || obs.played_bitmask[card_idx]
                    || current_trick_lookup[card_idx].is_some()
                    || inference.confirmed_hidden_rel(card_idx).is_some(),
                standing_for_my_team,
            }
        })
        .collect()
}

fn build_player_states(
    game: &Game,
    pov: &PlaceAtTable,
    obs: &Observation,
    inference: &InferenceState,
) -> Vec<CanonicalPlayerState> {
    let mut players = Vec::with_capacity(4);
    for relative_seat in 0..4 {
        let absolute_seat = PlaceAtTable(((pov.0 as usize + relative_seat) % 4) as u8);
        let is_self = relative_seat == 0;
        let is_partner = relative_seat == 2;
        let cards_remaining = obs.cards_remaining[relative_seat];

        if is_self {
            players.push(CanonicalPlayerState {
                relative_seat,
                absolute_seat: absolute_seat.0,
                is_self,
                is_partner,
                cards_remaining,
                confirmed_cards: obs.my_hand_indices.clone(),
                possible_cards: obs.my_hand_indices.clone(),
                void_suits: self_void_suits(game, absolute_seat),
                required_half_suits: [false; 4],
                required_pair_suits: [false; 4],
            });
            continue;
        }

        let hidden_idx = relative_to_hidden_index(relative_seat);
        let confirmed_cards = (0..36)
            .filter(|&card_idx| inference.confirmed_mask()[hidden_idx][card_idx])
            .collect::<Vec<_>>();
        let possible_cards = (0..36)
            .filter(|&card_idx| inference.possible_mask()[hidden_idx][card_idx])
            .collect::<Vec<_>>();

        let void_suits = inference.void_suits()[hidden_idx];

        let mut required_half_suits = [false; 4];
        let mut required_pair_suits = [false; 4];
        for suit_idx in 0..4 {
            match inference.half_constraints()[hidden_idx][suit_idx] {
                HalfConstraint::RequireAtLeastOne => required_half_suits[suit_idx] = true,
                HalfConstraint::RequireBoth => {
                    required_half_suits[suit_idx] = true;
                    required_pair_suits[suit_idx] = true;
                }
                HalfConstraint::Unknown => {}
            }
        }

        players.push(CanonicalPlayerState {
            relative_seat,
            absolute_seat: absolute_seat.0,
            is_self,
            is_partner,
            cards_remaining,
            confirmed_cards,
            possible_cards,
            void_suits,
            required_half_suits,
            required_pair_suits,
        });
    }
    players
}

fn build_team_states(obs: &Observation) -> Vec<CanonicalTeamState> {
    let remaining_points = remaining_unplayed_points(obs);
    vec![
        CanonicalTeamState {
            team_idx: 0,
            points: obs.points_my_team,
            secured_point_floor: obs.points_my_team,
            max_reachable_points: obs.points_my_team + remaining_points,
        },
        CanonicalTeamState {
            team_idx: 1,
            points: obs.points_opp_team,
            secured_point_floor: obs.points_opp_team,
            max_reachable_points: obs.points_opp_team + remaining_points,
        },
    ]
}

fn build_strategy_state(
    _game: &Game,
    obs: &Observation,
    cards: &[CanonicalCardState],
    players: &[CanonicalPlayerState],
) -> CanonicalStrategyState {
    let standing_card_indices = cards
        .iter()
        .filter_map(|card| match card.standing_for_my_team {
            Some(true) => Some(card.card_idx),
            _ => None,
        })
        .collect::<Vec<_>>();

    let mut exhausted_pair_suits = [false; 4];
    let mut visible_pair_points_floor = 0i32;
    let mut visible_pair_points_ceiling = 0i32;

    for suit_idx in 0..4 {
        let (ober_idx, king_idx) = (suit_idx * 9 + 5, suit_idx * 9 + 6);
        let ober_team_known = card_belongs_to_my_team(ober_idx, cards);
        let king_team_known = card_belongs_to_my_team(king_idx, cards);
        let ober_team_possible = card_may_belong_to_my_team(ober_idx, cards);
        let king_team_possible = card_may_belong_to_my_team(king_idx, cards);
        let pair_points = points_pair(suit_from_index(suit_idx)).0;

        if ober_team_known && king_team_known {
            visible_pair_points_floor += pair_points;
        }
        if ober_team_possible && king_team_possible {
            visible_pair_points_ceiling += pair_points;
        }

        exhausted_pair_suits[suit_idx] = cards[ober_idx].symbolically_resolved
            && cards[king_idx].symbolically_resolved;
    }

    let standing_point_floor = standing_card_indices
        .iter()
        .map(|&card_idx| points_card(card_from_index(card_idx)).0)
        .sum::<i32>();
    let my_team_ceiling = obs.points_my_team + remaining_unplayed_points(obs);
    let my_team_floor = obs.points_my_team;
    let raw_bid_floor = 115 + visible_pair_points_floor + standing_point_floor / 2;
    let raw_bid_ceiling = 115 + visible_pair_points_ceiling + (my_team_ceiling - my_team_floor) / 2;
    let makeable_bid_floor = clamp_bid(raw_bid_floor.max(120));
    let makeable_bid_ceiling = clamp_bid(raw_bid_ceiling.max(makeable_bid_floor));

    let _player_count = players.len();

    CanonicalStrategyState {
        standing_card_indices,
        exhausted_pair_suits,
        current_trump_suit: obs.trump,
        trump_called_count: obs.trump_announced.iter().filter(|&&v| v).count(),
        visible_pair_points_floor,
        visible_pair_points_ceiling,
        makeable_bid_floor,
        makeable_bid_ceiling,
    }
}

fn current_trick_lookup(obs: &Observation) -> [Option<usize>; 36] {
    let mut lookup = [None; 36];
    for (card_idx, relative_seat) in obs
        .current_trick_indices
        .iter()
        .zip(obs.current_trick_players.iter())
    {
        lookup[*card_idx] = Some(*relative_seat);
    }
    lookup
}

fn true_card_owner_class(
    game: &Game,
    pov: PlaceAtTable,
    card_idx: usize,
    current_trick_lookup: &[Option<usize>; 36],
) -> CardOwnerClass {
    if let Some(relative_seat) = current_trick_lookup[card_idx] {
        return match relative_seat {
            0 => CardOwnerClass::TrickSelf,
            1 => CardOwnerClass::TrickLeft,
            2 => CardOwnerClass::TrickPartner,
            3 => CardOwnerClass::TrickRight,
            _ => unreachable!("invalid current trick relative seat"),
        };
    }

    if game
        .state
        .all_tricks
        .iter()
        .any(|trick| trick.cards.iter().any(|card| card_index(card) == card_idx))
    {
        return CardOwnerClass::Played;
    }

    for relative_seat in 0..4 {
        let place = PlaceAtTable(((pov.0 as usize + relative_seat) % 4) as u8);
        if game
            .state
            .player_at_place(place)
            .cards
            .iter()
            .any(|card| card_index(card) == card_idx)
        {
            return match relative_seat {
                0 => CardOwnerClass::SelfHand,
                1 => CardOwnerClass::LeftHand,
                2 => CardOwnerClass::PartnerHand,
                3 => CardOwnerClass::RightHand,
                _ => unreachable!("invalid relative seat"),
            };
        }
    }

    CardOwnerClass::Played
}

fn standing_status_for_location(
    card_idx: usize,
    location: &ExactCardLocation,
    obs: &Observation,
    current_trick_lookup: &[Option<usize>; 36],
    inference: &InferenceState,
) -> Option<bool> {
    let owner_rel = match location {
        ExactCardLocation::MyHand => 0,
        ExactCardLocation::HiddenLeft => 1,
        ExactCardLocation::HiddenPartner => 2,
        ExactCardLocation::HiddenRight => 3,
        ExactCardLocation::CurrentTrick { relative_seat } => *relative_seat,
        ExactCardLocation::Played => return None,
    };
    let owner_team = owner_rel % 2;
    let card = card_from_index(card_idx);
    let higher = higher_cards(&card, obs.trump.map(suit_from_index), None);
    for higher_card in higher {
        let higher_idx = card_index(&higher_card);
        if obs.played_bitmask[higher_idx] {
            continue;
        }
        if current_trick_lookup[higher_idx].is_some() {
            if let Some(rel) = current_trick_lookup[higher_idx] {
                if rel % 2 != owner_team {
                    return Some(false_for_owner_team(owner_team));
                }
            }
            continue;
        }
        if obs.my_hand_bitmask[higher_idx] {
            if 0usize % 2 != owner_team {
                return Some(false_for_owner_team(owner_team));
            }
            continue;
        }
        if let Some(rel) = inference.confirmed_hidden_rel(higher_idx) {
            if rel % 2 != owner_team {
                return Some(false_for_owner_team(owner_team));
            }
            continue;
        }
        for rel in inference.possible_hidden_rel(higher_idx) {
            if rel % 2 != owner_team {
                return Some(false_for_owner_team(owner_team));
            }
        }
    }
    Some(owner_team == 0)
}

fn false_for_owner_team(owner_team: usize) -> bool {
    owner_team != 0
}

fn card_belongs_to_my_team(card_idx: usize, cards: &[CanonicalCardState]) -> bool {
    match cards[card_idx].exact_location {
        Some(ExactCardLocation::MyHand) | Some(ExactCardLocation::HiddenPartner) => true,
        Some(ExactCardLocation::CurrentTrick { relative_seat }) => relative_seat % 2 == 0,
        _ => false,
    }
}

fn card_may_belong_to_my_team(card_idx: usize, cards: &[CanonicalCardState]) -> bool {
    if card_belongs_to_my_team(card_idx, cards) {
        return true;
    }
    cards[card_idx].possible_hidden_rel.iter().any(|rel| rel % 2 == 0)
}

fn self_void_suits(game: &Game, absolute_seat: PlaceAtTable) -> [bool; 4] {
    let mut void_suits = [true; 4];
    for card in &game.state.player_at_place(absolute_seat).cards {
        void_suits[suit_index(card.suit)] = false;
    }
    void_suits
}

fn relative_to_hidden_index(relative_seat: usize) -> usize {
    match relative_seat {
        1 => 0,
        2 => 1,
        3 => 2,
        _ => panic!("relative seat {relative_seat} is not hidden"),
    }
}

fn relative_from_pov(pov: PlaceAtTable, seat: PlaceAtTable) -> usize {
    ((seat.0 + 4 - pov.0) % 4) as usize
}

fn remaining_unplayed_points(obs: &Observation) -> i32 {
    (0..36)
        .filter(|&card_idx| !obs.played_bitmask[card_idx] && !obs.current_trick_indices.contains(&card_idx))
        .map(|card_idx| points_card(card_from_index(card_idx)).0)
        .sum()
}

fn suit_from_index(idx: usize) -> Suit {
    match idx {
        0 => Suit::Green,
        1 => Suit::Acorns,
        2 => Suit::Bells,
        3 => Suit::Red,
        _ => panic!("invalid suit index {idx}"),
    }
}

fn clamp_bid(value: i32) -> i32 {
    let clamped = value.clamp(120, 420);
    let offset = clamped - 120;
    120 + (offset / 5) * 5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::gameevent::{ActionType, GameAction};

    fn start_bidding_game() -> Game {
        let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
        let mut game = Game::new("canonical".to_string(), names, None);
        for _ in 0..4 {
            let act = game.legal_actions[0].clone();
            game.apply_action_mut(act);
        }
        game
    }

    #[test]
    fn canonical_state_has_expected_sections() {
        let game = start_bidding_game();
        let state = build_canonical_state(&game, PlaceAtTable(0));
        assert_eq!(state.schema_version, CANONICAL_STATE_SCHEMA_VERSION);
        assert_eq!(state.cards.len(), 36);
        assert_eq!(state.players.len(), 4);
        assert_eq!(state.teams.len(), 2);
        assert_eq!(state.global.phase, "Bidding");
    }

    #[test]
    fn canonical_state_resolves_my_hand_and_hidden_candidates() {
        let game = start_bidding_game();
        let state = build_canonical_state(&game, PlaceAtTable(0));
        let my_count = state
            .cards
            .iter()
            .filter(|card| card.exact_location == Some(ExactCardLocation::MyHand))
            .count();
        assert_eq!(my_count, 9);
        for card in &state.cards {
            if card.exact_location == Some(ExactCardLocation::MyHand) {
                assert!(card.possible_hidden_rel.is_empty());
            }
        }
    }

    #[test]
    fn canonical_state_tracks_passed_cards_to_partner() {
        let names = ["S1", "S2", "S3", "S4"].map(|s| s.to_string());
        let mut game = Game::new("pass_state".to_string(), names, None);
        for _ in 0..4 {
            let act = game.legal_actions[0].clone();
            game.apply_action_mut(act);
        }

        let bid140 = GameAction {
            action_type: ActionType::NewBid(140),
            player: game.state.player_at_turn.clone(),
        };
        game.apply_action_mut(bid140);
        for _ in 0..4 {
            let act = game.legal_actions[3].clone();
            game.apply_action_mut(act);
        }
        for _ in 0..3 {
            let act = game.legal_actions[0].clone();
            game.apply_action_mut(act);
        }

        let passer = game.state.player_at_turn.clone();
        let pass_action = game.legal_actions[0].clone();
        let passed_cards = match &pass_action.action_type {
            ActionType::Pass(cards) => cards.clone(),
            _ => panic!("expected pass action"),
        };
        game.apply_action_mut(pass_action);

        let state = build_canonical_state(&game, passer);
        for card in passed_cards {
            let idx = card_index(&card);
            if state.cards[idx].exact_location == Some(ExactCardLocation::Played) {
                continue;
            }
            assert_eq!(state.cards[idx].confirmed_hidden_rel, Some(2));
        }
    }

    #[test]
    fn canonical_belief_targets_cover_all_cards_exactly_once() {
        let game = start_bidding_game();
        let targets = build_canonical_belief_targets(&game, PlaceAtTable(0), None);
        assert_eq!(targets.card_owner_classes.len(), 36);
        assert_eq!(
            targets
                .card_owner_classes
                .iter()
                .filter(|owner| {
                    matches!(
                        owner,
                        CardOwnerClass::LeftHand
                            | CardOwnerClass::PartnerHand
                            | CardOwnerClass::RightHand
                    )
                })
                .count(),
            targets.hidden_card_indices.len()
        );
        assert_eq!(targets.player_void_suits.len(), 4);
        assert_eq!(targets.player_has_half_suits.len(), 4);
        assert_eq!(targets.player_has_pair_suits.len(), 4);
    }
}
