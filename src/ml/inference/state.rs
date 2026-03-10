use crate::game::cards::{Card, Suit, Value};
use crate::game::gameevent::{ActionType, AnswerType, GameCallback};
use crate::game::player::PlaceAtTable;
use crate::game::Game;
use crate::ml::observation::{card_index, suit_index, Observation};

use super::engine::apply_hidden_set_constraints;
use super::terms::{
    HalfConstraint, HalfConstraintGrid, HiddenConfirmedMask, HiddenPossibleMask, CARD_COUNT,
    HIDDEN_SEATS, SUIT_COUNT,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InferenceDelta {
    pub played_cards: Vec<usize>,
    pub impossible_cards: Vec<(usize, usize)>,
    pub confirmed_cards: Vec<(usize, usize)>,
    pub half_constraints: Vec<(usize, usize, HalfConstraint)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceState {
    possible: HiddenPossibleMask,
    confirmed: HiddenConfirmedMask,
    half_constraints: HalfConstraintGrid,
    played: [bool; CARD_COUNT],
}

impl InferenceState {
    pub fn from_masks(
        possible: HiddenPossibleMask,
        confirmed: HiddenConfirmedMask,
        half_constraints: HalfConstraintGrid,
    ) -> Self {
        let mut state = Self {
            possible,
            confirmed,
            half_constraints,
            played: [false; CARD_COUNT],
        };
        state.recompute();
        state
    }

    pub fn empty() -> Self {
        Self::from_masks(
            [[false; CARD_COUNT]; HIDDEN_SEATS],
            [[false; CARD_COUNT]; HIDDEN_SEATS],
            [[HalfConstraint::Unknown; SUIT_COUNT]; HIDDEN_SEATS],
        )
    }

    pub fn apply_delta(&mut self, delta: &InferenceDelta) {
        for &card_idx in &delta.played_cards {
            if card_idx >= CARD_COUNT {
                continue;
            }
            self.played[card_idx] = true;
            for seat in 0..HIDDEN_SEATS {
                self.possible[seat][card_idx] = false;
                self.confirmed[seat][card_idx] = false;
            }
        }

        for &(seat, card_idx) in &delta.impossible_cards {
            if seat < HIDDEN_SEATS && card_idx < CARD_COUNT && !self.played[card_idx] {
                self.possible[seat][card_idx] = false;
                self.confirmed[seat][card_idx] = false;
            }
        }

        for &(seat, card_idx) in &delta.confirmed_cards {
            if seat < HIDDEN_SEATS && card_idx < CARD_COUNT && !self.played[card_idx] {
                self.possible[seat][card_idx] = true;
                self.confirmed[seat][card_idx] = true;
            }
        }

        for &(seat, suit_idx, incoming) in &delta.half_constraints {
            if seat < HIDDEN_SEATS && suit_idx < SUIT_COUNT {
                self.half_constraints[seat][suit_idx] =
                    self.half_constraints[seat][suit_idx].strongest(incoming);
            }
        }

        self.recompute();
    }

    pub fn possible_mask(&self) -> &HiddenPossibleMask {
        &self.possible
    }

    pub fn confirmed_mask(&self) -> &HiddenConfirmedMask {
        &self.confirmed
    }

    pub fn half_constraints(&self) -> &HalfConstraintGrid {
        &self.half_constraints
    }

    pub fn is_played(&self, card_idx: usize) -> bool {
        self.played.get(card_idx).copied().unwrap_or(false)
    }

    pub fn possible_hidden_rel(&self, card_idx: usize) -> Vec<usize> {
        if card_idx >= CARD_COUNT || self.played[card_idx] {
            return vec![];
        }
        let mut out = Vec::new();
        for seat in 0..HIDDEN_SEATS {
            if self.possible[seat][card_idx] {
                out.push(seat + 1);
            }
        }
        out
    }

    pub fn confirmed_hidden_rel(&self, card_idx: usize) -> Option<usize> {
        if card_idx >= CARD_COUNT || self.played[card_idx] {
            return None;
        }
        let mut owner = None;
        for seat in 0..HIDDEN_SEATS {
            if self.confirmed[seat][card_idx] {
                if owner.is_some() {
                    return None;
                }
                owner = Some(seat + 1);
            }
        }
        owner
    }

    pub fn void_suits(&self) -> [[bool; SUIT_COUNT]; HIDDEN_SEATS] {
        let mut out = [[false; SUIT_COUNT]; HIDDEN_SEATS];
        for seat in 0..HIDDEN_SEATS {
            for suit_idx in 0..SUIT_COUNT {
                let start = suit_idx * 9;
                let any_possible = (start..start + 9).any(|card_idx| self.possible[seat][card_idx]);
                out[seat][suit_idx] = !any_possible;
            }
        }
        out
    }

    fn recompute(&mut self) {
        apply_hidden_set_constraints(
            &mut self.possible,
            &mut self.confirmed,
            &self.half_constraints,
        );
        for card_idx in 0..CARD_COUNT {
            if self.played[card_idx] {
                for seat in 0..HIDDEN_SEATS {
                    self.possible[seat][card_idx] = false;
                    self.confirmed[seat][card_idx] = false;
                }
            }
        }
    }
}

pub fn build_inference_state_from_observation(
    game: &Game,
    pov: &PlaceAtTable,
    obs: &Observation,
) -> InferenceState {
    let mut state = InferenceState::from_masks(
        obs.possible_bitmasks,
        obs.confirmed_bitmasks,
        [[HalfConstraint::Unknown; SUIT_COUNT]; HIDDEN_SEATS],
    );

    let hidden_places = [pov.next(), pov.partner(), pov.prev()];
    let opp_places = [pov.next(), pov.partner(), pov.prev()];

    let mut current_trick_lead_suit: Option<Suit> = None;
    let mut historical_trump: Option<Suit> = None;
    let mut cards_in_trick: Vec<(PlaceAtTable, Card)> = vec![];

    for event in &game.all_events {
        if let ActionType::AnnounceTrump(suit) = &event.last_action.action_type {
            historical_trump = Some(*suit);
        } else if let Some(GameCallback::NewTrump(suit)) = &event.callback {
            historical_trump = Some(*suit);
        }

        if let Some(hidden_idx) = hidden_places
            .iter()
            .position(|place| place.0 == event.last_action.player.0)
        {
            let mut delta = InferenceDelta::default();
            match &event.last_action.action_type {
                ActionType::Answer(AnswerType::YesPair(suit)) => {
                    let suit_idx = suit_index(*suit);
                    let ober = Card {
                        suit: *suit,
                        value: Value::Ober,
                    };
                    let king = Card {
                        suit: *suit,
                        value: Value::King,
                    };
                    delta.confirmed_cards.push((hidden_idx, card_index(&ober)));
                    delta.confirmed_cards.push((hidden_idx, card_index(&king)));
                    delta
                        .half_constraints
                        .push((hidden_idx, suit_idx, HalfConstraint::RequireBoth));
                }
                ActionType::Answer(AnswerType::YesHalf(suit)) => {
                    delta.half_constraints.push((
                        hidden_idx,
                        suit_index(*suit),
                        HalfConstraint::RequireAtLeastOne,
                    ));
                }
                ActionType::Pass(cards) => {
                    let giver = event.last_action.player.clone();
                    let receiver = giver.partner();
                    if giver == *pov {
                        if let Some(partner_slot) = opp_places.iter().position(|p| *p == receiver) {
                            for card in cards {
                                let idx = card_index(card);
                                delta.confirmed_cards.push((partner_slot, idx));
                                for seat in 0..HIDDEN_SEATS {
                                    if seat != partner_slot {
                                        delta.impossible_cards.push((seat, idx));
                                    }
                                }
                            }
                        }
                    } else if receiver == *pov {
                        for card in cards {
                            let idx = card_index(card);
                            for seat in 0..HIDDEN_SEATS {
                                delta.impossible_cards.push((seat, idx));
                            }
                        }
                    }
                }
                _ => {}
            }
            if !delta.confirmed_cards.is_empty()
                || !delta.impossible_cards.is_empty()
                || !delta.half_constraints.is_empty()
                || !delta.played_cards.is_empty()
            {
                state.apply_delta(&delta);
            }
        }

        if let ActionType::CardPlayed(card) = &event.last_action.action_type {
            let mut delta = InferenceDelta::default();
            let played_idx = card_index(card);
            delta.played_cards.push(played_idx);

            let player = &event.last_action.player;
            if cards_in_trick.is_empty() {
                current_trick_lead_suit = Some(card.suit);
            } else if let Some(lead_suit) = current_trick_lead_suit {
                if card.suit != lead_suit {
                    let trump_is_lead = historical_trump
                        .as_ref()
                        .map(|t| *t == lead_suit)
                        .unwrap_or(false);
                    let played_trump = historical_trump
                        .as_ref()
                        .map(|t| *t == card.suit)
                        .unwrap_or(false);
                    if !trump_is_lead || !played_trump {
                        if let Some(opp_idx) = opp_places.iter().position(|place| place.0 == player.0) {
                            for suit_card_idx in 0..CARD_COUNT {
                                if suit_card_idx / 9 == suit_index(lead_suit) {
                                    delta.impossible_cards.push((opp_idx, suit_card_idx));
                                }
                            }
                        }
                    }
                }
            }
            cards_in_trick.push((player.clone(), card.clone()));
            if cards_in_trick.len() == 4 {
                cards_in_trick.clear();
                current_trick_lead_suit = None;
            }
            state.apply_delta(&delta);
        }
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::gameevent::{ActionType, AnswerType, GameAction, GameEvent};
    use crate::game::player::PlaceAtTable;
    use crate::ml::observation::build_observation;

    #[test]
    fn delta_played_card_removes_it_from_all_hidden_hands() {
        let mut possible = [[false; CARD_COUNT]; HIDDEN_SEATS];
        possible[0][7] = true;
        possible[1][7] = true;
        let mut state = InferenceState::from_masks(
            possible,
            [[false; CARD_COUNT]; HIDDEN_SEATS],
            [[HalfConstraint::Unknown; SUIT_COUNT]; HIDDEN_SEATS],
        );
        let mut delta = InferenceDelta::default();
        delta.played_cards.push(7);
        state.apply_delta(&delta);
        assert!(state.is_played(7));
        assert_eq!(state.possible_hidden_rel(7), Vec::<usize>::new());
        assert_eq!(state.confirmed_hidden_rel(7), None);
    }

    #[test]
    fn delta_confirmed_card_projects_unique_owner() {
        let mut possible = [[false; CARD_COUNT]; HIDDEN_SEATS];
        possible[0][5] = true;
        possible[1][5] = true;
        let mut state = InferenceState::from_masks(
            possible,
            [[false; CARD_COUNT]; HIDDEN_SEATS],
            [[HalfConstraint::Unknown; SUIT_COUNT]; HIDDEN_SEATS],
        );
        let mut delta = InferenceDelta::default();
        delta.confirmed_cards.push((1, 5));
        state.apply_delta(&delta);
        assert_eq!(state.confirmed_hidden_rel(5), Some(2));
    }

    #[test]
    fn half_constraints_only_strengthen_monotonically() {
        let mut state = InferenceState::empty();
        let mut delta = InferenceDelta::default();
        delta.half_constraints.push((0, 1, HalfConstraint::RequireAtLeastOne));
        state.apply_delta(&delta);
        assert_eq!(state.half_constraints()[0][1], HalfConstraint::RequireAtLeastOne);

        let mut weaker = InferenceDelta::default();
        weaker.half_constraints.push((0, 1, HalfConstraint::Unknown));
        state.apply_delta(&weaker);
        assert_eq!(state.half_constraints()[0][1], HalfConstraint::RequireAtLeastOne);

        let mut stronger = InferenceDelta::default();
        stronger.half_constraints.push((0, 1, HalfConstraint::RequireBoth));
        state.apply_delta(&stronger);
        assert_eq!(state.half_constraints()[0][1], HalfConstraint::RequireBoth);
    }

    #[test]
    fn void_suits_follow_possible_mask() {
        let mut possible = [[false; CARD_COUNT]; HIDDEN_SEATS];
        possible[0][0] = true;
        possible[1][9] = true;
        let state = InferenceState::from_masks(
            possible,
            [[false; CARD_COUNT]; HIDDEN_SEATS],
            [[HalfConstraint::Unknown; SUIT_COUNT]; HIDDEN_SEATS],
        );
        let voids = state.void_suits();
        assert!(!voids[0][0]);
        assert!(voids[0][1]);
        assert!(!voids[1][1]);
    }

    #[test]
    fn build_inference_state_replays_pair_and_played_information() {
        let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
        let mut game = Game::new("infer_test".to_string(), names, None);
        let pov = PlaceAtTable(0);
        game.all_events.push(GameEvent {
            last_action: GameAction {
                action_type: ActionType::Answer(AnswerType::YesPair(Suit::Green)),
                player: PlaceAtTable(1),
            },
            callback: None,
            player_at_turn: PlaceAtTable(2),
            time: "t".to_string(),
        });
        game.all_events.push(GameEvent {
            last_action: GameAction {
                action_type: ActionType::CardPlayed(Card {
                    suit: Suit::Green,
                    value: Value::Ober,
                }),
                player: PlaceAtTable(1),
            },
            callback: None,
            player_at_turn: PlaceAtTable(2),
            time: "t2".to_string(),
        });
        let obs = build_observation(&game, pov.clone());
        let inf = build_inference_state_from_observation(&game, &pov, &obs);
        let green_ober = card_index(&Card {
            suit: Suit::Green,
            value: Value::Ober,
        });
        let green_king = card_index(&Card {
            suit: Suit::Green,
            value: Value::King,
        });
        assert!(inf.is_played(green_ober));
        assert_eq!(inf.confirmed_hidden_rel(green_king), Some(1));
    }
}
