use rand::prelude::IndexedRandom;

use crate::game::gameevent::GameAction;

use crate::game::gameinfo::GameFinishedInfo;
use crate::game::gamestate::GamePhase;
use crate::game::Game;

use std::collections::HashMap;
use crate::ml::search::TtEntry;

/// A simple policy function type: given the current game state and a transposition table, choose one action index.
pub type PolicyFn = Box<dyn Fn(&Game, &mut HashMap<u128, TtEntry>) -> usize + Send + Sync>;

/// Run a game from its current state to completion, using the provided policy
/// to select actions at each step.
pub fn run_to_end(mut game: Game, policy: &PolicyFn, cache: &mut HashMap<u128, TtEntry>) -> (Game, GameFinishedInfo) {
    while game.state.phase != GamePhase::Ended {
        let idx = policy(&game, cache);
        let actions = &game.legal_actions;
        if actions.is_empty() {
            break;
        }
        let idx = idx.min(actions.len() - 1);
        game.apply_action_mut(actions[idx].clone());
    }
    let info = GameFinishedInfo::from(game.clone());
    (game, info)
}

/// Try every legal action at the current decision point, run each branch to
/// completion with the given policy, and return outcomes for all branches.
pub fn try_all_actions(game: &Game, policy: &PolicyFn, cache: &mut HashMap<u128, TtEntry>, num_rollouts: usize) -> Vec<(GameAction, Vec<GameFinishedInfo>)> {
    let actions = game.legal_actions.clone();
    let mut results = vec![];
    for action in actions {
        let mut group = vec![];
        for _ in 0..num_rollouts {
            let mut branch = game.clone();
            branch.apply_action_mut(action.clone());
            let (_final_game, info) = run_to_end(branch, policy, cache);
            group.push(info);
        }
        results.push((action, group));
    }
    results
}

/// Random policy: uniformly picks a random legal action.
pub fn random_policy() -> PolicyFn {
    Box::new(|game: &Game, _cache: &mut HashMap<u128, TtEntry>| {
        let actions = &game.legal_actions;
        if actions.is_empty() { return 0; }
        let mut rng = rand::rng();
        actions.choose(&mut rng)
            .map(|_| (0..actions.len()).collect::<Vec<_>>().choose(&mut rng).copied().unwrap_or(0))
            .unwrap_or(0)
    })
}

/// Heuristic policy: simple rule-based agent.
/// - Prefer playing an Ace if we can (maximum point capture).
/// - Otherwise play the highest card we're forced to (to win tricks).
/// - When leading: play lowest card to avoid giving away points.
/// - Announce own trump immediately if legal.
pub fn heuristic_policy() -> PolicyFn {
    Box::new(|game: &Game, _cache: &mut HashMap<u128, TtEntry>| {
        use crate::game::gameevent::{ActionType, QuestionType};
        use crate::game::cards::Value;
        let actions = &game.legal_actions;
        if actions.is_empty() { return 0; }

        // Last 3 tricks (trick 7, 8, 9): use optimal search
        // DISABLED due to excessive CPU utilization / hanging
        // if game.state.all_tricks.len() + 1 >= 7 {
        //     let (best_idx, _val) = crate::ml::search::find_best_action(game, 12, _cache); // 12 plies max (3 tricks * 4)
        //     if let Some(idx) = best_idx {
        //         return idx;
        //     }
        // }

        // Priority: AnnounceTrump > play Ace > play Ten > play lowest card
        // Answers are forced (only one legal), bids: stop bidding (conservative)

        // If there's a trump announcement, take it immediately
        for (i, a) in actions.iter().enumerate() {
            if matches!(a.action_type, ActionType::AnnounceTrump(_)) {
                return i;
            }
        }

        // If we can ask before leading, prefer informative questions that help
        // resolve our own halves into trump before we throw those cards away.
        let hand_now = game.state.player_at_turn().cards.clone();
        let my_halves = crate::game::cards::halves(hand_now.clone());
        let my_pairs = crate::game::cards::pairs(hand_now);
        let suit_rank = |s: &crate::game::cards::Suit| -> i32 {
            use crate::game::cards::Suit;
            match s {
                Suit::Red => 4,
                Suit::Bells => 3,
                Suit::Acorns => 2,
                Suit::Green => 1,
            }
        };
        let mut best_question: Option<(usize, i32)> = None;
        for (i, a) in actions.iter().enumerate() {
            let q_score = match &a.action_type {
                ActionType::Question(QuestionType::YourHalf(suit)) => {
                    let mut score = 0;
                    if my_halves.contains(suit) { score += 40; }
                    if my_pairs.contains(suit) { score -= 15; }
                    if !game.state.trump_called.contains(suit) { score += 15; }
                    else { score -= 20; }
                    score += suit_rank(suit);
                    Some(score)
                }
                ActionType::Question(QuestionType::Yours) => {
                    let mut score = 5;
                    if my_pairs.is_empty() { score += 10; }
                    if my_halves.len() >= 2 { score += 10; }
                    Some(score)
                }
                _ => None,
            };
            if let Some(score) = q_score {
                match best_question {
                    Some((_, best)) if score <= best => {}
                    _ => best_question = Some((i, score)),
                }
            }
        }
        if let Some((idx, _)) = best_question {
            return idx;
        }

        // Collect card-play actions
        let card_plays: Vec<(usize, &crate::game::cards::Card)> = actions.iter()
            .enumerate()
            .filter_map(|(i, a)| {
                if let ActionType::CardPlayed(card) = &a.action_type {
                    Some((i, card))
                } else {
                    None
                }
            })
            .collect();

        if !card_plays.is_empty() {
            // Prefer Ace
            if let Some(&(i, _)) = card_plays.iter().find(|(_, c)| c.value == Value::Ace) {
                return i;
            }
            // Prefer Ten
            if let Some(&(i, _)) = card_plays.iter().find(|(_, c)| c.value == Value::Ten) {
                return i;
            }
            // Play highest card to win
            let best = card_plays.iter()
                .max_by_key(|(_, c)| c.value.clone());
            if let Some(&(i, _)) = best {
                return i;
            }
        }

        let mut is_bidding_action = false;
        for a in actions {
            if matches!(a.action_type, ActionType::NewBid(_) | ActionType::StopBidding) {
                is_bidding_action = true; break;
            }
        }

        if is_bidding_action {
            let mut team_bid_count = 0usize;
            let mut current_max = 115i32;
            let my_seat = game.state.player_at_turn.0;
            let partner_seat = (my_seat + 2) % 4;
            let mut team_first_jump: Option<i32> = None;
            let mut partner_first_jump: Option<i32> = None;

            for (action, p) in &game.state.bidding_history {
                if let ActionType::NewBid(val) = action {
                    let jump = *val - current_max;
                    if p.0 == my_seat {
                        team_bid_count += 1;
                        if team_first_jump.is_none() {
                            team_first_jump = Some(jump);
                        }
                    } else if p.0 == partner_seat {
                        team_bid_count += 1;
                        if team_first_jump.is_none() {
                            team_first_jump = Some(jump);
                        }
                        if partner_first_jump.is_none() {
                            partner_first_jump = Some(jump);
                        }
                    }
                    current_max = *val;
                }
            }

            let hand = &game.state.players[my_seat as usize].cards;
            use crate::game::cards::{Suit, Value};
            let has_ace = hand.iter().any(|c| c.value == Value::Ace);
            let mut ace_count = 0i32;
            let mut ten_count = 0i32;
            let mut unmatched_halves = 0i32;
            let mut small_pair_count = 0i32;
            let mut big_pair_count = 0i32;
            let mut pair_points = 0i32;

            for c in hand {
                if c.value == Value::Ace {
                    ace_count += 1;
                }
                if c.value == Value::Ten {
                    ten_count += 1;
                }
            }

            for &suit in &[Suit::Acorns, Suit::Green, Suit::Bells, Suit::Red] {
                let has_king = hand.iter().any(|c| c.suit == suit && c.value == Value::King);
                let has_ober = hand.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                if has_king && has_ober {
                    let suit_pair_points = match suit {
                        Suit::Red => 100,
                        Suit::Bells => 80,
                        Suit::Acorns => 60,
                        Suit::Green => 40,
                    };
                    pair_points += suit_pair_points;
                    if matches!(suit, Suit::Red | Suit::Bells) {
                        big_pair_count += 1;
                    } else {
                        small_pair_count += 1;
                    }
                } else if has_king || has_ober {
                    unmatched_halves += 1;
                }
            }

            let has_small_pair = small_pair_count > 0;
            let has_big_pair = big_pair_count > 0;
            let has_pair = has_small_pair || has_big_pair;
            let own_pair_count = small_pair_count + big_pair_count;

            // Requested strict first/second team bidding protocol.
            let mut desired_step: Option<i32> = None;
            if team_bid_count == 0 {
                if has_ace {
                    // First bidder of team with ace always starts +5.
                    desired_step = Some(5);
                } else if has_big_pair {
                    desired_step = Some(15);
                } else if has_small_pair || unmatched_halves >= 3 {
                    desired_step = Some(10);
                }
            } else if team_bid_count == 1 {
                // Team second bid: ace-info bid no longer applies if team already opened +5.
                let first_was_ace_signal = team_first_jump == Some(5);
                if has_big_pair {
                    desired_step = Some(15);
                } else if has_small_pair || unmatched_halves >= 3 {
                    desired_step = Some(10);
                } else if first_was_ace_signal && unmatched_halves >= 2 {
                    desired_step = Some(5);
                }
            } else {
                // Later rounds: continue only in conservative +5 steps with clear own strength.
                if has_ace || has_pair || unmatched_halves >= 2 {
                    desired_step = Some(5);
                }
            }

            // Strict >140 gates from partner inference and team structure.
            let partner_signaled_10 = partner_first_jump == Some(10);
            let partner_signaled_15 = partner_first_jump == Some(15);
            let partner_signaled_ace = partner_first_jump == Some(5);
            let team_opened_with_ace_step = team_first_jump == Some(5);
            let team_has_ace_signal = has_ace || partner_signaled_ace || team_opened_with_ace_step;

            // Partner opened +10 and we have at least 2 unmatched halves.
            let inferred_pair_from_10 = partner_signaled_10 && unmatched_halves >= 2;
            // Partner opened +15 guarantees pair info.
            let inferred_pair_from_15 = partner_signaled_15;
            // Team opened +5 on ace and partner also showed +5, while we have strong own support.
            let inferred_from_double_five_with_strength =
                team_opened_with_ace_step
                    && partner_signaled_ace
                    && (has_pair || unmatched_halves >= 3);
            // No-ace exception: only allow >140 with stronger pair certainty.
            let inferred_two_pairs_without_ace =
                !team_has_ace_signal && own_pair_count >= 1 && partner_signaled_10 && unmatched_halves >= 1;

            let allow_over_140 = inferred_pair_from_15
                || inferred_pair_from_10
                || inferred_from_double_five_with_strength
                || inferred_two_pairs_without_ace;

            // Cautious hand-value estimate:
            // base + own standing-card strength + half of pair value, capped to 200.
            let king_count = hand.iter().filter(|c| c.value == Value::King).count() as i32;
            let ober_count = hand.iter().filter(|c| c.value == Value::Ober).count() as i32;
            let own_standing_est = ace_count * 12 + ten_count * 9 + king_count * 6 + ober_count * 4 + unmatched_halves * 2;
            let mut estimated_value = 115 + own_standing_est + (pair_points / 2);
            if partner_signaled_15 {
                estimated_value += 20;
            } else if partner_signaled_10 {
                estimated_value += 10;
            }
            estimated_value = estimated_value.clamp(115, 200);

            let mut max_willing_bid = estimated_value;
            if !allow_over_140 {
                max_willing_bid = max_willing_bid.min(140);
            }
            if !team_has_ace_signal && !inferred_two_pairs_without_ace {
                max_willing_bid = 140;
            }

            let mut chosen_action = None;
            if let Some(step) = desired_step {
                let target_bid = current_max + step;
                if target_bid <= max_willing_bid {
                    if let Some(idx) = actions.iter().position(|a| matches!(a.action_type, ActionType::NewBid(v) if v == target_bid)) {
                        chosen_action = Some(idx);
                    }
                }
            }

            if chosen_action.is_none() {
                chosen_action = actions.iter().position(|a| matches!(a.action_type, ActionType::StopBidding));
            }
            if let Some(idx) = chosen_action {
                return idx;
            }
        }

        let passing_actions: Vec<(usize, &Vec<crate::game::cards::Card>)> = actions.iter()
            .enumerate()
            .filter_map(|(i, a)| if let ActionType::Pass(cards) = &a.action_type { Some((i, cards)) } else { None })
            .collect();

        if !passing_actions.is_empty() {
            let my_seat = game.state.player_at_turn.0;
            let hand = &game.state.players[my_seat as usize].cards;
            let is_forth = matches!(game.state.phase, GamePhase::PassingForth);
            let partner_seat = (my_seat + 2) % 4;
            let incoming_from_partner: Vec<crate::game::cards::Card> = if is_forth {
                vec![]
            } else {
                game.all_events
                    .iter()
                    .rev()
                    .find_map(|ev| match &ev.last_action.action_type {
                        ActionType::Pass(cards) => Some(cards.clone()),
                        _ => None,
                    })
                    .unwrap_or_default()
            };
            let mut hand_suits = [0i32; 4];
            for c in hand {
                hand_suits[c.suit.clone() as usize] += 1;
            }
            let num_suits_before = hand_suits.iter().filter(|&&c| c > 0).count() as i32;

            // Extract first jump info from bidding to estimate what was already communicated.
            let mut current_bid = 115i32;
            let mut my_first_jump: Option<i32> = None;
            let mut partner_first_jump: Option<i32> = None;
            for (action, p) in &game.state.bidding_history {
                if let ActionType::NewBid(v) = action {
                    let jump = *v - current_bid;
                    if p.0 == my_seat && my_first_jump.is_none() {
                        my_first_jump = Some(jump);
                    } else if p.0 == partner_seat && partner_first_jump.is_none() {
                        partner_first_jump = Some(jump);
                    }
                    current_bid = *v;
                }
            }
            let communicated_ace = my_first_jump == Some(5);
            let partner_communicated_ace = partner_first_jump == Some(5);

            let mut incoming_suits = [0i32; 4];
            for c in &incoming_from_partner {
                incoming_suits[c.suit.clone() as usize] += 1;
            }
             
            let mut best_score = i32::MIN;
            let mut best_idx = passing_actions[0].0;

            for &(idx, passed_cards) in &passing_actions {
                let mut score = 0i32;
                let mut kept = hand.clone();
                for c in passed_cards {
                    if let Some(pos) = kept.iter().position(|x| x == c) {
                        kept.remove(pos);
                    }
                }
                
                let mut suits_kept = [0i32; 4];
                for c in &kept {
                    suits_kept[c.suit.clone() as usize] += 1;
                }
                let num_suits = suits_kept.iter().filter(|&&c| c > 0).count() as i32;
                let cut_suit_count = (0..4)
                    .filter(|&s| hand_suits[s] > 0 && suits_kept[s] == 0)
                    .count() as i32;
                
                use crate::game::cards::{Suit, Value};
                
                if is_forth {
                    // Primary: shape hand to 2 suits.
                    if num_suits <= 2 { score += 450; }
                    else if num_suits == 3 { score += 120; }
                    else { score -= 180; }
                    score += cut_suit_count * 140;
                    if num_suits > 2 {
                        score -= (num_suits - 2) * 120;
                    }

                    // Secondary: pass high-value cards not explicitly communicated yet.
                    for c in passed_cards {
                        match c.value {
                            Value::Ace => {
                                score += if communicated_ace { 20 } else { 55 };
                                if !partner_communicated_ace {
                                    score += 10;
                                }
                            }
                            Value::Ten => score += 18,
                            Value::King | Value::Ober => score += 10,
                            _ => {}
                        }
                    }

                    // Strongly avoid splitting own pairs while passing forth.
                    for &suit in &[Suit::Acorns, Suit::Green, Suit::Bells, Suit::Red] {
                        let had_king = hand.iter().any(|c| c.suit == suit && c.value == Value::King);
                        let had_ober = hand.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                        if had_king && had_ober {
                            let passed_king = passed_cards.iter().any(|c| c.suit == suit && c.value == Value::King);
                            let passed_ober = passed_cards.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                            if passed_king || passed_ober {
                                score -= 280;
                            }
                        }
                    }
                } else {
                    // Back-passer: avoid bouncing back same cards.
                    let immediate_return_count = passed_cards
                        .iter()
                        .filter(|c| incoming_from_partner.contains(c))
                        .count() as i32;
                    score -= immediate_return_count * 180;

                    // Also avoid passing back incoming suit colors in general.
                    for c in passed_cards {
                        if incoming_suits[c.suit.clone() as usize] > 0 {
                            score -= 35;
                        }
                    }

                    // Prefer trimming to 2 suits and actually cutting one suit.
                    if num_suits <= 2 { score += 260; }
                    else if num_suits == 3 { score += 80; }
                    else { score -= 120; }
                    score += cut_suit_count * 100;
                    if num_suits_before - num_suits >= 1 {
                        score += 60;
                    }

                    // Keep pair/trump potential and do not split pairs.
                    let mut kept_pair_count = 0i32;
                    for &suit in &[Suit::Acorns, Suit::Green, Suit::Bells, Suit::Red] {
                        let had_king = hand.iter().any(|c| c.suit == suit && c.value == Value::King);
                        let had_ober = hand.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                        let has_king_kept = kept.iter().any(|c| c.suit == suit && c.value == Value::King);
                        let has_ober_kept = kept.iter().any(|c| c.suit == suit && c.value == Value::Ober);
                        if has_king_kept && has_ober_kept {
                            kept_pair_count += 1;
                        }
                        if had_king && had_ober && !(has_king_kept && has_ober_kept) {
                            score -= 260;
                        }
                    }
                    score += kept_pair_count * 110;

                    // Standing-card shaping: keep at least one ace and preserve tops.
                    let aces_kept = kept.iter().filter(|c| c.value == Value::Ace).count() as i32;
                    let tens_kept = kept.iter().filter(|c| c.value == Value::Ten).count() as i32;
                    if aces_kept >= 1 { score += 90; } else { score -= 150; }
                    score += aces_kept * 25 + tens_kept * 12;

                    // If we pass an ace back, keep a low card in same suit for control.
                    for ace in passed_cards.iter().filter(|c| c.value == Value::Ace) {
                        let has_low_same_suit = kept.iter().any(|c| {
                            c.suit == ace.suit
                                && matches!(c.value, Value::Six | Value::Seven | Value::Eight | Value::Nine)
                        });
                        if has_low_same_suit {
                            score += 20;
                        } else {
                            score -= 60;
                        }
                    }
                }
                
                if score > best_score {
                    best_score = score;
                    best_idx = idx;
                }
            }
            return best_idx;
        }

        // Default: first legal action
        0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_started_game() -> Game {
        let names = ["P0", "P1", "P2", "P3"].map(|s| s.to_string());
        let mut game = Game::new("sim_test".to_string(), names, None);
        let mut actions = game.legal_actions.clone();
        for _ in 0..4 {
            game = game.apply_action(actions.pop().unwrap()).unwrap();
            actions = game.legal_actions.clone();
        }
        game
    }

    #[test]
    fn test_run_to_end_random() {
        let game = make_started_game();
        let mut cache = std::collections::HashMap::new();
        let (_final_game, info) = run_to_end(game, &random_policy(), &mut cache);
        // A game should have 9 tricks
        assert_eq!(info.tricks.len(), 9);
    }

    #[test]
    fn test_run_to_end_heuristic() {
        let game = make_started_game();
        let mut cache = std::collections::HashMap::new();
        let (_final_game, info) = run_to_end(game, &heuristic_policy(), &mut cache);
        assert_eq!(info.tricks.len(), 9);
    }

    #[test]
    fn test_try_all_actions_first_decision() {
        let game = make_started_game();
        let mut cache = std::collections::HashMap::new();
        // In bidding phase, try all bids
        let results = try_all_actions(&game, &random_policy(), &mut cache, 5);
        assert!(!results.is_empty());
        for (_, infos) in &results {
            for info in infos {
                assert_eq!(info.tricks.len(), 9);
            }
        }
    }
}
