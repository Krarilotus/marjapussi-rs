# Specification: Marjapussi Inference Engine

## 1. Goal
Build a deterministic inference engine that runs for each seat and maintains the strongest logically valid knowledge about every card, based on:

1. Official game rules.
2. Publicly observed actions.
3. Private information visible to that seat (own hand, own pass interaction).

The engine must update after every new action and produce:

1. Current knowledge (facts and remaining possibilities).
2. Derived tactical flags (standing cards, trump potential).
3. Counterfactual knowledge impact for each own legal action.

This document is the baseline specification for implementation and test design.

## 2. Scope and Inference Policy

1. Hard inference only: derive only what is guaranteed by rules and observations.
2. No psychology/heuristics in core inference (bidding style reads, bluff reads).
3. Knowledge is seat-relative: each player runs their own engine with their own visibility.
4. Engine is incremental: consume only new actions since last update.
5. Engine is monotonic for a forward-only action stream.
6. If undo is supported, recompute from the last valid checkpoint/event index.

## 3. Terminology and Mapping

1. Suits (code): `Rot`, `Schell`, `Eichel`, `Gruen`.
2. Values (high to low): `Ass`, `Zehn`, `Koenig`, `Ober`, `Unter`, `Neun`, `Acht`, `Sieben`, `Sechs`.
3. Pair in a suit: `Koenig + Ober`.
4. Half in a suit: `Koenig` or `Ober`.
5. Seats: `PlaceAtTable(0..3)`, partner is `+2 mod 4`.
6. Parties:
   1. Playing party (S-party): bidding winner seat and partner.
   2. Non-playing party (N-party): the other two seats.
7. Total cards: 36, each player starts with 9.
8. Trick count: 9 tricks total.

## 4. Required AI Module Architecture

Create `src/ai` with at least:

1. `observation.rs`
2. `inference.rs`
3. `rules.rs`

Recommended additional modules:

1. `types.rs` (knowledge structs, enums, bitsets)
2. `constraints.rs` (set-theoretic closure helpers)
3. `impact.rs` (counterfactual action-impact layer)
4. `mod.rs`

### 4.1 observation.rs responsibilities

1. Convert game-facing stream (`GameEvent` + visible state) into normalized observed actions.
2. Preserve event ordering and trick context.
3. Keep `last_seen_event_index`.
4. Emit only unseen actions to inference.
5. Handle visibility:
   1. Public actions visible to all.
   2. Passing cards:
      1. If seat is directly involved in pass, include known cards.
      2. If not visible, emit `PassHidden`.
6. Reconstruct interaction-level semantics for question/answer/trump sequences from action + callback.
7. Enforce sequence invariants during reconstruction (illegal sequence -> contradiction marker).

### 4.2 inference.rs responsibilities

1. Own `KnowledgeState`.
2. Apply observed actions one by one.
3. After each action:
   1. Add direct facts.
   2. Run rule engine to fixed point.
4. Expose query API:
   1. Card status by card.
   2. Per-player known/possible cards.
   3. Standing/trump potential flags.
   4. Contradiction diagnostics.

### 4.3 rules.rs responsibilities

1. Rule registry with deterministic ordering.
2. Each rule returns newly derived facts (or none).
3. Rules do not mutate outside the knowledge API.
4. No duplicated logic between rules and engine.

### 4.4 Normalized observed action model (mandatory)

`observation.rs` must output a normalized stream for inference, not raw backend actions only.

Required shape:

```rust
enum ObservedAction {
    Start { actor: PlaceAtTable },
    NewBid { actor: PlaceAtTable, value: i32 },
    StopBidding { actor: PlaceAtTable },
    PassKnown { actor: PlaceAtTable, target: PlaceAtTable, cards: Vec<Card> },
    PassHidden { actor: PlaceAtTable, target: PlaceAtTable },
    CardPlayed { actor: PlaceAtTable, card: Card, trick_index: u8, pos_in_trick: u8 },
    RaiseToValue { actor: PlaceAtTable, value: i32 },
    AnnounceTrumpOwnPair { actor: PlaceAtTable, suit: Suit, was_new_call: bool },
    AskPair { actor: PlaceAtTable, target: PlaceAtTable, interaction_id: u32 },
    AskHalf { actor: PlaceAtTable, target: PlaceAtTable, suit: Suit, interaction_id: u32 },
    AnswerPairYes { actor: PlaceAtTable, suit: Suit, interaction_id: u32 },
    AnswerPairNo { actor: PlaceAtTable, interaction_id: u32 },
    AnswerHalfYes { actor: PlaceAtTable, suit: Suit, interaction_id: u32 },
    AnswerHalfNo { actor: PlaceAtTable, suit: Suit, interaction_id: u32 },
    QaResolution { interaction_id: u32, outcome: QaOutcome },
}
```

Where `QaOutcome` is:

```rust
enum QaOutcome {
    TrumpCalled { suit: Suit, by_party: [PlaceAtTable; 2], was_new_call: bool },
    NoTrump,
    OnlyResponderHasHalf { suit: Suit }, // corresponds to backend callback OnlyHalf
}
```

Reconstruction requirements for question/trump cases:

1. Own pair trump call:
   1. Input: `AnnounceTrump(suit)`.
   2. Output: `AnnounceTrumpOwnPair { was_new_call = !trump_called_before }`.
   3. Rule: valid only if suit was not called before.
2. Ask pair -> yes pair:
   1. Input sequence: `Question(Yours)` then `Answer(YesPair(suit))`.
   2. Output: `AskPair`, `AnswerPairYes`, `QaResolution::TrumpCalled`.
   3. Rule: trump is called only if suit not called before (in current backend this path is only legal for uncalled suit).
3. Ask pair -> no pair:
   1. Input sequence: `Question(Yours)` then `Answer(NoPair)`.
   2. Output: `AskPair`, `AnswerPairNo`, `QaResolution::NoTrump`.
4. Ask half -> yes half -> asker does not have other half:
   1. Input sequence: `Question(YourHalf(suit))`, `Answer(YesHalf(suit))`, callback `OnlyHalf(suit)`.
   2. Output: `AskHalf`, `AnswerHalfYes`, `QaResolution::OnlyResponderHasHalf`.
   3. Rule: no trump call.
5. Ask half -> yes half -> asker also has half:
   1. Input sequence: `Question(YourHalf(suit))`, `Answer(YesHalf(suit))`, callback `NewTrump(suit)` or `StillTrump(suit)`.
   2. Output: `AskHalf`, `AnswerHalfYes`, `QaResolution::TrumpCalled`.
   3. Rule: new trump only if suit not called before; otherwise `was_new_call = false`.
6. Ask half -> no half:
   1. Input sequence: `Question(YourHalf(suit))`, `Answer(NoHalf(suit))` (and/or callback `NoHalf(suit)`).
   2. Output: `AskHalf`, `AnswerHalfNo`, `QaResolution::NoTrump`.

### 4.5 ObservedAction semantics (variant-by-variant)

Common fields:

1. `actor`: seat that performed the action.
2. `target`: seat directly addressed by `actor` (only for pass/question actions).
3. `interaction_id`: unique id linking ask/answer/resolution in one QA interaction.
4. `trick_index`: `0..8` absolute trick number.
5. `pos_in_trick`: `0..3` order inside current trick.
6. `was_new_call`: `true` only when suit transitions from uncalled to called at that action.

Variant semantics:

1. `Start { actor }`
   1. Meaning: player marks ready/start in pre-game phase.
   2. Note: no card-information effect; only phase/timeline effect.
2. `NewBid { actor, value }`
   1. Meaning: actor bids to `value` during bidding.
3. `StopBidding { actor }`
   1. Meaning: actor passes from bidding permanently for current game.
4. `PassKnown { actor, target, cards }`
   1. Meaning: a 4-card pass with card identities visible from this seat perspective.
5. `PassHidden { actor, target }`
   1. Meaning: same pass event, but cards hidden for this seat perspective.
6. `CardPlayed { actor, card, trick_index, pos_in_trick }`
   1. Meaning: actor played `card` at this exact trick position.
7. `RaiseToValue { actor, value }`
   1. Meaning: actor sets final announced game value in raising phase.
   2. Separation from `NewBid` is intentional for cleaner inference logic.
8. `AnnounceTrumpOwnPair { actor, suit, was_new_call }`
   1. Meaning: actor declares trump from own pair in `suit`.
   2. `was_new_call = false` means suit had already been called earlier.
9. `AskPair { actor, target, interaction_id }`
   1. Meaning: actor asks partner "Hast du ein Paar?".
10. `AskHalf { actor, target, suit, interaction_id }`
   1. Meaning: actor asks partner "Hast du eine Haelfte in `<suit>`?".
11. `AnswerPairYes { actor, suit, interaction_id }`
   1. Meaning: responder confirms pair in `suit`.
12. `AnswerPairNo { actor, interaction_id }`
   1. Meaning: responder confirms no pair (for legal pair-question context).
13. `AnswerHalfYes { actor, suit, interaction_id }`
   1. Meaning: responder confirms at least one half in `suit`.
14. `AnswerHalfNo { actor, suit, interaction_id }`
   1. Meaning: responder confirms no half in `suit`.
15. `QaResolution { interaction_id, outcome }`
   1. Meaning: normalized result of the QA interaction after considering backend callback and prior state.
   2. This is the canonical action for downstream trump/no-trump rule application.

## 5. Public Game State Schema (Ground Layer)

```rust
pub enum PublicPhase {
    WaitingForStart,
    Bidding,
    PassingForth,
    PassingBack,
    Raising,
    StartTrick,
    Trick,
    AnsweringPair,
    AnsweringHalf(Suit),
    Ended,
}

pub enum BidAction {
    NewBid(i32),
    StopBidding,
    RaiseToValue(i32),
}

pub struct FinishedPublicTrick {
    pub trick_index: u8,
    pub cards: Vec<(PlaceAtTable, Card)>,
    pub winner: PlaceAtTable,
    pub points: i32,
}

pub struct OpenQaInteraction {
    pub interaction_id: u32,
    pub asker: PlaceAtTable,
    pub responder: PlaceAtTable,
    pub question: QaQuestion,
}

pub struct PublicGameState {
    pub phase: PublicPhase,
    pub player_at_turn: PlaceAtTable,
    pub current_value: i32,
    pub bidding_history: Vec<(PlaceAtTable, BidAction)>,
    pub playing_party: Option<[PlaceAtTable; 2]>,
    pub current_trump: Option<Suit>,
    pub trump_called: std::collections::HashSet<Suit>,
    pub current_trick: Vec<(PlaceAtTable, Card)>,
    pub finished_tricks: Vec<FinishedPublicTrick>,
    pub hand_size: [u8; 4],
    pub open_interaction: Option<OpenQaInteraction>,
    pub start_ready: [bool; 4],
}
```

`PublicGameState` is the deterministic state maintained incrementally from normalized observed actions (plus perspective-dependent pass visibility).

It is the foundation for inference and must be updated incrementally on each new action. Full replay from `action_log` is a validation/debug fallback (and for undo recovery), not the default execution path.

Required fields and meaning:

1. `phase: PublicPhase`
   1. Current public game phase (`WaitingForStart`, `Bidding`, `PassingForth`, `PassingBack`, `Raising`, `StartTrick`, `Trick`, `AnsweringPair`, `AnsweringHalf(suit)`, `Ended`).
2. `player_at_turn: PlaceAtTable`
   1. Seat that is currently expected to act.
3. `current_value: i32`
   1. Current bid/announced value tracked publicly.
4. `bidding_history: Vec<(PlaceAtTable, BidAction)>`
   1. Ordered bidding actions.
5. `playing_party: Option<[PlaceAtTable; 2]>`
   1. Determined once bidding winner is known; `None` for no-one-plays branch.
6. `current_trump: Option<Suit>`
   1. Currently active trump suit.
7. `trump_called: HashSet<Suit>`
   1. Suits that have been called as trump at least once.
8. `current_trick: Vec<(PlaceAtTable, Card)>`
   1. Cards played in ongoing trick in order.
9. `finished_tricks: Vec<FinishedPublicTrick>`
   1. Completed tricks with winner and trick points.
10. `hand_size: [u8; 4]`
   1. Remaining hand size per seat (public count, not identities).
11. `open_interaction: Option<OpenQaInteraction>`
   1. Pending ask/answer context if currently inside QA flow.
12. `start_ready: [bool; 4]`
   1. Which players already issued `Start`.

Integration rule:

1. `KnowledgeState` must contain `public_state: PublicGameState`.
2. Inference rules consume `public_state` + source facts, never raw backend events directly.

## 6. Knowledge State Schema

```rust
pub struct KnowledgeState {
    pub seat: PlaceAtTable,
    pub public_state: PublicGameState,
    pub current_trump: Option<Suit>,
    pub trump_called: std::collections::HashSet<Suit>,
    pub game_value_bid: i32,
    pub bidding_history: Vec<(PlaceAtTable, BidAction)>,
    pub action_log: Vec<ObservedAction>,
    pub qa_interactions: Vec<QaInteraction>,
    pub trump_history: Vec<TrumpEvent>,
    pub trick_history: Vec<ObservedTrick>,
    pub current_trick_cards: Vec<(PlaceAtTable, Card)>,
    pub hand_size: [u8; 4],
    pub source_card_state: std::collections::HashMap<Card, SourceCardKnowledge>,
    pub derived_card_state: std::collections::HashMap<Card, DerivedCardKnowledge>,
    pub fact_provenance: Vec<DerivedFactRecord>,
    pub contradictions: Vec<String>,
}

pub struct SourceCardKnowledge {
    pub observed_owner: Option<PlaceAtTable>,
    pub observed_played_info: Option<(u8, PlaceAtTable, u8)>,
    pub observed_not_in_hand_mask: u8, // lower 4 bits for seats 0..3
}

pub struct DerivedCardKnowledge {
    pub known_holder: Option<PlaceAtTable>,
    pub possible_holders_mask: u8, // lower 4 bits for seats 0..3
    pub played_info: Option<(u8, PlaceAtTable, u8)>,
    pub not_in_hand_mask: u8, // lower 4 bits for seats 0..3
    pub standing_now: bool,
    pub standing_if_trump_static: bool,
    pub can_still_enable_future_trump: bool,
}

pub struct QaInteraction {
    pub interaction_id: u32,
    pub asker: PlaceAtTable,
    pub responder: PlaceAtTable,
    pub question: QaQuestion,
    pub answer: QaAnswer,
    pub resolution: QaResolution,
    pub resolved_trump: Option<(Suit, bool)>, // (suit, was_new_call)
}

pub struct SuitKnowledgeSummary {
    pub void_suits: [[bool; 4]; 4], // [player][suit_index]
    pub min_suit_count: [[u8; 4]; 4], // [player][suit_index]
    pub max_suit_count: [[u8; 4]; 4], // [player][suit_index]
}
```

`KnowledgeState` must contain at least:

1. `seat`: perspective seat.
2. `public_state: PublicGameState`.
3. `current_trump: Option<Suit>`.
4. `trump_called: HashSet<Suit>`.
5. `game_value_bid: i32` (last winning bid / announced value).
6. `bidding_history`.
7. `action_log: Vec<ObservedAction>`.
8. `qa_interactions: Vec<QaInteraction>` (one object per ask-answer-resolution block).
9. `trump_history` including actor, suit, source (`OwnCall`, `PairQuestion`, `HalfQuestion`), and `was_new_call`.
10. `trick_history` with all played cards and winners.
11. `current_trick_cards`.
12. `hand_size[player]` current remaining cards count.
13. `source_card_state[Card]` (ground-source facts only).
14. `derived_card_state[Card]` (rule-derived facts).
15. `fact_provenance` for each derived fact (`rule_id`, `event_index`, optional dependencies).

Ground-source vs derived must be explicit:

1. `source_card_state` may only contain facts directly observed by this seat:
   1. Own hand cards.
   2. Publicly played cards and play actor.
   3. Directly visible pass cards involving this seat.
   4. Public QA/trump declarations.
2. `derived_card_state` contains all inferences (void suits, not-in-hand, forced holders, standing, trump potential).
3. No derived fact may overwrite source fact; source fact has highest authority.

`source_card_state[Card]` should support:

1. `observed_owner: Option<PlaceAtTable>`
2. `observed_played_info: Option<(trick_index, player, position_in_trick)>`
3. `observed_not_in_hand[player]: bool` (only when directly observable, e.g. own hand exclusion)

`derived_card_state[Card]` should support:

1. `known_holder: Option<PlaceAtTable>`
2. `possible_holders: BitSet<PlaceAtTable>`
3. `played_info: Option<(trick_index, player, position_in_trick)>`
4. `not_in_hand[player]: bool`
5. `standing_now: bool`
6. `standing_if_trump_static: bool`
7. `can_still_enable_future_trump: bool` (mainly relevant for O/K cards)

`QaInteraction` should support:

1. `interaction_id`.
2. `asker`.
3. `responder`.
4. `question`: `AskPair` or `AskHalf(suit)`.
5. `answer`: `YesPair(suit)` / `NoPair` / `YesHalf(suit)` / `NoHalf(suit)`.
6. `resolution`: `TrumpCalled` / `NoTrump` / `OnlyResponderHasHalf`.
7. `resolved_trump: Option<(Suit, bool)>` where `bool = was_new_call`.

Derived suit/player summaries:

1. `void_suits[player][suit]` (known no cards of suit).
2. `min_suit_count[player][suit]`.
3. `max_suit_count[player][suit]`.
4. `known_cards_in_hand[player]`.
5. `possible_cards_in_hand[player]`.

## 7. Inference Engine Cycle

For each new observed action:

1. Apply direct event facts.
2. Run rule pass groups:
1. Play-legality rules.
2. Question/trump rules.
3. Set-theoretic closure rules.
4. Standing/trump-potential rules.
3. Iterate until no new fact is derived (fixed point).
4. Validate invariants:
1. No played card still possible in any hand.
2. Every unplayed card has at least one possible holder.
3. Per-player possible count >= hand size.
4. Per-player known count <= hand size.
5. If violated, surface contradiction.

Direct fact application must always happen in this order:

1. Write source facts (`source_card_state`).
2. Update normalized histories (`action_log`, `qa_interactions`, `trick_history`, `trump_history`).
3. Run incremental rule closure on changed facts to update `derived_card_state`.

### 7.1 Incremental reducer contract

1. `public_state` and `derived_card_state` are updated only from newly observed actions.
2. Processing is monotonic for forward-only streams:
   1. source facts are append-only.
   2. derived facts can only become stricter (`unknown -> possible -> excluded/known`).
3. Replay from `action_log` is used only for:
   1. integrity checks,
   2. undo rollback recovery,
   3. regression test verification.

### 7.2 Efficient rule execution model (pattern matching + agenda)

Use forward-chaining with a worklist/agenda and precondition indexing; do not scan every rule on every update.

```rust
pub enum Fact {
    KnownHolder(Card, PlaceAtTable),
    NotInHand(Card, PlaceAtTable),
    Played(Card, PlaceAtTable, u8, u8), // trick, pos
    SuitVoid(PlaceAtTable, Suit),
    SuitCountMin(PlaceAtTable, Suit, u8),
    SuitCountMax(PlaceAtTable, Suit, u8),
    HandSlotsRemaining(PlaceAtTable, u8),
    TrumpCurrent(Option<Suit>),
    TrumpCalled(Suit),
}

pub struct RuleDef {
    pub id: RuleId,
    pub watches: Vec<FactKind>, // precondition index keys
    pub apply: fn(&MatchCtx) -> Vec<Fact>, // emits only new facts
}

pub struct InferenceRuntime {
    pub fact_store: FactStore,
    pub fact_index: FactIndex, // by card, player, suit, kind
    pub watch_index: WatchIndex, // FactKind -> RuleId[]
    pub agenda: std::collections::VecDeque<Fact>,
}
```

Design constraint:

1. Do not persist abstract dominance-bound facts (like "not higher than X in suit S") in the long-lived fact store.
2. As soon as such a bound is inferred, materialize it into concrete `NotInHand(card, player)` facts for each affected card.
3. Persist only concrete, query-ready outcomes to avoid repeated reinterpretation.

Execution algorithm:

1. Insert new source/public facts into `agenda`.
2. Pop changed fact `f`.
3. Find subscribed rules via `watch_index[f.kind()]`.
4. For each candidate rule:
   1. Resolve variable bindings from `fact_index`.
   2. Check preconditions.
   3. Emit consequences.
5. Insert only consequences not already known.
6. Continue until agenda empty (fixed point).

Guarantees:

1. Work is proportional to changed facts and their dependent rules.
2. Rules fire only when preconditions could have changed.
3. No duplicate recomputation once fact is known.

### 7.3 Generalized invariant rules (suit/set/hand)

Rules must be template-based (parameterized by player/suit/value/card sets), not hardcoded for single card patterns.

Required invariant fact families:

1. Suit invariants:
   1. `SuitVoid(player, suit)`
   2. `SuitCountMin/Max(player, suit, n)`
   3. materialized suit-exclusion sets via `NotInHand(card, player)`
2. Hand invariants:
   1. `HandSlotsRemaining(player, n)`
   2. known/possible card count bounds
3. Set invariants:
   1. remaining cards per suit
   2. candidate-holder sets per card
   3. candidate-taker sets for trick-winning contexts

Example generalized rule (dominance materialization):

1. Preconditions:
   1. play-legality context infers that player `P` cannot hold any card above value `V` in suit `S`
   2. unplayed cards in suit `S` are known
2. Consequence:
   1. for every card `C` with `C.suit = S` and `C.value > V`, emit `NotInHand(C, P)`

Example generalized rule (your stated case pattern):

1. Preconditions:
   1. unplayed-card set in suit `Gruen` is known,
   2. one opponent already has explicit exclusions for all `Gruen` cards above `Neun`,
   3. another player already played `Gruen-Zehn`,
   4. hypothetical play context sets `Gruen` as trump.
2. Consequence:
   1. candidate-taker set for `Gruen-Unter` excludes players whose explicit exclusions prevent overtaking.

## 8. Full Rule Catalog

Rule IDs are normative. Implementation can split/merge internally, but behavior must match.

### 8.1 Base visibility and card-location rules

`R-BASE-001` Own hand certainty:
all cards in own hand are `known_holder = self`.

`R-BASE-002` Played card certainty:
when card is played, mark as `played_info` and remove all hand possibilities.

`R-BASE-003` Card uniqueness:
each card has exactly one true location; if `known_holder = p`, remove all others.

`R-BASE-004` Hand-size upper bound:
player cannot hold more cards than current `hand_size[player]`.

`R-BASE-005` Hand-size completion:
if known cards for player equal `hand_size[player]`, all other cards are `not_in_hand[player]`.

`R-BASE-006` Last possible holder:
if a card has only one possible holder left, assign that holder.

`R-BASE-007` Unplayed-card domain:
every unplayed card must be possible in at least one player's hand.

### 8.2 Bidding and party-structure rules

`R-BID-001` Playing party detection:
once bidding winner is determined, mark S-party seats and N-party seats.

`R-BID-002` Nobody-plays branch:
if all pass (value stays initial), mark no S/N split for this game.

`R-BID-003` Raise phase value:
final game value equals last valid bid/raise.

`R-BID-004` Bid history retention:
store full public sequence for analysis layers (not card-certainty by itself).

### 8.3 Passing rules (private visibility aware)

`R-PASS-001` Pass known by actor:
if seat performs pass action, exact passed cards are known.

`R-PASS-002` Pass known by receiver:
receiver knows exact incoming cards.

`R-PASS-003` Closed pass for uninvolved seats:
if pass cards are hidden, do not derive card identity facts from the pass event.

`R-PASS-004` Post-pass known ownership:
cards currently in own hand after pass remain certain self-owned.

`R-PASS-005` Partner certainty from own return-pass:
if seat sends cards to partner in return phase, those cards are known to be in partner hand immediately after pass.

`R-PASS-006` Exclusion from own retained cards:
any card known in own hand after pass is `not_in_hand` for all others.

### 8.4 Trick play legality rules

`R-PLAY-001` Must follow led suit:
if player did not play led suit, mark player void in led suit.

`R-PLAY-002` Must trump when void in led suit and trump exists:
if player played neither led suit nor trump while trump exists, mark player void in trump suit too.

`R-PLAY-003` Must overtake with led suit if possible:
if player followed led suit but did not overtake when overtaking with led suit would be mandatory, infer they had no such overtaking led-suit card.

`R-PLAY-004` Must overtake with trump if possible:
if player had to trump and played a non-overtaking trump, infer they had no trump card that could overtake current high.

`R-PLAY-005` First card of game constraint:
first leader must play Ass if any Ass exists in hand; else Gruen if any Gruen exists; else any card.
Observed violation alternatives imply absence:
1. If first lead is non-Ass, leader had no Ass.
2. If first lead is non-Ass and non-Gruen, leader had no Ass and no Gruen.

`R-PLAY-006` First trick Ass-of-led requirement:
on first trick, any non-leader who follows suit with non-Ass implies they do not hold Ass of led suit.

`R-PLAY-007` Play removes future ownership:
once player plays card C, player cannot hold C anymore (already covered by base, kept explicit for event application).

`R-PLAY-008` Trick winner consistency:
winner derived from cards and trump must match observed next leader; mismatch is contradiction.

`R-PLAY-009` Void persistence:
once a player is inferred void in a suit, remain void for rest of hand.

### 8.5 Question/answer/trump rules

`R-QA-001` AnnounceTrump(suit):
announcer currently has both `Ober(suit)` and `Koenig(suit)` in own hand at announcement moment.

`R-QA-002` Answer YesPair(suit):
answering partner currently has both `Ober(suit)` and `Koenig(suit)` at answer moment.

`R-QA-003` Answer NoPair:
answering player has no uncalled pair in hand at that moment.

`R-QA-004` Answer NoHalf(suit):
answering player has neither `Ober(suit)` nor `Koenig(suit)` at that moment.

`R-QA-005` Answer YesHalf(suit):
answering player has at least one of `Ober(suit)` or `Koenig(suit)` at that moment.

`R-QA-006` Callback NewTrump(suit):
set current trump to `suit`; add to called-suits if first time.

`R-QA-007` Callback StillTrump(suit):
trump remains `suit`; no new pair points; confirms prior call already existed.

`R-QA-008` Callback OnlyHalf(suit):
combined with `YesHalf`, infer asker has no half of that suit at that moment.

`R-QA-009` Callback NoHalf(suit):
combined with answer event, infer responder no-half fact (same as `R-QA-004`, but callback form supported).

`R-QA-010` New trump requires pair in announcing party:
at moment of new trump, both half cards of that suit are in hands of announcing party.

`R-QA-011` Suit call uniqueness:
each suit can be newly called trump at most once per game.

`R-QA-012` Played half card prevents future first-time trump in suit:
if either `Ober(suit)` or `Koenig(suit)` is already played before suit was called, suit can no longer become new trump.

`R-QA-013` Split-party half ownership blocks future first-time trump:
if `Ober(suit)` and `Koenig(suit)` are known to be in opposite parties before suit call, suit can no longer become new trump.

`R-QA-014` AskPair + YesPair sequence:
must resolve to trump call for the answered suit; if suit already called, mark contradiction (or no-op only if backend version allows it explicitly).

`R-QA-015` AskPair + NoPair sequence:
resolves to no trump call.

`R-QA-016` AskHalf + NoHalf sequence:
resolves to no trump call.

`R-QA-017` AskHalf + YesHalf + OnlyHalf callback:
resolves to no trump call and implies asker has no matching half at that moment.

`R-QA-018` AskHalf + YesHalf + NewTrump callback:
resolves to trump call for that suit (`was_new_call = true`).

`R-QA-019` AskHalf + YesHalf + StillTrump callback:
resolves to trump call semantics with `was_new_call = false` (already called earlier).

`R-QA-020` Trump suit uniqueness:
a suit can transition `uncalled -> called` at most once per game.

### 8.6 Set-theoretic closure rules

`R-SET-001` Exact hand capacity closure:
if player has `k` remaining slots and exactly `k` possible cards, assign all.

`R-SET-002` Card singleton closure:
if card possible in one player only, assign.

`R-SET-003` Suit max bound closure:
if max possible suit cards for player reaches known count, exclude remaining suit cards from that player.

`R-SET-004` Suit min bound closure:
if player must still hold `m` cards of suit and only `m` suit cards remain possible for that player, assign all.

`R-SET-005` Party-half closure:
if one half card location is fixed and rules force pair in same party for a future/just-announced trump condition, constrain the other half accordingly.

`R-SET-006` Full-deck conservation:
known hand cards + known played cards + unresolved cards equals 36 always.

`R-SET-007` Global consistency closure:
if any assignment causes impossible hand capacities, eliminate that assignment.

`R-SET-008` Forced-by-elimination (generalized):
apply repeated elimination until fixed point across all cards and players.

Implementation note:
`R-SET-*` can be implemented as deterministic constraint propagation with optional exact-check fallback (small CSP/bitset search) to derive globally forced facts safely.

### 8.7 Standing-card rules

`R-STAND-001` Standing-now (trick local):
card currently highest in ongoing trick under current trump is standing-now for this trick.

`R-STAND-002` Standing-with-static-trump:
an unplayed card is standing-if-trump-static if no higher same-suit card remains unplayed and no trump-overrule can apply in its expected play context.

`R-STAND-003` Absolute top trump:
highest remaining trump card is standing-if-trump-static whenever holder can follow legality constraints to play it.

`R-STAND-004` Standing revocation by future trump change:
if future trump change is still possible to another suit, downgrade absolute standing certainty to conditional standing.

`R-STAND-005` Guaranteed standing flag:
set only when card cannot be beaten by any legally possible future card under all still-possible trump evolutions.

### 8.8 Future trump-change potential rules

`R-TRP-001` Suit future-call possible:
a suit can still become future trump iff:
1. Suit not already newly called before.
2. Both half cards unplayed.
3. It is still possible both halves are in one party's hands at a future lead moment.

`R-TRP-002` Suit future-call impossible:
if any of conditions above fails, mark suit not future-callable.

`R-TRP-003` Card-level trump-potential flag:
for each half card, `can_still_enable_future_trump = true` iff its suit is future-callable and mate half is still unplayed and not logically blocked.

`R-TRP-004` Current trump can change:
current trump is marked unstable while at least one other suit remains future-callable.

`R-TRP-005` Current trump locked:
if no other suit future-callable, mark trump locked for remainder of game.

### 8.9 Invariant-template rule families

`R-INV-001` Dominance exclusion:
when a dominance bound is inferred, immediately emit explicit `NotInHand(card, player)` facts for all dominated cards; do not persist the bound as a standalone fact.

`R-INV-002` Suit remaining-set closure:
when all but one holder are excluded for cards in a suit subset, assign remaining holder(s) by capacity.

`R-INV-003` Hand-capacity saturation:
if `HandSlotsRemaining(player, n)` equals count of currently possible cards, assign all as known-holder.

`R-INV-004` Candidate-taker pruning:
in trick context, remove players from candidate-taker set when suit/trump constraints or explicit card exclusions make overtake impossible.

`R-INV-005` Candidate-taker fixation:
if candidate-taker set cardinality becomes 1, mark forced winner for that context.

## 9. Counterfactual Action-Impact Layer (Second Layer)

Goal: estimate knowledge gain/loss caused by own candidate actions.

For each legal own action `a`:

1. Simulate applying `a` to a cloned `KnowledgeState`.
2. Enumerate opponent response classes that are still consistent with knowledge.
3. For each branch, run inference closure.
4. Compute impact metrics:
1. `new_forced_cards_min/max/expected`
2. `entropy_delta` over card-holder distribution
3. `void_info_gain`
4. `trump_stability_delta`
5. `standing_guarantee_delta`

Output: rank actions by strategic objective plus information objective.

## 10. Data Types (Suggested Rust Shapes)

```rust
pub struct InferenceEngine {
    seat: PlaceAtTable,
    state: KnowledgeState,
    last_seen_event: usize,
}

pub struct KnowledgeState {
    pub seat: PlaceAtTable,
    pub public_state: PublicGameState,
    pub current_trump: Option<Suit>,
    pub trump_called: std::collections::HashSet<Suit>,
    pub game_value_bid: i32,
    pub current_trick_cards: Vec<(PlaceAtTable, Card)>,
    pub hand_size: [u8; 4],
    pub source_card_state: std::collections::HashMap<Card, SourceCardKnowledge>,
    pub derived_card_state: std::collections::HashMap<Card, DerivedCardKnowledge>,
    pub action_log: Vec<ObservedAction>,
    pub qa_interactions: Vec<QaInteraction>,
    pub trump_history: Vec<TrumpEvent>,
    pub trick_history: Vec<ObservedTrick>,
    pub bidding_history: Vec<(PlaceAtTable, BidAction)>,
    pub fact_provenance: Vec<DerivedFactRecord>,
    pub contradictions: Vec<String>,
}

pub struct PublicGameState {
    pub phase: PublicPhase,
    pub player_at_turn: PlaceAtTable,
    pub current_value: i32,
    pub bidding_history: Vec<(PlaceAtTable, BidAction)>,
    pub playing_party: Option<[PlaceAtTable; 2]>,
    pub current_trump: Option<Suit>,
    pub trump_called: std::collections::HashSet<Suit>,
    pub current_trick: Vec<(PlaceAtTable, Card)>,
    pub finished_tricks: Vec<FinishedPublicTrick>,
    pub hand_size: [u8; 4],
    pub open_interaction: Option<OpenQaInteraction>,
    pub start_ready: [bool; 4],
}

pub struct SourceCardKnowledge {
    pub observed_owner: Option<PlaceAtTable>,
    pub observed_played_info: Option<(u8, PlaceAtTable, u8)>,
    pub observed_not_in_hand_mask: u8,
}

pub struct DerivedCardKnowledge {
    pub known_holder: Option<PlaceAtTable>,
    pub possible_holders_mask: u8, // lower 4 bits used
    pub played_info: Option<(u8, PlaceAtTable, u8)>,
    pub not_in_hand_mask: u8, // lower 4 bits used
    pub standing_now: bool,
    pub standing_if_trump_static: bool,
    pub can_still_enable_future_trump: bool,
}

pub struct QaInteraction {
    pub interaction_id: u32,
    pub asker: PlaceAtTable,
    pub responder: PlaceAtTable,
    pub question: QaQuestion,
    pub answer: QaAnswer,
    pub resolution: QaResolution,
    pub resolved_trump: Option<(Suit, bool)>,
}
```

### 10.1 Implementation Order and Interfaces (normative)

Implement in this order:

1. `src/ai/types.rs`
   1. Define `ObservedAction`, `QaOutcome`, `QaInteraction`, `KnowledgeState`, card-knowledge structs.
2. `src/ai/observation.rs`
   1. Build `ObservationTracker` with `last_seen_event_index`.
   2. Implement `fn observe_new_actions(&mut self, events: &[GameEvent], seat: PlaceAtTable) -> Vec<ObservedAction>`.
   3. Implement QA sequence reconstruction and `interaction_id` assignment.
3. `src/ai/inference.rs`
   1. Build `InferenceEngine`.
   2. Implement `fn apply_observed_action(&mut self, action: &ObservedAction)`.
   3. Implement `fn run_closure(&mut self)` to fixed point.
4. `src/ai/rules.rs`
   1. Implement all `R-BASE`, `R-PLAY`, `R-QA`, `R-SET`, `R-STAND`, `R-TRP`.
   2. Each rule reports newly derived facts with provenance.
5. `src/ai/impact.rs`
   1. Add counterfactual branch evaluation over legal own actions.

Required public API for model-facing export:

1. `fn export_source_view(&self) -> SourceView`
2. `fn export_derived_view(&self) -> DerivedView`
3. `fn export_training_frame(&self) -> TrainingFrame`

`TrainingFrame` must keep source and derived features separated (do not flatten away provenance).

## 11. Test Specification

### 11.1 Unit tests per rule

1. One focused test per rule ID.
2. Minimal fixture with exact pre-state, action, expected derived facts.

### 11.2 Scenario tests (full games)

1. Fully deterministic fixed-deck games that include:
1. No-trump first trick behavior.
2. Pass forth/back information.
3. All question/answer variants.
4. Trump changes across multiple suits.
5. Edge cases with forced undertrump and overtake rules.

2. For each seat:
1. Replay full action stream incrementally.
2. Assert no contradictions.
3. Assert all hard-derived facts are true in omniscient ground truth.
4. Assert unknown facts are never incorrectly forced.

### 11.3 Property tests

1. Random legal games from engine.
2. For each event and each seat:
1. Inferred known-holder cards must match true holder.
2. Inferred not-in-hand facts must match truth.
3. Hand capacity invariants always hold.

### 11.4 Regression snapshots

1. Store serialized knowledge states after each action for curated games.
2. Detect unintended inference changes across refactors.

## 12. Milestone Plan

1. Milestone 1: `observation.rs` + core `KnowledgeState`.
2. Milestone 2: implement `R-BASE`, `R-PLAY`, `R-SET` minimal fixed point.
3. Milestone 3: add `R-QA`, `R-TRP`, `R-STAND`.
4. Milestone 4: action-impact layer.
5. Milestone 5: full-game test suite and regression fixtures.

## 13. Non-goals (for this spec version)

1. Bidding-style heuristic inference.
2. Opponent model probabilities in core deterministic layer.
3. Search policy for best move selection (separate agent layer).
