# Hidden-Hand Inference Engine

This module is the set-theory inference engine used by ML observations.

It now also serves as the authoritative symbolic core for the structured
neurosymbolic state exported by `src/ml/state.rs`.

## Entry Point

- `apply_hidden_set_constraints(...)`
  - File: `src/ml/inference/engine.rs`
  - Called from: `src/ml/observation.rs` in `build_observation(...)`
- `InferenceState`
  - File: `src/ml/inference/state.rs`
  - Purpose: explicit monotonic/delta-friendly symbolic inference state wrapper for the four-model pipeline

## Terminology Mapping (Code -> Game Meaning)

- `HiddenPossibleMask`:
  - `possible_bitmasks[seat][card]`
  - "Card may still be in that hidden seat."
- `HiddenConfirmedMask`:
  - `confirmed_bitmasks[seat][card]`
  - "Card is known to be in that hidden seat."
- `HalfConstraintGrid`:
  - Q&A-derived constraints per `(hidden seat, suit)`.
- `HalfConstraint::RequireAtLeastOne`:
  - Seat must hold at least one of `(Ober, King)` in that suit.
- `HalfConstraint::RequireBoth`:
  - Seat must hold both `(Ober, King)` in that suit.

Hidden seat indices are always POV-relative:

- `0 = left`
- `1 = partner`
- `2 = right`

## Rulebook (Execution Order)

Rules are listed in `engine.rs` in the `RULES` array and implemented in `rules.rs`.

1. `rule_confirmed_consistency`
   - Confirmed cards must remain possible.
   - A card cannot be confirmed in multiple hidden seats.
2. `rule_half_constraints`
   - Applies half/pair Q&A constraints to `possible` and `confirmed`.

After fixpoint iteration, `finalize_confirmed_projection` runs as a final consistency pass.

## Where To Extend

Add new inference rules in:

- `src/ml/inference/rules.rs`

Then register the rule in:

- `RULES` array in `src/ml/inference/engine.rs`

Keep each rule:

- deterministic
- monotonic where possible
- local (single responsibility)

That keeps the inference calculus readable and testable.

## Relationship To Canonical State

- `src/ml/state.rs`
  - builds the structured `CanonicalState`
  - reuses observation-derived hidden-hand knowledge
  - lifts the low-level masks into:
    - card section
    - player section
    - team section
    - strategy/meta section

This keeps the fast mask-level inference small and cache-friendly while exposing
an interpretable higher-level state to dataset conversion and future model
pipelines.

## Delta-Oriented API

`InferenceState` is the first explicit Layer-2 scaffold for the new
neurosymbolic stack.

It provides:

- persistent `possible`, `confirmed`, and `half_constraints` masks
- `InferenceDelta` for monotonic updates
- explicit played-card projection
- helpers for:
  - `possible_hidden_rel(card_idx)`
  - `confirmed_hidden_rel(card_idx)`
  - `void_suits()`

This keeps the authoritative symbolic state in Rust and avoids pushing rule
maintenance into Python or into the models themselves.
