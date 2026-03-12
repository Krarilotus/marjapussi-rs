# Bidding Rule Teacher

This document defines the explicit rule-derived supervision now used for the
`BiddingModel`. The goal is not to hard-force runtime bidding, but to make the
training signal reflect common Marjapussi bidding semantics instead of learning
only from delayed win/loss rewards.

## Purpose

The bidding model previously learned mostly from:

- imitation,
- generic makeable-bid bounds,
- final game outcome.

That allowed stable local minima such as:

- `always pass`,
- `mostly 120`,
- `120/125` with weak partner-information updates.

The bidding rule teacher adds explicit local supervision for the meaning of bid
steps.

## Encoded Common Rules

The current teacher encodes the following core rules:

1. First team bid:
- `+5` if the bidder holds at least one Ace.
- `+10` for either:
  - `3` or `4` unconnected halves, or
  - one small pair (`Green` or `Acorns`).
- `+15` for a big pair (`Bells` or `Red`).

2. Second team bid:
- `+5` only if the team already opened with an Ace-style `+5` and the player has
  exactly the weaker support pattern (`2` unconnected halves, no stronger jump).
- `+10` / `+15` still dominate if the player has the corresponding stronger
  structure.

3. Later team bids:
- conservative continuation by `+5` if the player still has clear strength
  (Ace, pair, or at least `2` unconnected halves).

4. Over-`140` gate:
- do not treat `>140` as safe unless pair evidence can be deduced for the team,
  e.g.:
  - partner first jumped `+15`,
  - partner first jumped `+10` and own halves imply a forced pair,
  - double-`+5` plus own strong follow-up support,
  - no-Ace exception only with stronger forced pair structure.

5. Conservative estimated value:
- use a cautious hand-value estimate from:
  - own Aces,
  - Tens,
  - Kings,
  - Obers,
  - pair points / 2,
  - small partner jump bonuses.
- cap the estimate conservatively.

## Structured State Fields

These bidding semantics are materialized in `canonical_state.strategy`:

- `bidding_current_highest_bid`
- `bidding_team_bid_count`
- `bidding_self_bid_count`
- `bidding_partner_bid_count`
- `bidding_team_first_step`
- `bidding_self_first_step`
- `bidding_partner_first_step`
- `bidding_team_has_ace_signal`
- `bidding_allow_over_140`
- `bidding_estimated_value`
- `bidding_recommended_step`
- `bidding_own_ace_count`
- `bidding_own_unmatched_halves`
- `bidding_own_small_pair_count`
- `bidding_own_big_pair_count`

## Training Signals

The bidding trainer now uses three aligned signals:

1. Human imitation:
- still the primary target.

2. Rule-derived auxiliary targets:
- makeable floor/ceiling,
- Ace-signal present,
- over-`140` admissibility,
- recommended next step,
- estimated value,
- partner first step,
- team bid count,
- unmatched halves,
- big-pair count.

3. Rule-derived teacher policy:
- prefer the legal action matching the recommended next bid if it is admissible,
- otherwise prefer `StopBidding`.

This creates explicit supervision for the informational meaning of bidding steps
without turning the runtime policy into a rigid heuristic bot.
