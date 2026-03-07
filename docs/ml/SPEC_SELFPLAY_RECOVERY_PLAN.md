# Self-Play Recovery Plan

## Context
- Current online training can collapse into a low-risk `pass` equilibrium late in `full_game`.
- The existing codebase already has strong observation features, hidden-state supervision, and parallel GPU inference.
- The dominant failure is not model capacity. It is a mismatch between what the model is rewarded/trained on and what it is later asked to do in self-play.

## Problem Statement
The current pipeline has four concrete failure modes:

1. `full_game` counterfactual targeting is too narrow.
   - In `ml/train_online.py`, targeted advantage queries in `full_game` only hit info actions.
   - Bid and pass choices therefore receive mostly delayed terminal signal.

2. Inference policy and training policy are inconsistent.
   - Batched inference in `ml/train/pool.py` samples from unmasked logits.
   - Padded/illegal actions can receive probability mass.
   - This silently distorts self-play policy quality and makes training targets inconsistent with runtime behavior.

3. Entropy is computed from unmasked logits in training.
   - `ml/train/loss.py` masks logits for log-probs but still computes entropy from raw logits.
   - This rewards distribution mass on invalid actions.

4. Bid calibration logic is duplicated and incomplete.
   - Training already has a bid soft-cap regularizer, but inference does not.
   - There is no shared mechanism discouraging `StopBidding` when the model predicts that a legal raise is still makeable.

## Goals
- Keep game rules unchanged.
- Keep 24 workers and 4 MC rollouts for the primary training profile.
- Improve learning signal for bidding and passing without hard-coding a human heuristic policy into the model.
- Remove training/inference inconsistencies and avoid duplicated adjustment logic.

## Non-Goals
- No ruleset changes in Rust game logic.
- No redesign of the observation encoder or model family.
- No heavy search-based inference in the main self-play loop.

## Implementation Plan

### Workstream 1: Fix `full_game` credit assignment
- Add a dedicated helper for advantage-target selection in `ml/train_online.py`.
- Preserve current curriculum semantics:
  - `start_trick == -1`: bidding targets
  - `start_trick == 0`: passing targets
  - `start_trick == N`: trick `N` plus info actions
- Extend `start_trick is None` (`full_game`) so target actions include:
  - bidding
  - passing
  - info/trump questions

Expected effect:
- Bid/pass decisions receive direct counterfactual supervision in the long-horizon phase instead of being dominated by terminal reward.

### Workstream 2: Unify bid-policy calibration in train and infer
- Keep the shared helper in `ml/train/policy_adjust.py`.
- Route both training and runtime inference through the same adjustment layer.
- Support two model-consistency mechanisms:
  - soft-cap for bids above predicted make-value
  - stop-bid penalty when a legal raise appears comfortably makeable

Expected effect:
- The model is nudged toward internally coherent bidding based on its own predicted team value, not by a fixed heuristic table.

### Workstream 3: Remove invalid-action probability leakage
- In `ml/train/loss.py`:
  - compute entropy from masked logits
- In `ml/train/pool.py`:
  - sample from masked and adjusted logits

Expected effect:
- Illegal or padded actions no longer distort policy entropy or runtime sampling.

### Workstream 4: Add run profile for a fresh training restart
- Add a new `just` recipe that:
  - starts from the existing pretraining checkpoint
  - keeps `workers=24` and `mc_rollouts=4`
  - increases bid/pass/info counterfactual coverage
  - reduces long forced-heuristic windows
  - enables the shared stop-bid calibration knobs
  - makes `pass` less attractive without turning it into a forbidden move

Expected effect:
- The restart profile is explicit, reproducible, and separate from older recipes.

## Acceptance Criteria
- `full_game` advantage targeting covers bid/pass/info actions.
- Batched inference never samples from masked actions.
- Training entropy is computed on masked logits.
- Bid soft-cap and stop-bid penalty use the same helper in training and inference.
- A dedicated fresh-run `just` recipe exists with 24 workers and 4 rollouts.
- Unit tests cover:
  - bid consistency helper
  - `full_game` target selection
  - batched inference mask handling

## Validation Plan
- Run focused Python tests for the new helper and inference masking.
- Start a fresh online run from the pretraining checkpoint.
- Watch these late-phase indicators:
  - `PassGames`
  - `AvgBid`
  - `Made`
  - overbid / bid-vs-predicted stats
- Evaluate with `just eval-fixed` on the 100-game deterministic suite.
