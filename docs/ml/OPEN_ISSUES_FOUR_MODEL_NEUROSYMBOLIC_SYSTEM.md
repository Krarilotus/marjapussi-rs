# Open Issues: Four-Model Neurosymbolic Training System

## Closed decisions
- Use four fully separate models:
  - `BeliefModel`
  - `BiddingModel`
  - `PassingModel`
  - `PlayingModel`
- Use a strict authoritative symbolic inference layer.
- Use a hybrid canonical structured state:
  - global,
  - cards,
  - players/teams,
  - strategy/meta.
- Use structured, interpretable state representation rather than dense-only tensors.
- Use belief output for:
  - card section,
  - selected player-level uncertainty aggregates.
- Use hard symbolic masking plus global conflict-free assignment in belief decoding.
- Train decision models with:
  - policy,
  - value/Q,
  - phase-specific auxiliary heads.
- Human-first lifecycle:
  - decision models pretrained first on human games,
  - then belief on the full human dataset,
  - then simulation-only co-evolution.
- Human pretraining is primarily imitation, with player-quality weighting and local target reconstruction.
- Player-quality weighting is hybrid and primarily player-based.
- Joint phase is asymmetrical:
  - early belief-heavy,
  - later decision-heavy.
- Belief promotion gate is hybrid:
  - min duration,
  - hard metrics,
  - aggregate metrics,
  - calibration metrics.
- After human pretraining, joint phase uses simulated data only.

## Remaining design questions

### 1. Rule-catalog boundary
- Which strategic interval/bound facts belong in Layer 2 versus auxiliary targets computed later?
- Recommendation:
  - keep only deterministic or sound-bounded meta facts in Layer 2,
  - move heuristic scoring functions to training/eval utilities.

### 2. Belief decoder algorithm choice
- Candidate implementations:
  - Hungarian-style assignment on score matrix,
  - constrained greedy plus repair,
  - min-cost flow.
- Recommendation:
  - prototype constrained greedy plus repair first for speed,
  - keep min-cost flow as fallback if consistency quality is insufficient.

### 3. Phase-boundary ownership
- Some actions couple phases semantically, especially:
  - late bidding implications for passing,
  - info/trump choices during play.
- Recommendation:
  - keep ownership by action family, not by abstract “stage name”, and expose cross-phase diagnostics.

### 4. Model-size budget
- Four separate models add inference and checkpoint complexity.
- Recommendation:
  - keep the belief model compact and the phase models narrow before scaling width/depth.

### 5. Selection score definition
- Best-checkpoint selection must avoid degenerate but numerically stable policies.
- Recommendation:
  - define two saved winners:
    - `best_internal_eval.pt`
    - `best_fixed_suite.pt`
  - and a promoted serving checkpoint chosen by a behavior-aware composite score.
