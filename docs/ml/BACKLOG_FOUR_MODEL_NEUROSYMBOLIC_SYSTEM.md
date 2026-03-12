# Backlog: Four-Model Neurosymbolic Training System

## Epics (prioritized)
1. Epic: Canonical state and symbolic inference foundation
- Outcome:
  - one authoritative, structured, debuggable state contract
- Scope:
  - canonical schema, symbolic fact engine, delta updates, state logging
- Dependencies:
  - current Rust engine observation/replay path
- Risks:
  - wrong symbolic rules poison all downstream models

2. Epic: Belief model and conflict-free decoder
- Outcome:
  - hidden-card prediction is consistent, constrained, and trainable
- Scope:
  - belief dataset generation, decoder, metrics, calibration
- Dependencies:
  - canonical state and symbolic inference
- Risks:
  - decoder latency and calibration instability

3. Epic: Split decision models and human pretraining
- Outcome:
  - bidding, passing, and playing are trained as distinct tasks
- Scope:
  - separate model configs, losses, heads, replay datasets, player-quality weighting
- Dependencies:
  - canonical state, belief interface
- Risks:
  - dataset extraction complexity and phase mislabeling

4. Epic: Joint simulation-only co-evolution
- Outcome:
  - all four models improve together without pass/minimal-bid collapse
- Scope:
  - asymmetrical training schedule, stability gates, behavior-aware eval/selection
- Dependencies:
  - belief and decision-model pretraining complete
- Risks:
  - collapse remains possible if selection/gating is weak

5. Epic: Evaluation and governance
- Outcome:
  - best checkpoints are selected on real behavior rather than misleading internal metrics
- Scope:
  - fixed-suite score, anti-collapse score, checkpoint governance, debug reports
- Dependencies:
  - all prior epics
- Risks:
  - bad scoring still promotes degenerate policies

## MVP (must-have)
- P0-1: Define `CanonicalState` schema and versioning
  - Status:
    - done
  - Acceptance criteria:
    - structured sections `global/cards/players/teams/strategy`
    - serializable and diffable
    - schema version validated at load time
  - Notes:
    - land before model changes

- P0-2: Implement Layer-2 symbolic inference engine scaffold
  - Status:
    - in progress; explicit `InferenceState` / `InferenceDelta` scaffold landed, canonical-state transport is live, full event-driven runtime replacement still pending
  - Acceptance criteria:
    - monotonic fact update API
    - delta-based update path
    - initial hard facts and impossibility masks

- P0-3: Build symbolic rule catalog and test harness
  - Acceptance criteria:
    - rule catalog versioned
    - replay-based tests for non-follow/question/announcements/played cards
    - contradiction test suite exists

- P0-4: Create human replay dataset pipeline for canonical state
  - Status:
    - done for `canonical_state` + `belief_targets` export
  - Acceptance criteria:
    - replay human games through current engine
    - export phase-tagged training examples
    - attach player-quality weights

- P0-5: Implement `BeliefModel` I/O contract
  - Status:
    - in progress
  - Acceptance criteria:
    - input is structured symbolic state
    - output covers card section and player uncertainty aggregates
    - teacher supervision uses exact reconstructed hidden state

- P0-6: Implement conflict-free belief decoder
  - Status:
    - partial; exact capacity-constrained hidden-card decoder landed, live runtime/eval/UI integration is in place, training-loop serving integration still pending
  - Acceptance criteria:
    - hard symbolic mask first
    - global assignment enforces one owner per unknown card
    - remaining hand sizes respected

- P0-7: Define belief metrics and stability gate
  - Status:
    - in progress; metric schema and asymmetrical gate/schedule primitive landed, trainer integration still pending
  - Acceptance criteria:
    - card-location accuracy
    - constraint consistency
    - player aggregate metrics
    - calibration metrics
    - hybrid gate with min-duration + thresholds

- P0-8: Split decision models into three separate training targets
  - Status:
    - in progress; decision-state compiler, separate model shells, and human-pretraining trainer landed
  - Acceptance criteria:
    - independent `BiddingModel`, `PassingModel`, `PlayingModel`
    - each with policy + value/Q + auxiliary heads
    - all consume combined symbolic+belief state

- P0-9: Human pretraining for decision models
  - Status:
    - in progress; imitation-first trainer with player-winrate weighting and phase-local aux targets landed
  - Acceptance criteria:
    - imitation is primary loss
    - player-quality weighting active
    - local target reconstruction added per phase

- P0-10: Human pretraining for belief model
  - Status:
    - in progress; standalone trainer landed and is now wired into the stage-1 four-model orchestrator
  - Acceptance criteria:
    - full human dataset seen at least once before joint phase
    - metrics logged for stability gate

- P0-11: Joint training coordinator
  - Status:
    - in progress; asymmetrical schedule primitive, simulated-data joint trainer, and fixed-suite governance wiring landed; direct live-serving integration still pending
  - Acceptance criteria:
    - start with belief-heavy schedule
    - switch to decision-heavy only when gate passes
    - after human stages, training uses simulated data only

- P0-12: Behavior-aware checkpoint selection
  - Status:
    - in progress; base scoring primitive, fixed-deal/runtime hooks, and manifest promotion logic landed in the joint/end-to-end path; legacy online trainer integration still pending
  - Acceptance criteria:
    - score combines point diff with pass/minimal-bid collapse penalties
    - `best_eval.pt` and `best_fixed_suite.pt` are separately stored

## Next (should-have)
- P1-1: Structured debug UI / report for exact vs symbolic vs belief state
- P1-2: Fixed-suite selection integrated directly into training loop
  - Status:
    - partial; manifest-aware fixed-suite eval and promotion are now wired into the four-model joint/end-to-end loops, but not into the legacy monolithic trainer
- P1-3: Phase-specific calibration reports for bidding/passing/playing
- P1-4: Optional tiny Monte Carlo branch evaluator for offline diagnostics only
- P1-5: Cross-model consistency checks, for example bid intent vs passing behavior

## Later (could-have)
- P2-1: Distillation from optional search tool into offline targets
- P2-2: Learned confidence estimator over symbolic strategic bounds
- P2-3: Automatic rule-catalog mining suggestions from replay anomalies

## First Sprint Plan
- Task 1:
  - write canonical schema and state builders
  - Status:
    - done
- Task 2:
  - implement Layer-2 rule engine scaffold with monotonic update tests
  - Status:
    - partial; delta-update scaffold and monotonic tests landed, event-level integration still pending
- Task 3:
  - extend human replay conversion to export canonical structured examples
  - Status:
    - done
- Task 4:
  - define belief dataset targets and decoder interface
  - Status:
    - partial; dataset targets, model I/O, and exact hidden-hand decoder landed, runtime integration still pending
- Task 5:
  - land separate human pretraining path for bidding/passing/playing
  - Status:
    - partial; generic trainer and stage-1 human orchestrator landed, checkpoint manifest/orchestration is now explicit
- Task 6:
  - land benchmark/perf harness for symbolic and belief latency
  - Status:
    - partial; four-model runtime micro-benchmark landed, online self-play latency harness still pending
- Task 7:
  - expose canonical state through live ml_server/env transport and runtime consumers
  - Status:
    - done
- Task 8:
  - make fixed-eval and UI load either legacy checkpoints or four-model manifests
  - Status:
    - done
- Task 9:
  - generate simulated self-play datasets directly from a four-model manifest
  - Status:
    - done; generator now supports explicit phase-balanced `full`/`bidding`/`passing` mixes so passing coverage does not depend on accidental policy behavior
- Task 10:
  - run end-to-end cycles that refresh simulated data before each joint-training step
  - Status:
    - done
