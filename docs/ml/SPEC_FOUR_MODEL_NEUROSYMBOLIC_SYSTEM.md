# Spec: Four-Model Neurosymbolic Training System

## 1. Problem
- The current training stack can learn stable but strategically bad equilibria such as `always pass` or `minimal 120 contracts`.
- Pure PPO-style delayed reward is too weak for bidding, passing, information exchange, and contract calibration.
- The current end-to-end setup does not explicitly separate:
  - rule-correct inference,
  - hidden-card belief completion,
  - phase-specific decision making.
- The resulting system is hard to debug because model errors, missing inference, and reward pathologies are entangled.

## 2. Goals
- Build a neurosymbolic architecture that separates:
  - deterministic inference,
  - conflict-free hidden-card prediction,
  - phase-specific decision policies.
- Preserve efficient simulation throughput in online training.
- Make every intermediate representation interpretable and directly testable.
- Improve strategic quality in:
  - bidding,
  - passing,
  - information exchange,
  - trick play.
- Ensure training runs are operationally controllable and phase durations are bounded.

### Success metrics
- `PassGames` does not collapse into dominant equilibrium in late self-play.
- `AvgBid` does not collapse toward `120` in healthy late training.
- `Made`, fixed-suite point diff, and trump/info usage improve versus current PPO pipeline.
- Belief model satisfies hard consistency constraints and stable calibration thresholds.
- Simulation throughput remains within the current efficient regime for online training.

## 3. Non-Goals
- No heavy search-first training or inference architecture in the main loop.
- No Monte Carlo tree expansion with large branching factors in online training.
- No replacement of the Rust rules engine as the source of truth.
- No opaque dense-only hidden state that cannot be inspected or diffed.

## 4. Users & Permissions
- ML engineer:
  - define schema versions,
  - run training,
  - inspect state and belief diffs,
  - tune phase/model schedules.
- Game/rules engineer:
  - extend symbolic inference rules,
  - validate monotonic correctness,
  - verify invariants against the Rust engine.
- Evaluator/operator:
  - run fixed-suite evaluations,
  - compare checkpoints,
  - inspect collapse guards and health metrics.

## 5. Scope & Assumptions
### In scope
- Four fully separate learned models:
  - `BeliefModel`
  - `BiddingModel`
  - `PassingModel`
  - `PlayingModel`
- A deterministic, authoritative symbolic inference layer.
- A canonical structured state representation shared across the system.
- Human-first pretraining followed by simulated-data-only co-evolution.
- New checkpoint selection and anti-collapse governance.

### Out of scope
- Large search teacher in the standard simulation path.
- End-to-end single-network replacement of all phases.
- Rule changes in the underlying game.

### Assumptions
- Human gameplay data exists and can be replayed through the current engine.
- The Rust engine remains the source of legal actions and rule transitions.
- Existing GPU inference infrastructure can host multiple small models efficiently.

### Constraints
- Efficient simulation is mandatory.
- Search, if used at all, is limited to small Monte Carlo branching and optional tools, not the main training dependency.
- Symbolic inference must be monotonic and incrementally updatable from game deltas.

## 6. Current Implementation Seed
- Landed:
  - canonical structured Rust state in `src/ml/state.rs`
  - replay export of `canonical_state` and separate `belief_targets`
  - explicit monotonic `InferenceState` / `InferenceDelta` scaffold in Rust
  - Python parsers and fast belief-dataset bridge
  - standalone `BeliefNet` pretraining path
  - exact small-state conflict-free belief decoder scaffold
  - decision-state compiler and separate `BiddingNet` / `PassingNet` / `PlayingNet` shells
  - imitation-first human-pretraining path for separate decision models with player-quality weighting
  - stage-1 four-model human-pretraining orchestrator that runs decision models first, then belief
  - simulated-only joint-training coordinator that continues from a human-pretrained manifest
  - simulated self-play dataset generator for the four-model stack
  - end-to-end four-model coordinator that regenerates self-play data each cycle
  - runtime bridge that injects decoded belief output into split decision features
  - manifest loader and runtime bundle for the four-model artifact set
  - asymmetrical joint-phase schedule primitive with explicit belief-stability gate
  - behavior-aware scoring primitive for anti-collapse checkpoint governance
  - live `ml_server -> env.py` transport of `canonical_state`
  - manifest-aware fixed-deal evaluator
  - manifest-aware UI loading path
- Not landed yet:
  - full event-native replacement of legacy hidden inference with `InferenceDelta` updates in the live engine loop
  - direct four-model online self-play serving inside the existing production trainer
  - direct replacement of the legacy monolithic trainer with the four-model runtime

## 6. Functional Requirements

### 6.1 Core entities

#### Entity: CanonicalState
- Sections:
  - `global`
  - `cards`
  - `players`
  - `teams`
  - `strategy`
- Identifiers:
  - `schema_version`
  - `game_id`
  - `ply_index`
  - `seat_pov`
- Lifecycle:
  - created from Rust `GameState`
  - enriched by Layer 2 symbolic inference
  - completed by Layer 3 belief output
  - consumed by one Layer 4 decision model

#### Entity: SymbolicState
- Definition:
  - the authoritative deterministic subset of `CanonicalState`
- Contents:
  - hard facts,
  - derived concepts,
  - strategic interval/bound information,
  - no contradictions

#### Entity: BeliefState
- Definition:
  - conflict-free completion of unknown card locations and selected player-level uncertainty fields
- Constraints:
  - may never contradict `SymbolicState`
  - every unknown card has exactly one decoded location

#### Entity: TrainingExample
- Fields:
  - `symbolic_state`
  - `belief_input`
  - `phase`
  - `legal_actions`
  - `human_action` or `self_play_action`
  - phase-local targets
  - final outcome targets
  - player-quality weight

### 6.2 Layered architecture

#### Layer 1: Raw GameState
- Source:
  - current Rust engine state plus event history
- Responsibilities:
  - exact rules,
  - legal actions,
  - exact scoring,
  - exact game transition,
  - exact observed cards/events

#### Layer 2: Authoritative Symbolic Inference
- Responsibilities:
  - apply only correct inference rules from the rule catalog,
  - monotonically accumulate deducible facts,
  - update incrementally from state deltas,
  - produce structured, inspectable output.

##### Required fact classes
- Hard card/location facts:
  - card is in hand,
  - card is played,
  - card cannot be in seat X,
  - seat must hold one of set S,
  - current remaining hand size constraints.
- Derived game concepts:
  - standing cards,
  - exhausted halves,
  - trump availability and exhaustion,
  - void/free suits,
  - forced assignments from non-follow, question/answer, bidding implications.
- Strategic meta facts and bounds:
  - bidable range,
  - lower/upper estimated contract value,
  - pass target candidates,
  - likely standing-card potential,
  - information opportunity intervals.

##### Inference engine constraints
- Pattern-matching based.
- Delta-based and incremental.
- Monotonic with respect to game progression.
- No rule may invalidate a previously correct hard fact except where the game transition itself removes availability, such as played cards leaving hands.

#### Layer 3: BeliefModel
- Input:
  - structured `SymbolicState`
- Output:
  - predicted card locations for unknown cards,
  - player-level uncertainty aggregates:
    - void probabilities,
    - suit-distribution estimates,
    - trump/half possession probabilities.
- Decoding:
  - hard symbolic masking first,
  - then global conflict-free assignment/matching.

##### Belief invariants
- Symbolically fixed information is passed through unchanged.
- Symbolically impossible placements cannot be predicted.
- Unknown cards cannot be assigned to multiple owners.
- Decoded state must satisfy remaining hand-size constraints.

#### Layer 4: Decision Models
- `BiddingModel`
- `PassingModel`
- `PlayingModel`

Each consumes:
- authoritative symbolic state,
- conflict-free belief completion,
- legal actions for the current phase.

### 6.3 Model outputs

#### BiddingModel
- Outputs:
  - policy over legal bid/pass actions,
  - value/Q head,
  - `makeable_value`,
  - `contract_success`,
  - `overbid_risk`,
  - `underbid_risk` or stop-bid insufficiency proxy.

#### PassingModel
- Outputs:
  - policy over legal pass-card selections,
  - value/Q head,
  - `information_gain`,
  - `team_shape_gain`,
  - `standing_card_gain`,
  - `partner_helpfulness`.

#### PlayingModel
- Outputs:
  - policy over legal play/info/trump actions in play phase,
  - value/Q head,
  - `trick_win`,
  - `standing_card_realization`,
  - `control_gain`,
  - `tempo` or initiative proxy.

### 6.4 Workflows

#### Workflow A: Runtime inference
1. Rust engine emits current exact state.
2. Layer 2 updates `SymbolicState` from the previous state plus delta.
3. `BeliefModel` predicts only unresolved hidden-card structure.
4. Constraint decoder merges Layer 2 and Layer 3 into a consistent combined state.
5. Appropriate phase model consumes combined state and scores legal actions.
6. Action is sampled/selected and applied by the Rust engine.

#### Workflow B: Human pretraining
1. Replay human games through current rules engine.
2. Build `CanonicalState` at each decision point.
3. Generate player-quality weights.
4. Train decision models primarily by imitation, with phase-local reconstruction targets.
5. After decision-model human pretraining, train `BeliefModel` on the full human dataset.

#### Workflow C: Joint co-evolution
1. Start only after the belief model has seen the entire human dataset.
2. Early joint phase is belief-heavy.
3. Later joint phase becomes decision-heavy once belief passes stability gate.
4. Training uses only simulated data after human pretraining is complete.

### 6.5 Requirements list

#### FR-1: Canonical structured state
- The system must expose a canonical structured state with separate global, card, player/team, and strategy sections.
- Acceptance criteria:
  - state can be serialized,
  - state can be diffed between plies,
  - schema version is carried end-to-end.

#### FR-2: Authoritative symbolic inference
- Symbolic inference must only add correct facts and may not emit contradictions.
- Acceptance criteria:
  - monotonic update tests pass,
  - rule regression suite exists,
  - played-card removal is handled as a state transition, not as a contradiction.

#### FR-3: Conflict-free belief decoding
- The belief layer must output globally consistent hidden-card assignments.
- Acceptance criteria:
  - no card assigned to multiple players,
  - no assignment violates symbolic constraints,
  - remaining hand sizes remain valid.

#### FR-4: Phase-separated decision models
- Bidding, passing, and playing must be separate trainable models.
- Acceptance criteria:
  - each has independent checkpointing,
  - each uses the combined state interface,
  - each exposes phase-specific heads.

#### FR-5: Human-first training lifecycle
- Human pretraining must happen before simulation-only co-evolution.
- Acceptance criteria:
  - decision models can be pretrained independently on human data,
  - belief model sees the full human dataset before joint phase starts,
  - no simulated-only joint run may start without this gate.

#### FR-6: Player-quality weighted imitation
- Human data must be weighted primarily by player quality.
- Acceptance criteria:
  - all human samples remain in dataset,
  - players above target winrate threshold receive higher weight,
  - weighting metadata is reproducible from dataset snapshot.

#### FR-7: Belief stability gate
- Training schedule must not switch to decision-heavy co-evolution until belief is stable enough.
- Acceptance criteria:
  - minimum training duration is enforced,
  - card/location, aggregate, and calibration metrics all exist,
  - gate policy is configurable and logged.

#### FR-8: Efficient online simulation
- Standard training must remain efficient without large search budgets.
- Acceptance criteria:
  - the default online path uses no heavyweight search,
  - optional Monte Carlo branching is bounded and configurable,
  - throughput regressions are measured and reported.

#### FR-9: Anti-collapse governance
- Checkpoint selection and run stopping must detect pass/minimal-bid collapse.
- Acceptance criteria:
  - model selection uses behavior-aware score, not raw internal avg diff only,
  - collapse metrics are recorded for each eval,
  - guardrails can stop or down-rank degenerate checkpoints.

## 7. Data & Integrations

### Inputs
- Human gameplay logs replayable through current engine.
- Simulated self-play trajectories generated by the Rust server.
- Joint-phase self-play must be phase-balanced:
  - full-game episodes for end-to-end outcome pressure,
  - bidding-targeted episodes via `start_trick=-1`,
  - passing-targeted episodes via `start_trick=0`,
  - so `passing` coverage never depends on the current policy accidentally reaching passing often enough.
- Player long-term quality statistics for weighting.

### Outputs
- Four model checkpoints.
- Structured state snapshots for debugging.
- Belief evaluation reports.
- Phase-model evaluation reports.
- Best-checkpoint artifacts selected by behavior-aware criteria.

### Data contracts
- Human replay records must preserve:
  - player identity,
  - action legality,
  - game outcome,
  - action phase,
  - full reconstructed exact hidden state for supervision.

### Integrations
- Rust engine:
  - exact rules,
  - legal actions,
  - state replay,
  - simulation.
- Python training stack:
  - dataset conversion,
  - model training,
  - evaluation,
  - checkpoint governance.

## 8. Non-Functional Requirements

### Performance
- Symbolic inference updates must be incremental and low-latency.
- Belief decoding must be globally consistent without large combinatorial blow-up.
- Online training must preserve practical simulation throughput.

### Reliability
- Schema/version mismatches must fail fast.
- Contradictions at any interface must be surfaced immediately.

### Observability
- Log symbolic facts, belief predictions, and decoded assignments.
- Support per-ply debug dumps of:
  - symbolic facts,
  - belief diffs,
  - decision-model auxiliary outputs.

### Security / privacy
- No hidden-information leak into training/inference beyond the intended supervision path.
- Teacher-only targets must be isolated from runtime inference inputs.

### Retention / auditing
- Every checkpoint must record:
  - schema version,
  - rule catalog version,
  - player-weighting config,
  - gate thresholds,
  - selection score components.

## 9. UX / Interaction Design
- Debug tooling must make it possible to inspect:
  - exact vs inferred vs believed card locations,
  - why a card was ruled impossible,
  - why a bid/pass/play action was preferred.
- Fixed-suite evaluation output must summarize:
  - pass rate,
  - avg bid,
  - contract made rate,
  - trump/info behavior,
  - selection score.

## 10. Architecture

### Components and responsibilities
- Rust rules engine:
  - exact game truth and simulation.
- Symbolic inference engine:
  - authoritative monotonic reasoning.
- Belief model and decoder:
  - unresolved hidden-card completion.
- Phase models:
  - bidding,
  - passing,
  - playing.
- Training coordinator:
  - human pretraining,
  - joint co-evolution scheduling,
  - gates,
  - checkpoint selection.

### Data flow
1. Engine state -> canonical raw state.
2. Raw state -> symbolic state.
3. Symbolic state -> belief prediction.
4. Symbolic state + belief decode -> combined state.
5. Combined state -> phase model.
6. Phase action -> engine transition.
7. Replay/simulation output -> training targets.

### Storage
- Structured state logs and eval summaries under `ml/runs/...`
- Separate checkpoints per model family and stage.

### Deployment topology
- Training remains Python-orchestrated with Rust simulation backend.
- Runtime inference may use separate loaded models for the three decision phases plus belief.

### Config / secrets
- Versioned config in `ml/config/`.
- No hidden-information flags may alter runtime behavior silently.

### Migration plan
- Introduce canonical state schema first.
- Add symbolic inference engine second.
- Add belief model and decoder third.
- Split decision models last, then migrate training lifecycle.

## 11. Testing Strategy

### Unit
- Rule-level inference tests.
- Monotonicity and contradiction tests.
- Global assignment decoder tests.
- Per-model loss/target tests.

### Integration
- Replay human games into canonical state and belief supervision.
- End-to-end runtime path from engine -> symbolic -> belief -> phase model.

### E2E
- Full training smoke test for:
  - human pretraining,
  - belief pretraining,
  - joint phase gate transition.

### Load / performance
- Measure symbolic update cost per ply.
- Measure belief decoding cost per ply.
- Measure full-game online throughput.

### Security / correctness
- Hidden-information leak tests.
- Schema-compatibility tests.

## 12. Rollout Plan
- Phase 1:
  - canonical state schema,
  - symbolic inference scaffold,
  - debug tooling.
- Phase 2:
  - belief pretraining pipeline and consistency decoder.
- Phase 3:
  - separate bidding/passing/playing models with human pretraining.
- Phase 4:
  - simulation-only co-evolution with belief-heavy start.
- Phase 5:
  - new checkpoint selection and anti-collapse governance.

## 13. Risks & Tradeoffs

### Risk: engineering complexity
- Impact:
  - significantly larger system than current single-pipeline PPO loop.
- Mitigation:
  - land in layered phases with strict interfaces and test gates.

### Risk: symbolic rule errors become authoritative
- Impact:
  - wrong rule means systematically wrong state for all downstream models.
- Mitigation:
  - rule catalog versioning, exhaustive tests, monotonicity checks, replay validation.

### Risk: belief decoder becomes a bottleneck
- Impact:
  - simulation throughput drops.
- Mitigation:
  - constrained matching only on unresolved cards, incremental updates, performance budget tests.

### Risk: model disagreement across phases
- Impact:
  - bidding, passing, and playing learn incompatible priors.
- Mitigation:
  - shared combined-state contract, shared evals, cross-phase diagnostics.

### Key tradeoffs
- Chosen approach favors:
  - interpretability,
  - correctness,
  - modularity,
  - controllable credit assignment.
- It sacrifices:
  - simplicity of a single end-to-end network,
  - short-term implementation speed.

## 14. Superseded / Related Specs
- This spec extends and partially supersedes:
  - [SPEC_NEUROSYMBOLIC_ARCHITECTURE.md](/c:/Users/Johannes/Documents/marjapussi-rs/docs/ml/SPEC_NEUROSYMBOLIC_ARCHITECTURE.md)
  - [SPEC_TRAINING_LIFECYCLE.md](/c:/Users/Johannes/Documents/marjapussi-rs/docs/ml/SPEC_TRAINING_LIFECYCLE.md)
  - [SPEC_SELFPLAY_RECOVERY_PLAN.md](/c:/Users/Johannes/Documents/marjapussi-rs/docs/ml/SPEC_SELFPLAY_RECOVERY_PLAN.md)
