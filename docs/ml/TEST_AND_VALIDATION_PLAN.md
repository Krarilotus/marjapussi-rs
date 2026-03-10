# Test and Validation Plan

## 1. Purpose

Define the minimum test coverage needed before large-scale training and public-facing AI play.

## 2. Test Categories

### 2.1 Rules and scoring correctness

1. Contract success/failure scoring.
2. Schwarz special-case scoring behavior by configured mode.
3. No-one-played game handling.
4. Last-trick bonus handling.
5. First lead constraints:
   first card of first trick must be Ace, else Green, else any.
6. First-trick follow rule:
   Ace of led suit must be played if held.
7. Follow/trump/overtake obligations:
   follow suit if possible, else trump if required, and overtake when required by rules.
8. Partner-question sequencing rules:
   ordering and exclusivity constraints on pair/half questions and self-declarations.

### 2.2 Observation privacy

1. ML serializer does not expose forbidden hidden fields.
2. Debug serializer can expose full fields without ML contamination.
3. Non-POV turn legal-action encoding does not leak hidden exact cards.
4. Python loader rejects unsupported `schema_version` at runtime.

### 2.3 Dataset conversion integrity

1. Legacy game replay reproduces legal action sequence.
2. Converted decision records align with observation schema.
3. Action index mapping remains valid per decision.

### 2.4 Training objective correctness

1. PPO ratio path only for model-sampled actions.
2. Imitation path for heuristic-forced actions.
3. No NaN/inf and mask-consistent logits.

### 2.5 Counterfactual estimator reliability

1. Deterministic reproducibility with fixed seeds.
2. Advantage normalization sanity.
3. Budget/performance instrumentation.

### 2.6 Passing representation and curriculum

1. Pass-set encoding uniqueness checks.
2. Curriculum gating behavior by stage.
3. Candidate generation includes required baseline alternatives.

## 3. Test Levels

1. Unit tests for pure functions and scoring logic.
2. Integration tests across Rust server and Python loaders.
3. End-to-end smoke tests for short training/eval loops.
4. Regression tests for known historical bugs.

Current baseline tests include:

1. Rust inference/set-theory tests in `src/ml/inference.rs`.
2. Rust legacy-converter parser tests in `src/bin/ml_convert_legacy.rs`.
3. Python env/reward sanity tests in `ml/tests/test_env_reward_basics.py`.
4. Rust canonical-state tests in `src/ml/state.rs`.
5. Rust monotonic inference-state tests in `src/ml/inference/state.rs`.
6. Python canonical-state and belief-target tests in:
   - `ml/tests/test_neurosymbolic_state.py`
   - `ml/tests/test_neurosymbolic_dataset.py`
7. Python belief-model and collation smoke tests in:
   - `ml/tests/test_belief_model.py`
   - `ml/tests/test_train_belief_from_dataset.py`
8. Python decision-state / split-model interface tests in:
   - `ml/tests/test_decision_state.py`
   - `ml/tests/test_decision_model.py`
   - `ml/tests/test_train_decision_from_dataset.py`
9. Stage-1 four-model pretraining orchestrator smoke test:
   - `ml/tests/test_train_four_model_human_pretrain.py`
10. Runtime bridge tests:
   - `ml/tests/test_four_model_runtime.py`
11. Four-model manifest validation tests:
   - `ml/tests/test_four_model_manifest.py`
12. Lightweight runtime benchmark utility tests:
   - `ml/tests/test_benchmark_four_model_runtime.py`
13. Joint-schedule stability-gate tests:
   - `ml/tests/test_four_model_schedule.py`
14. Behavior-aware scoring tests:
   - `ml/tests/test_behavior_score.py`
15. Checkpoint metadata / resume tests:
   - `ml/tests/test_checkpoint_utils.py`
16. Simulated-only joint-training coordinator smoke tests:
   - `ml/tests/test_train_four_model_joint.py`
17. Manifest-aware fixed-eval tests:
   - `ml/tests/test_eval_fixed_deals.py`
18. Live env structured-state merge tests:
   - `ml/tests/test_env_structured_state.py`
19. Four-model self-play dataset generator smoke tests:
   - `ml/tests/test_generate_four_model_selfplay.py`
20. End-to-end four-model coordinator smoke tests:
   - `ml/tests/test_train_four_model_endtoend.py`
21. Fixed-suite governance selection tests:
   - `ml/tests/test_four_model_governance.py`
22. Joint-trainer governance metadata propagation:
   - `ml/tests/test_train_four_model_joint.py`
23. Autorun strict phase-validation / retry / self-play-coverage tests:
   - `ml/tests/test_train_four_model_autorun.py`

## 4. CI Gate Requirements

Must pass before merge:

1. Rust unit/integration tests.
2. Python unit tests for ML data and loss path.
3. Schema compatibility checks.
4. Privacy denylist checks.

## 5. Determinism and Reproducibility

Test harness must support:

1. Fixed random seeds.
2. Stable replay from logged game/action sequences.
3. Checkpoint metadata assertions.

## 6. Documentation Conformance

Any PR that changes behavior in reward, observation schema, or action encoding must:

1. Update relevant spec documents.
2. Add or update matching tests.
3. Record migration notes for old checkpoints/datasets.
