# Marjapussi ML Documentation Index

This folder defines the implementation contracts for the Marjapussi ML system.

Project-level intent is defined in:

1. `docs/ML_PROJECT_OVERVIEW.md`

Active technical specs in this folder:

1. `SPEC_FOUR_MODEL_NEUROSYMBOLIC_SYSTEM.md`
2. `BACKLOG_FOUR_MODEL_NEUROSYMBOLIC_SYSTEM.md`
3. `OPEN_ISSUES_FOUR_MODEL_NEUROSYMBOLIC_SYSTEM.md`
4. `SPEC_NEUROSYMBOLIC_ARCHITECTURE.md`
5. `SPEC_TRAINING_LIFECYCLE.md`
6. `SPEC_REWARD_AND_SCORING.md`
7. `SPEC_OBSERVATION_PRIVACY.md`
8. `SPEC_COUNTERFACTUAL_ADVANTAGE.md`
9. `SPEC_PASSING_CURRICULUM.md`
10. `SPEC_PARALLEL_MODEL_ARCHITECTURE_V2.md`
11. `TEST_AND_VALIDATION_PLAN.md`
12. `LEGACY_MONOLITH_STATUS.md`

Deprecated or superseded specs:

1. `SPEC_SELFPLAY_RECOVERY_PLAN.md`
   - superseded by `SPEC_FOUR_MODEL_NEUROSYMBOLIC_SYSTEM.md`
   - still useful as historical context for pass-collapse debugging

Legacy compatibility notes:

1. `LEGACY_MONOLITH_STATUS.md`
   - documents the status of `ml/train_online.py` and related PPO-era recipes as compatibility-only

Execution plans:

1. `PLAN_V2_IMPLEMENTATION_TASKLIST.md`

Operational tooling:

1. Fixed-deal evaluator: `ml/eval_fixed_deals.py`
2. Editable suites: `ml/eval/fixed_deals_100.json`, `ml/eval/fixed_deals_custom_template.json`
3. Canonical state exporter: `src/bin/ml_convert_legacy.rs`
4. Belief pretrainer: `ml/train_belief_from_dataset.py`
5. Belief decoder: `ml/belief_decoder.py`
6. Decision-state compiler and split model shells:
   - `ml/decision_state.py`
   - `ml/decision_model.py`
7. Decision-model human pretrainer:
   - `ml/train_decision_from_dataset.py`
   - `just pretrain-decision-human task=bidding`
   - `just pretrain-decision-human task=passing`
   - `just pretrain-decision-human task=playing`
8. Four-model human-pretraining orchestrator:
   - `ml/train_four_model_human_pretrain.py`
   - `just pretrain-four-model-human`
9. Four-model manifest/runtime layer:
   - `ml/four_model_manifest.py`
   - `ml/check_four_model_manifest.py`
   - `just check-four-model-manifest`
10. Lightweight four-model runtime benchmark:
   - `ml/benchmark_four_model_runtime.py`
   - `just benchmark-four-model-runtime`
11. Runtime bridge from decoded belief state into split decision features:
   - `ml/four_model_runtime.py`
12. Asymmetrical joint-phase scheduler primitive:
   - `ml/four_model_schedule.py`
13. Behavior-aware scoring primitive for anti-collapse checkpoint selection:
   - `ml/behavior_score.py`
14. Simulated-only joint training coordinator:
   - `ml/train_four_model_joint.py`
   - `just train-four-model-joint`
   - now performs behavior-aware fixed-suite checkpoint governance in the joint loop
15. Simulated self-play generator for the four-model stack:
   - `ml/generate_four_model_selfplay.py`
   - `just generate-four-model-selfplay`
   - `just train-four-model-endtoend`
   - `ml/train_four_model_endtoend.py`
   - end-to-end loop can also perform fixed-suite governance every cycle
16. Strict phase-by-phase autorun:
   - `ml/train_four_model_autorun.py`
   - promotes validated phase checkpoints
   - writes `phase_reports/*.json`
   - retries failed phases and rejects self-play cycles with missing task coverage
17. Live structured-state transport and manifest-aware runtime:
   - `src/ml/proto.rs`
   - `ml/env.py`
   - `ml/four_model_runtime.py`
   - `ml/eval_fixed_deals.py`
   - `ml/ui_server.py`

Order of implementation should follow:

1. Canonical state + symbolic inference.
2. Belief pretraining and conflict-free decode.
3. Split decision-model human pretraining.
4. Simulated-only joint co-evolution and checkpoint governance.
5. Live runtime serving, eval, and UI integration.

Verified smoke path:

1. `pretrain-four-model-human`
2. `generate-four-model-selfplay`
3. `train-four-model-joint`
4. `eval-fixed-four-model`

The joint smoke path has been run end-to-end with:

- human manifest output
- generated self-play dataset
- simulated-only joint update
- fixed-suite governance artifacts:
  - `governance/last_fixed_suite_eval.json`
  - `governance/best_fixed_suite_manifest.json`
