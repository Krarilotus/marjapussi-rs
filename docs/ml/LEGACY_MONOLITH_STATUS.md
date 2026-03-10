# Legacy Monolith Status

## Scope

This note documents the status of the legacy single-model PPO trainer:

- `ml/train_online.py`
- `ml/train/reward.py`
- `ml/config/train_online.toml`
- the older `just` recipes that call `ml/train_online.py`

## Current policy

The legacy monolith is now a compatibility path.

It is kept because:

1. historical experiments depend on it,
2. old checkpoints remain loadable,
3. some comparative baselines still use it.

It is not the primary development path anymore.

## New primary path

The actively developed stack is the four-model neurosymbolic pipeline:

1. canonical symbolic state
2. authoritative inference layer
3. belief model
4. separate bidding/passing/playing models

See:

- `SPEC_FOUR_MODEL_NEUROSYMBOLIC_SYSTEM.md`
- `BACKLOG_FOUR_MODEL_NEUROSYMBOLIC_SYSTEM.md`
- `ml/README.md`

## Cleanup policy

- No new core features should be added to the monolith unless required for compatibility.
- New training/governance work should land in the four-model stack.
- Documentation should clearly separate:
  - legacy monolith workflows
  - new four-model workflows

## Remaining legacy debt

1. `train_online.py` still owns the old online self-play path.
2. Legacy `just` recipes remain numerous and should be considered historical/compatibility recipes.
3. Some docs in the repo still refer to the old path as if it were the default; these should be treated as maintenance debt.
