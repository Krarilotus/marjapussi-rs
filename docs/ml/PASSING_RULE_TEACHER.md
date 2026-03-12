# Passing Rule Teacher

## Purpose

The passing path now has explicit structural supervision instead of relying only on
final game outcome and weak auxiliary correlations.

## Current targets

`PassingNet` is supervised with:

1. `target_retained_suits`
2. `prefer_blank_suits`
3. `preserve_pairs`
4. `preserve_aces`
5. `preserve_trump`
6. `standing_cards`
7. `pair_points_ceiling`
8. `point_diff`

These are state-level targets.

## Teacher policy

In passing phases, legal pass-card sets are ranked by a heuristic teacher policy.

### PassingForth priorities

1. blank non-trump suits
2. reduce retained suits toward two suits
3. preserve pairs and halves in the remaining hand
4. avoid throwing trump
5. prefer passing aces in suits that become blanked

### PassingBack priorities

1. preserve pairs
2. preserve halves
3. keep at least one ace when possible
4. avoid throwing trump
5. keep suit structure stable, with less pressure to blank aggressively

## Important limitation

This is still a structural teacher, not yet a full counterfactual pass-value model.
It closes the biggest gap in the spec by making suit splitting and information-aware
passing explicit training targets, but it does not yet evaluate every pass set by
downstream rollout EV.
