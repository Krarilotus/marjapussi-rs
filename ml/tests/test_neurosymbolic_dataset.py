from ml.neurosymbolic_dataset import (
    NUM_OWNER_CLASSES,
    OWNER_PARTNER_HAND,
    OWNER_SELF_HAND,
    build_belief_targets,
    build_belief_features,
)
from ml.neurosymbolic_state import CanonicalState
from ml.tests.test_neurosymbolic_state import sample_payload


def test_build_belief_targets_maps_known_and_unknown_cards():
    state = CanonicalState.from_dict(sample_payload())
    targets = build_belief_targets(state)
    assert len(targets.card_owner_targets) == 36
    assert targets.card_owner_targets[0] == OWNER_SELF_HAND
    assert targets.card_owner_targets[1] == OWNER_PARTNER_HAND
    assert targets.owner_candidate_mask[1] == (False, True, True, False, False, False, False, False, False)
    assert len(targets.owner_candidate_mask[0]) == NUM_OWNER_CLASSES


def test_build_belief_targets_exports_player_level_constraints():
    state = CanonicalState.from_dict(sample_payload())
    targets = build_belief_targets(state)
    assert len(targets.player_void_targets) == 4
    assert targets.player_has_half_targets[2][0] is True


def test_build_belief_features_has_expected_shapes():
    state = CanonicalState.from_dict(sample_payload())
    feats = build_belief_features(state)
    assert tuple(feats.card_features.shape) == (36, 23)
    assert tuple(feats.player_features.shape) == (4, 18)
    assert tuple(feats.global_features.shape) == (12,)
