import torch

from ml.belief_model import BeliefNet
from ml.neurosymbolic_dataset import build_belief_features, build_belief_targets
from ml.neurosymbolic_state import CanonicalState
from ml.tests.test_neurosymbolic_state import sample_payload


def test_belief_model_forward_shapes():
    state = CanonicalState.from_dict(sample_payload())
    feats = build_belief_features(state)
    targets = build_belief_targets(state)

    model = BeliefNet()
    outputs = model(
        card_features=feats.card_features.unsqueeze(0),
        player_features=feats.player_features[1:].unsqueeze(0),
        global_features=feats.global_features.unsqueeze(0),
    )

    assert tuple(outputs["card_logits"].shape) == (1, 36, len(targets.owner_candidate_mask[0]))
    assert tuple(outputs["player_void_logits"].shape) == (1, 3, 4)
    assert tuple(outputs["player_half_logits"].shape) == (1, 3, 4)
    assert tuple(outputs["player_pair_logits"].shape) == (1, 3, 4)


def test_belief_model_can_apply_candidate_mask():
    state = CanonicalState.from_dict(sample_payload())
    feats = build_belief_features(state)
    targets = build_belief_targets(state)
    model = BeliefNet()
    outputs = model(
        card_features=feats.card_features.unsqueeze(0),
        player_features=feats.player_features[1:].unsqueeze(0),
        global_features=feats.global_features.unsqueeze(0),
    )
    candidate_mask = torch.tensor(targets.owner_candidate_mask, dtype=torch.bool).unsqueeze(0)
    masked = outputs["card_logits"].masked_fill(~candidate_mask, -1e4)
    assert torch.isfinite(masked.max())
