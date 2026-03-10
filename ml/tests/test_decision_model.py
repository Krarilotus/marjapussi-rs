from ml.decision_model import BiddingNet, PassingNet, PlayingNet
from ml.decision_state import build_decision_features_from_record
from ml.tests.test_decision_state import sample_record


def test_decision_models_forward_shapes():
    features = build_decision_features_from_record(sample_record(), use_teacher_belief=True)
    models = [BiddingNet(), PassingNet(), PlayingNet()]
    for model in models:
        out = model(
            card_features=features.card_features.unsqueeze(0),
            player_features=features.player_features.unsqueeze(0),
            global_features=features.global_features.unsqueeze(0),
            action_features=features.action_features.unsqueeze(0),
            action_mask=features.action_mask.unsqueeze(0),
        )
        assert tuple(out["policy_logits"].shape) == (1, 2)
        assert tuple(out["value"].shape) == (1,)
        assert tuple(out["aux"].shape) == (1, 3)
