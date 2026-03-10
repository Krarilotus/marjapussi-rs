import json
from pathlib import Path

import torch

from ml.belief_model import BeliefNet
from ml.decision_model import BiddingNet, PassingNet, PlayingNet
from ml.four_model_runtime import (
    build_runtime_decision_features,
    choose_action_pos_with_bundle,
    load_four_model_bundle,
    normalize_runtime_record,
    predict_belief_owner_onehot,
    predict_with_bundle,
)
from ml.neurosymbolic_state import CanonicalState
from ml.tests.test_belief_decoder import decoder_payload
from ml.tests.test_decision_state import sample_record


class FakeBeliefModel(torch.nn.Module):
    def forward(self, card_features, player_features, global_features):
        logits = torch.zeros(card_features.shape[0], 36, 9)
        logits[:, 1, 1] = 10.0
        logits[:, 2, 2] = 10.0
        logits[:, 3, 3] = 10.0
        zeros = torch.zeros(card_features.shape[0], 3, 4)
        return {
            "card_logits": logits,
            "player_void_logits": zeros,
            "player_half_logits": zeros,
            "player_pair_logits": zeros,
        }


def test_predict_belief_owner_onehot_decodes_conflict_free_assignment():
    state = CanonicalState.from_dict(decoder_payload())
    belief_owner = predict_belief_owner_onehot(state, FakeBeliefModel(), device="cpu")
    assert tuple(belief_owner.shape) == (36, 9)
    assert belief_owner[1, 1].item() == 1.0
    assert belief_owner[2, 2].item() == 1.0
    assert belief_owner[3, 3].item() == 1.0


def test_build_runtime_decision_features_uses_predicted_belief():
    record = sample_record()
    record["canonical_state"] = decoder_payload()
    record["obs"]["phase"] = "Trick"
    record["obs"]["legal_actions"] = [
        {"action_token": 1, "bid_value": None, "card_idx": 1, "suit_idx": 0},
        {"action_token": 2, "bid_value": None, "card_idx": 2, "suit_idx": 0},
    ]
    features = build_runtime_decision_features(record, FakeBeliefModel(), device="cpu")
    assert features.task == "playing"
    assert tuple(features.card_features.shape) == (36, 32)
    assert features.card_features[1, -9 + 1].item() == 1.0


def test_normalize_runtime_record_accepts_live_obs_payload():
    payload = sample_record()["obs"]
    payload["canonical_state"] = decoder_payload()
    payload["belief_targets"] = decoder_payload()["belief_targets"]
    normalized = normalize_runtime_record(payload)
    assert "obs" in normalized
    assert "canonical_state" in normalized
    assert normalized["obs"]["phase"] == payload["phase"]


def test_load_four_model_bundle_and_predict(tmp_path: Path):
    belief_path = tmp_path / "belief.pt"
    bidding_path = tmp_path / "bidding.pt"
    passing_path = tmp_path / "passing.pt"
    playing_path = tmp_path / "playing.pt"

    torch.save({"state_dict": BeliefNet().state_dict()}, belief_path)
    torch.save({"state_dict": BiddingNet().state_dict()}, bidding_path)
    torch.save({"state_dict": PassingNet().state_dict()}, passing_path)
    torch.save({"state_dict": PlayingNet().state_dict()}, playing_path)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_path": "ml/data/human_dataset.ndjson",
                "device": "cpu",
                "workers": 1,
                "decision_stages": [
                    {"task": "bidding", "epochs": 1, "batch": 8, "target_acc": 0.5},
                    {"task": "passing", "epochs": 1, "batch": 8, "target_acc": 0.5},
                    {"task": "playing", "epochs": 1, "batch": 8, "target_acc": 0.5},
                ],
                "belief_stage": {
                    "epochs": 1,
                    "batch": 8,
                    "target_hidden_acc": 0.7,
                },
                "outputs": {
                    "bidding": str(bidding_path),
                    "passing": str(passing_path),
                    "playing": str(playing_path),
                    "belief": str(belief_path),
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = load_four_model_bundle(manifest_path, device="cpu")
    record = sample_record()
    record["canonical_state"] = decoder_payload()
    record["obs"]["phase"] = "Trick"
    outputs = predict_with_bundle(bundle, record)
    assert outputs["task"] == "playing"
    assert tuple(outputs["policy_logits"].shape) == (2,)
    pos, conf = choose_action_pos_with_bundle(bundle, record["obs"] | {"canonical_state": record["canonical_state"]})
    assert pos in (0, 1)
    assert 0.0 <= conf <= 1.0
