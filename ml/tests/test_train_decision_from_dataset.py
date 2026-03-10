import json
from pathlib import Path

import torch

from ml.tests.test_decision_state import sample_record
from ml.train_decision_from_dataset import collate_decision, train


def make_record(phase: str, action_token: int) -> dict:
    record = sample_record()
    record["canonical_state"]["global"]["phase"] = phase
    record["obs"]["phase"] = phase
    record["obs"]["legal_actions"][0]["action_token"] = action_token
    return record


def test_collate_decision_builds_batch_tensors():
    batch = collate_decision([make_record("Bidding", 41), make_record("Bidding", 41)])
    assert batch is not None
    assert tuple(batch["card_features"].shape) == (2, 36, 32)
    assert tuple(batch["player_features"].shape) == (2, 4, 18)
    assert tuple(batch["global_features"].shape) == (2, 15)
    assert tuple(batch["action_features"].shape) == (2, 2, 87)
    assert tuple(batch["policy_targets"].shape) == (2,)
    assert tuple(batch["aux_targets"].shape) == (2, 3)


def test_collate_decision_pads_variable_action_counts():
    rec_a = make_record("Bidding", 41)
    rec_b = make_record("Bidding", 41)
    rec_b["obs"]["legal_actions"] = rec_b["obs"]["legal_actions"][:1]
    batch = collate_decision([rec_a, rec_b])
    assert batch is not None
    assert tuple(batch["action_features"].shape) == (2, 2, 87)
    assert tuple(batch["action_mask"].shape) == (2, 2)
    assert bool(batch["action_mask"][1, 1].item()) is True


def test_train_decision_from_dataset_smoke(tmp_path: Path):
    data_path = tmp_path / "decision.ndjson"
    records = [
        make_record("Bidding", 41),
        make_record("Bidding", 41),
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    ckpt_dir = tmp_path / "ckpts"
    train(
        data_path=str(data_path),
        task="bidding",
        epochs=1,
        batch=2,
        lr=1e-3,
        device="cpu",
        workers=0,
        checkpoints_dir=ckpt_dir,
        log_every=1,
        max_steps=1,
        no_amp=True,
    )
    latest = ckpt_dir / "bidding_latest.pt"
    assert latest.exists()

    train(
        data_path=str(data_path),
        task="bidding",
        epochs=1,
        batch=2,
        lr=1e-3,
        device="cpu",
        workers=0,
        checkpoints_dir=ckpt_dir,
        log_every=1,
        max_steps=1,
        no_amp=True,
        checkpoint=latest,
    )
    payload = torch.load(latest, map_location="cpu")
    assert payload["metadata"]["epochs_seen"] >= 2
