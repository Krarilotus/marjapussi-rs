import json
from pathlib import Path

import torch

from ml.tests.test_neurosymbolic_state import sample_payload
from ml.train_belief_from_dataset import collate_belief, train


def test_collate_belief_builds_batch_tensors():
    batch = collate_belief(
        [
            {"canonical_state": sample_payload()},
            {"canonical_state": sample_payload()},
        ]
    )
    assert batch is not None
    assert tuple(batch["card_features"].shape) == (2, 36, 23)
    assert tuple(batch["player_features"].shape) == (2, 3, 18)
    assert tuple(batch["global_features"].shape) == (2, 12)
    assert tuple(batch["card_targets"].shape) == (2, 36)


def test_train_belief_from_dataset_resume_smoke(tmp_path: Path):
    data_path = tmp_path / "belief.ndjson"
    with data_path.open("w", encoding="utf-8") as handle:
        for _ in range(2):
            handle.write(json.dumps({"canonical_state": sample_payload()}) + "\n")

    ckpt_dir = tmp_path / "ckpts"
    train(
        data_path=str(data_path),
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
    latest = ckpt_dir / "belief_latest.pt"
    assert latest.exists()

    train(
        data_path=str(data_path),
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
