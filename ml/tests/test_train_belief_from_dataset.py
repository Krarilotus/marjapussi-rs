import json
from pathlib import Path

import torch

from ml.neurosymbolic_dataset import build_belief_features, build_belief_targets, load_canonical_state
from ml.pretrain_cache import load_or_build_belief_cache
from ml.tests.test_neurosymbolic_state import sample_payload
from ml.train_belief_from_dataset import collate_belief, train


def test_collate_belief_builds_batch_tensors():
    samples = []
    for _ in range(2):
        state = load_canonical_state({"canonical_state": sample_payload()})
        feats = build_belief_features(state)
        targets = build_belief_targets(state)
        samples.append(
            {
                "card_features": feats.card_features,
                "player_features": feats.player_features[1:],
                "global_features": feats.global_features,
                "card_targets": torch.tensor(targets.card_owner_targets, dtype=torch.long),
                "hidden_mask": torch.tensor(targets.hidden_card_mask, dtype=torch.bool),
                "candidate_mask": torch.tensor(targets.owner_candidate_mask, dtype=torch.bool),
                "void_targets": torch.tensor(targets.player_void_targets[1:], dtype=torch.float32),
                "half_targets": torch.tensor(targets.player_has_half_targets[1:], dtype=torch.float32),
                "pair_targets": torch.tensor(targets.player_has_pair_targets[1:], dtype=torch.float32),
            }
        )
    batch = collate_belief(samples)
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
        epochs=2,
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
        epochs=2,
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


def test_train_belief_resume_skips_when_epoch_budget_reached(tmp_path: Path):
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
    before = torch.load(latest, map_location="cpu")
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
    after = torch.load(latest, map_location="cpu")
    assert after["metadata"]["epochs_seen"] == before["metadata"]["epochs_seen"] == 1


def test_belief_cache_builds_and_reuses(tmp_path: Path):
    data_path = tmp_path / "belief.ndjson"
    with data_path.open("w", encoding="utf-8") as handle:
        for _ in range(2):
            handle.write(json.dumps({"canonical_state": sample_payload()}) + "\n")

    cache_path, samples = load_or_build_belief_cache(data_path)
    assert cache_path.exists()
    assert len(samples) == 2

    cache_path_again, samples_again = load_or_build_belief_cache(data_path)
    assert cache_path_again == cache_path
    assert len(samples_again) == 2
