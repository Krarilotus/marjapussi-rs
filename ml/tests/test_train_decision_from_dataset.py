import json
from pathlib import Path

import torch

from ml.decision_state import build_decision_features_from_record, build_decision_targets_from_record
from ml.pretrain_cache import load_or_build_decision_cache
from ml.tests.test_decision_state import sample_passing_record, sample_record
from ml.train_decision_from_dataset import collate_decision, train


def make_record(phase: str, action_token: int) -> dict:
    record = sample_record()
    record["canonical_state"]["global"]["phase"] = phase
    record["obs"]["phase"] = phase
    record["obs"]["legal_actions"][0]["action_token"] = action_token
    return record


def test_collate_decision_builds_batch_tensors():
    records = [make_record("Bidding", 41), make_record("Bidding", 41)]
    samples = []
    for record in records:
        features = build_decision_features_from_record(record, use_teacher_belief=True)
        targets = build_decision_targets_from_record(record)
        samples.append(
            {
                "card_features": features.card_features,
                "player_features": features.player_features,
                "global_features": features.global_features,
                "action_features": features.action_features,
                "action_mask": features.action_mask,
                "policy_target": int(targets.policy_idx),
                "value_target": float(targets.value_target),
                "aux_targets": targets.aux_targets,
                "sample_weight": float(targets.sample_weight),
                "teacher_policy": targets.teacher_policy,
            }
        )
    batch = collate_decision(samples)
    assert batch is not None
    assert tuple(batch["card_features"].shape) == (2, 36, 32)
    assert tuple(batch["player_features"].shape) == (2, 4, 18)
    assert tuple(batch["global_features"].shape) == (2, 30)
    assert tuple(batch["action_features"].shape) == (2, 2, 87)
    assert tuple(batch["policy_targets"].shape) == (2,)
    assert tuple(batch["aux_targets"].shape) == (2, 10)
    assert tuple(batch["teacher_policy"].shape) == (2, 2)


def test_collate_decision_pads_variable_action_counts():
    rec_a = make_record("Bidding", 41)
    rec_b = make_record("Bidding", 41)
    rec_b["obs"]["legal_actions"] = rec_b["obs"]["legal_actions"][:1]
    samples = []
    for record in [rec_a, rec_b]:
        features = build_decision_features_from_record(record, use_teacher_belief=True)
        targets = build_decision_targets_from_record(record)
        samples.append(
            {
                "card_features": features.card_features,
                "player_features": features.player_features,
                "global_features": features.global_features,
                "action_features": features.action_features,
                "action_mask": features.action_mask,
                "policy_target": int(targets.policy_idx),
                "value_target": float(targets.value_target),
                "aux_targets": targets.aux_targets,
                "sample_weight": float(targets.sample_weight),
                "teacher_policy": targets.teacher_policy,
            }
        )
    batch = collate_decision(samples)
    assert batch is not None
    assert tuple(batch["action_features"].shape) == (2, 2, 87)
    assert tuple(batch["action_mask"].shape) == (2, 2)
    assert tuple(batch["teacher_policy"].shape) == (2, 2)
    assert bool(batch["action_mask"][1, 1].item()) is True


def test_collate_decision_pads_teacher_policy_to_action_count():
    rec_a = make_record("Playing", 300)
    rec_b = make_record("Playing", 300)
    rec_a["obs"]["legal_actions"] = rec_a["obs"]["legal_actions"][:14]
    rec_b["obs"]["legal_actions"] = rec_b["obs"]["legal_actions"][:6]
    samples = []
    for record in [rec_a, rec_b]:
        features = build_decision_features_from_record(record, use_teacher_belief=True)
        targets = build_decision_targets_from_record(record)
        teacher_policy = targets.teacher_policy
        if teacher_policy is not None and teacher_policy.numel() > 0:
            teacher_policy = teacher_policy[:-1]
        samples.append(
            {
                "card_features": features.card_features,
                "player_features": features.player_features,
                "global_features": features.global_features,
                "action_features": features.action_features,
                "action_mask": features.action_mask,
                "policy_target": int(targets.policy_idx),
                "value_target": float(targets.value_target),
                "aux_targets": targets.aux_targets,
                "sample_weight": float(targets.sample_weight),
                "teacher_policy": teacher_policy,
            }
        )
    batch = collate_decision(samples)
    assert batch is not None
    assert batch["teacher_policy"].shape[0] == 2
    assert batch["teacher_policy"].shape[1] == batch["action_features"].shape[1]


def test_collate_decision_builds_passing_teacher_batch():
    samples = []
    for record in [sample_passing_record(), sample_passing_record()]:
        features = build_decision_features_from_record(record, use_teacher_belief=True)
        targets = build_decision_targets_from_record(record)
        samples.append(
            {
                "card_features": features.card_features,
                "player_features": features.player_features,
                "global_features": features.global_features,
                "action_features": features.action_features,
                "action_mask": features.action_mask,
                "policy_target": int(targets.policy_idx),
                "value_target": float(targets.value_target),
                "aux_targets": targets.aux_targets,
                "sample_weight": float(targets.sample_weight),
                "teacher_policy": targets.teacher_policy,
            }
        )
    batch = collate_decision(samples)
    assert batch is not None
    assert tuple(batch["aux_targets"].shape) == (2, 8)
    assert tuple(batch["teacher_policy"].shape) == (2, 2)
    assert float(batch["teacher_policy"][0].sum().item()) > 0.99


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
    latest = ckpt_dir / "bidding_latest.pt"
    assert latest.exists()

    train(
        data_path=str(data_path),
        task="bidding",
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


def test_train_decision_resume_skips_when_epoch_budget_reached(tmp_path: Path):
    data_path = tmp_path / "decision.ndjson"
    with data_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(make_record("Bidding", 41)) + "\n")
        handle.write(json.dumps(make_record("Bidding", 41)) + "\n")

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
    before = torch.load(latest, map_location="cpu")
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
    after = torch.load(latest, map_location="cpu")
    assert after["metadata"]["epochs_seen"] == before["metadata"]["epochs_seen"] == 1


def test_decision_cache_builds_and_reuses(tmp_path: Path):
    data_path = tmp_path / "decision.ndjson"
    with data_path.open("w", encoding="utf-8") as handle:
        for _ in range(2):
            handle.write(json.dumps(make_record("Bidding", 41)) + "\n")

    cache_path, samples = load_or_build_decision_cache(data_path, "bidding")
    assert cache_path.exists()
    assert len(samples) == 2

    cache_path_again, samples_again = load_or_build_decision_cache(data_path, "bidding")
    assert cache_path_again == cache_path
    assert len(samples_again) == 2
