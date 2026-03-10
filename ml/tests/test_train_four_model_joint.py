import json
from pathlib import Path

import ml.train_four_model_joint as mod
from ml.four_model_manifest import BeliefStageManifest, DecisionStageManifest, FourModelOutputs, write_four_model_manifest


def test_run_joint_training_writes_joint_manifest(tmp_path: Path, monkeypatch):
    base_root = tmp_path / "base"
    for task in ("bidding", "passing", "playing"):
        task_dir = base_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / f"{task}_latest.pt").write_text("base", encoding="utf-8")
    belief_dir = base_root / "belief"
    belief_dir.mkdir(parents=True, exist_ok=True)
    (belief_dir / "belief_latest.pt").write_text("base", encoding="utf-8")

    base_manifest = write_four_model_manifest(
        base_root / "human_pretrain_manifest.json",
        data_path="ml/data/human.ndjson",
        device="cpu",
        workers=0,
        decision_stages=(
            DecisionStageManifest(task="bidding", epochs=1, batch=8, target_acc=0.4),
            DecisionStageManifest(task="passing", epochs=1, batch=8, target_acc=0.4),
            DecisionStageManifest(task="playing", epochs=1, batch=8, target_acc=0.4),
        ),
        belief_stage=BeliefStageManifest(epochs=1, batch=8, target_hidden_acc=0.7),
        outputs=FourModelOutputs(
            bidding=base_root / "bidding" / "bidding_latest.pt",
            passing=base_root / "passing" / "passing_latest.pt",
            playing=base_root / "playing" / "playing_latest.pt",
            belief=base_root / "belief" / "belief_latest.pt",
        ),
        metadata={"training_stage": "human_pretrain"},
    )

    def fake_checkpoint_metadata(path, device="cpu"):
        return {
            "global_step": 1000,
            "epochs_seen": 2,
            "belief_metrics": {
                "card_owner_acc": 0.8,
                "constraint_consistency": 1.0,
                "void_suit_acc": 0.9,
                "half_pair_acc": 0.85,
                "calibration_score": 0.7,
            },
        }

    def fake_train_belief(**kwargs):
        ckpt_dir = Path(kwargs["checkpoints_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "belief_latest.pt").write_text("joint", encoding="utf-8")
        return {
            "card_owner_acc": 0.82,
            "constraint_consistency": 1.0,
            "void_suit_acc": 0.91,
            "half_pair_acc": 0.86,
            "calibration_score": 0.72,
            "global_step": 1100.0,
            "epochs_seen": 3.0,
        }

    def fake_train_decision(**kwargs):
        ckpt_dir = Path(kwargs["checkpoints_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / f"{kwargs['task']}_latest.pt").write_text("joint", encoding="utf-8")
        return {
            "accuracy": 0.5,
            "epochs_seen": 2.0,
            "global_step": 120.0,
            "policy_loss": 0.2,
            "value_loss": 0.1,
            "aux_loss": 0.05,
        }

    monkeypatch.setattr(mod, "checkpoint_metadata", fake_checkpoint_metadata)
    monkeypatch.setattr(mod, "train_belief", fake_train_belief)
    monkeypatch.setattr(mod, "train_decision", fake_train_decision)

    manifest_path = mod.run_joint_training(
        base_manifest_path=base_manifest,
        sim_data_path="ml/data/selfplay.ndjson",
        checkpoints_dir=tmp_path / "joint",
        cycles=1,
        belief_epochs_per_update=1,
        decision_epochs_per_update=1,
        belief_max_steps=1,
        decision_max_steps=1,
        device="cpu",
        workers=0,
        no_amp=True,
    )

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert payload["metadata"]["training_stage"] == "joint_simulated"
    assert payload["metadata"]["joint_cycles_completed"] == 1
    assert payload["metadata"]["last_schedule_phase"] == "decision_heavy"
    assert (tmp_path / "joint" / "joint_summary.json").exists()


def test_run_joint_training_attaches_governance_metadata(tmp_path: Path, monkeypatch):
    base_root = tmp_path / "base"
    for task in ("bidding", "passing", "playing"):
        task_dir = base_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / f"{task}_latest.pt").write_text("base", encoding="utf-8")
    belief_dir = base_root / "belief"
    belief_dir.mkdir(parents=True, exist_ok=True)
    (belief_dir / "belief_latest.pt").write_text("base", encoding="utf-8")

    base_manifest = write_four_model_manifest(
        base_root / "human_pretrain_manifest.json",
        data_path="ml/data/human.ndjson",
        device="cpu",
        workers=0,
        decision_stages=(
            DecisionStageManifest(task="bidding", epochs=1, batch=8, target_acc=0.4),
            DecisionStageManifest(task="passing", epochs=1, batch=8, target_acc=0.4),
            DecisionStageManifest(task="playing", epochs=1, batch=8, target_acc=0.4),
        ),
        belief_stage=BeliefStageManifest(epochs=1, batch=8, target_hidden_acc=0.7),
        outputs=FourModelOutputs(
            bidding=base_root / "bidding" / "bidding_latest.pt",
            passing=base_root / "passing" / "passing_latest.pt",
            playing=base_root / "playing" / "playing_latest.pt",
            belief=base_root / "belief" / "belief_latest.pt",
        ),
        metadata={"training_stage": "human_pretrain"},
    )

    monkeypatch.setattr(
        mod,
        "checkpoint_metadata",
        lambda path, device="cpu": {
            "global_step": 1000,
            "epochs_seen": 2,
            "belief_metrics": {
                "card_owner_acc": 0.8,
                "constraint_consistency": 1.0,
                "void_suit_acc": 0.9,
                "half_pair_acc": 0.85,
                "calibration_score": 0.7,
            },
        },
    )
    monkeypatch.setattr(
        mod,
        "train_belief",
        lambda **kwargs: {
            "card_owner_acc": 0.82,
            "constraint_consistency": 1.0,
            "void_suit_acc": 0.91,
            "half_pair_acc": 0.86,
            "calibration_score": 0.72,
            "global_step": 1100.0,
            "epochs_seen": 3.0,
        },
    )
    monkeypatch.setattr(mod, "train_decision", lambda **kwargs: None)
    monkeypatch.setattr(
        mod,
        "evaluate_manifest_behavior",
        lambda *args, **kwargs: type(
            "EvalResult",
            (),
            {
                "manifest_path": str(args[0]),
                "suite_path": kwargs["suite_path"],
                "summary": {"games": 4, "pass_game_rate": 0.0},
                "score": {"total": 7.5},
            },
        )(),
    )
    monkeypatch.setattr(
        mod,
        "maybe_promote_best_manifest",
        lambda **kwargs: (True, Path(kwargs["governance_dir"]) / "best_fixed_suite_manifest.json", {"total": 7.5}),
    )

    manifest_path = mod.run_joint_training(
        base_manifest_path=base_manifest,
        sim_data_path="ml/data/selfplay.ndjson",
        checkpoints_dir=tmp_path / "joint_gov",
        cycles=1,
        belief_epochs_per_update=1,
        decision_epochs_per_update=1,
        belief_max_steps=1,
        decision_max_steps=1,
        device="cpu",
        workers=0,
        no_amp=True,
        fixed_suite_path="ml/eval/fixed_deals_100.json",
        fixed_suite_max_cases=4,
    )
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert payload["metadata"]["best_fixed_suite_promoted"] is True
    assert payload["metadata"]["last_fixed_suite_eval"]["total"] == 7.5
