import json
from pathlib import Path

import pytest
import torch

import ml.train_four_model_autorun as mod


def test_run_autorun_retries_joint_until_metrics_pass(tmp_path: Path, monkeypatch):
    calls = {"joint": 0, "selfplay": 0}

    def fake_decision(**kwargs):
        ckpt_dir = Path(kwargs["checkpoints_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": {},
            "metadata": {
                "task": kwargs["task"],
                "accuracy": 0.7,
                "policy_loss": 0.1,
            },
        }
        torch.save(payload, ckpt_dir / f"{kwargs['task']}_latest.pt")
        torch.save(payload, ckpt_dir / f"{kwargs['task']}_best.pt")
        return {
            "accuracy": 0.7,
            "epochs_seen": 1.0,
            "global_step": 1.0,
            "policy_loss": 0.1,
            "value_loss": 0.1,
            "aux_loss": 0.1,
        }

    def fake_belief(**kwargs):
        ckpt_dir = Path(kwargs["checkpoints_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "card_owner_acc": 0.5,
            "constraint_consistency": 1.0,
            "void_suit_acc": 0.8,
            "half_pair_acc": 0.8,
            "pair_acc": 0.8,
            "calibration_score": 0.5,
        }
        payload = {
            "state_dict": {},
            "metadata": {
                "belief_metrics": metrics,
            },
        }
        torch.save(payload, ckpt_dir / "belief_latest.pt")
        torch.save(payload, ckpt_dir / "belief_best.pt")
        return dict(metrics, global_step=2.0, epochs_seen=1.0)

    monkeypatch.setattr(mod, "train_decision", fake_decision)
    monkeypatch.setattr(mod, "train_belief", fake_belief)

    def fake_selfplay(manifest, output, *, games, full_games=None, bidding_games=0, passing_games=0, seed_start, max_steps=300, max_seed_tries_per_target=32, device=None):
        calls["selfplay"] += 1
        records = [{"canonical_state": {"global": {"phase": "Bidding"}}} for _ in range(8)]
        records.extend({"canonical_state": {"global": {"phase": "PassingForth"}}} for _ in range(4))
        records.extend({"canonical_state": {"global": {"phase": "PassingBack"}}} for _ in range(4))
        records.extend({"canonical_state": {"global": {"phase": "Trick"}}} for _ in range(40))
        Path(output).write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return None

    def fake_joint(**kwargs):
        calls["joint"] += 1
        out = Path(kwargs["checkpoints_dir"]) / "joint_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "last_fixed_suite_eval": {
                    "summary": {
                        "pass_game_rate": 0.9 if calls["joint"] == 1 else 0.1,
                        "contract_made_rate": 0.0 if calls["joint"] == 1 else 0.2,
                        "avg_highest_bid": 120.0 if calls["joint"] == 1 else 140.0,
                        "avg_playing_margin_points": -10.0 if calls["joint"] == 1 else 5.0,
                    }
                }
            }
        }
        out.write_text(json.dumps(payload), encoding="utf-8")
        return out

    monkeypatch.setattr(mod, "generate_selfplay_dataset", fake_selfplay)
    monkeypatch.setattr(mod, "run_joint_training", fake_joint)

    manifest = mod.run_autorun(
        data_path="ml/data/human_dataset_canonical_smoke.ndjson",
        output_dir=tmp_path / "autorun",
        device="cpu",
        workers=0,
        no_amp=True,
        decision_max_steps=1,
        belief_max_steps=1,
        selfplay_games_per_cycle=2,
        selfplay_seed_start=1,
        joint_cycles_per_attempt=1,
        max_joint_attempts=3,
        fixed_suite_path="ml/eval/fixed_deals_100.json",
        fixed_suite_max_cases=2,
        strict_param_budget=28_000_000,
        max_phase_retries=2,
        thresholds=mod.TaskThresholds(),
    )

    assert manifest.exists()
    assert calls["joint"] == 2
    progress = json.loads((tmp_path / "autorun" / "autorun_progress.json").read_text(encoding="utf-8"))
    assert any(entry["phase"] == "joint" and entry["passed"] for entry in progress["phase_history"])


def test_run_autorun_fails_when_decision_phase_never_passes(tmp_path: Path, monkeypatch):
    def bad_decision(**kwargs):
        ckpt_dir = Path(kwargs["checkpoints_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": {},
            "metadata": {
                "task": kwargs["task"],
                "accuracy": 0.01,
                "policy_loss": 5.0,
            },
        }
        torch.save(payload, ckpt_dir / f"{kwargs['task']}_latest.pt")
        torch.save(payload, ckpt_dir / f"{kwargs['task']}_best.pt")
        return {"accuracy": 0.01, "policy_loss": 5.0}

    monkeypatch.setattr(mod, "train_decision", bad_decision)

    with pytest.raises(mod.PhaseValidationError):
        mod.run_autorun(
            data_path="ml/data/human_dataset_canonical_smoke.ndjson",
            output_dir=tmp_path / "autorun_fail",
            device="cpu",
            workers=0,
            no_amp=True,
            decision_max_steps=1,
            belief_max_steps=1,
            selfplay_games_per_cycle=2,
            selfplay_seed_start=1,
            joint_cycles_per_attempt=1,
            max_joint_attempts=1,
            fixed_suite_path="ml/eval/fixed_deals_100.json",
            fixed_suite_max_cases=2,
            strict_param_budget=28_000_000,
            max_phase_retries=2,
            thresholds=mod.TaskThresholds(),
        )
    report = json.loads((tmp_path / "autorun_fail" / "phase_reports" / "bidding.json").read_text(encoding="utf-8"))
    assert report["failed"] is True


def test_validate_selfplay_coverage_requires_task_presence(tmp_path: Path):
    path = tmp_path / "selfplay.ndjson"
    records = [
        {"canonical_state": {"global": {"phase": "Bidding"}}},
        {"canonical_state": {"global": {"phase": "PassingForth"}}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    passed, metrics = mod._validate_selfplay_coverage(path, mod.TaskThresholds())
    assert passed is False
    assert metrics["playing"] == 0.0


def test_run_autorun_resumes_from_existing_human_manifest(tmp_path: Path, monkeypatch):
    root = tmp_path / "autorun_resume"
    human_dir = root / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    ckpts = {}
    for task in ("bidding", "passing", "playing", "belief"):
        ckpt = human_dir / f"{task}_validated.pt"
        torch.save({"state_dict": {}, "metadata": {"accuracy": 0.9}}, ckpt)
        ckpts[task] = ckpt

    manifest = {
        "data_path": "ml/data/human_dataset_canonical_smoke.ndjson",
        "device": "cpu",
        "workers": 0,
        "decision_stages": [
            {"task": "bidding", "epochs": 1, "batch": 8, "target_acc": 0.5},
            {"task": "passing", "epochs": 1, "batch": 8, "target_acc": 0.5},
            {"task": "playing", "epochs": 1, "batch": 8, "target_acc": 0.5},
        ],
        "belief_stage": {"epochs": 1, "batch": 8, "target_hidden_acc": 0.5},
        "outputs": {k: str(v) for k, v in ckpts.items()},
        "metadata": {"phase_history": []},
    }
    (human_dir / "human_pretrain_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(mod, "train_decision", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not retrain decision phases")))
    monkeypatch.setattr(mod, "train_belief", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not retrain belief phase")))

    def fake_selfplay(manifest, output, *, games, full_games=None, bidding_games=0, passing_games=0, seed_start, max_steps=300, max_seed_tries_per_target=32, device=None):
        records = [{"canonical_state": {"global": {"phase": "Bidding"}}} for _ in range(8)]
        records.extend({"canonical_state": {"global": {"phase": "PassingForth"}}} for _ in range(4))
        records.extend({"canonical_state": {"global": {"phase": "PassingBack"}}} for _ in range(4))
        records.extend({"canonical_state": {"global": {"phase": "Trick"}}} for _ in range(40))
        Path(output).write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return None

    def fake_joint(**kwargs):
        out = Path(kwargs["checkpoints_dir"]) / "joint_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "metadata": {
                        "last_fixed_suite_eval": {
                            "summary": {
                                "pass_game_rate": 0.1,
                                "contract_made_rate": 0.2,
                                "avg_highest_bid": 140.0,
                                "avg_playing_margin_points": 5.0,
                            }
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return out

    monkeypatch.setattr(mod, "generate_selfplay_dataset", fake_selfplay)
    monkeypatch.setattr(mod, "run_joint_training", fake_joint)

    out_manifest = mod.run_autorun(
        data_path="ml/data/human_dataset_canonical_smoke.ndjson",
        output_dir=root,
        device="cpu",
        workers=0,
        no_amp=True,
        decision_max_steps=1,
        belief_max_steps=1,
        selfplay_games_per_cycle=2,
        selfplay_seed_start=1,
        joint_cycles_per_attempt=1,
        max_joint_attempts=1,
        fixed_suite_path="ml/eval/fixed_deals_100.json",
        fixed_suite_max_cases=2,
        strict_param_budget=28_000_000,
        max_phase_retries=2,
        thresholds=mod.TaskThresholds(),
    )
    assert out_manifest.exists()


def test_train_decision_phase_resumes_latest_attempt_checkpoint(tmp_path: Path, monkeypatch):
    phase_dir = tmp_path / "human" / "playing" / "attempt_01"
    phase_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt = phase_dir / "playing_latest.pt"
    torch.save({"state_dict": {}, "metadata": {"epochs_seen": 2, "accuracy": 0.6, "policy_loss": 0.2}}, resume_ckpt)

    seen = {}

    def fake_decision(**kwargs):
        seen["checkpoint"] = kwargs.get("checkpoint")
        ckpt_dir = Path(kwargs["checkpoints_dir"])
        payload = {
            "state_dict": {},
            "metadata": {
                "task": kwargs["task"],
                "accuracy": 0.7,
                "policy_loss": 0.1,
            },
        }
        torch.save(payload, ckpt_dir / "playing_latest.pt")
        torch.save(payload, ckpt_dir / "playing_best.pt")
        return {"accuracy": 0.7, "policy_loss": 0.1}

    monkeypatch.setattr(mod, "train_decision", fake_decision)
    stage = next(stage for stage in mod.default_decision_stages() if stage.task == "playing")
    ckpt, attempts = mod._train_decision_phase(
        data_path="ml/data/human_dataset_canonical_smoke.ndjson",
        root=tmp_path / "human",
        stage=stage,
        device="cpu",
        workers=0,
        max_steps=1,
        no_amp=True,
        min_epochs=1,
        target_acc_streak=2,
        thresholds=mod.TaskThresholds(),
        max_retries=3,
    )
    assert ckpt.exists()
    assert seen["checkpoint"] == resume_ckpt
    assert attempts[0].attempt == 1
