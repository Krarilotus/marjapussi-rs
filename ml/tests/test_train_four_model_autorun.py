import json
from pathlib import Path

import ml.train_four_model_autorun as mod


def test_run_autorun_retries_joint_until_metrics_pass(tmp_path: Path, monkeypatch):
    calls = {"joint": 0, "selfplay": 0}

    monkeypatch.setattr(
        mod,
        "train_decision",
        lambda **kwargs: {
            "accuracy": 0.7,
            "epochs_seen": 1.0,
            "global_step": 1.0,
            "policy_loss": 0.1,
            "value_loss": 0.1,
            "aux_loss": 0.1,
        },
    )
    monkeypatch.setattr(
        mod,
        "train_belief",
        lambda **kwargs: {
            "card_owner_acc": 0.5,
            "constraint_consistency": 1.0,
            "void_suit_acc": 0.8,
            "half_pair_acc": 0.8,
            "calibration_score": 0.5,
            "global_step": 2.0,
            "epochs_seen": 1.0,
        },
    )

    def fake_selfplay(manifest, output, *, games, seed_start, max_steps=300):
        calls["selfplay"] += 1
        Path(output).write_text("{}", encoding="utf-8")
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
