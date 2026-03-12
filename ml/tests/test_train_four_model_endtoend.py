import json
from pathlib import Path

from ml import train_four_model_endtoend as mod


def test_end_to_end_trainer_regenerates_data_each_cycle(tmp_path: Path, monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_generate(
        manifest_path,
        output_path,
        *,
        games,
        full_games=None,
        bidding_games=0,
        passing_games=0,
        seed_start,
        max_steps=300,
        max_seed_tries_per_target=32,
    ):
        Path(output_path).write_text("{}", encoding="utf-8")
        return mod.SelfPlaySummary(
            games=games,
            records=games,
            avg_actions_per_game=10.0,
            task_counts={"bidding": bidding_games, "passing": passing_games, "playing": full_games or games},
            generated_by_target={"full": full_games or games, "bidding": bidding_games, "passing": passing_games},
            skipped_by_target={"full": 0, "bidding": 0, "passing": 0},
        )

    def fake_joint(
        base_manifest_path,
        sim_data_path,
        checkpoints_dir,
        *,
        cycles,
        device,
        workers,
        belief_epochs_per_update,
        decision_epochs_per_update,
        belief_max_steps,
        decision_max_steps,
        no_amp,
        fixed_suite_path,
        fixed_suite_max_cases,
        strict_param_budget,
    ):
        calls.append((str(base_manifest_path), str(sim_data_path)))
        out = Path(checkpoints_dir) / f"cycle_{len(calls):03d}_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"ok": True}), encoding="utf-8")
        return out

    monkeypatch.setattr(mod, "generate_selfplay_dataset", fake_generate)
    monkeypatch.setattr(mod, "run_joint_training", fake_joint)

    final_manifest = mod.run_end_to_end_training(
        "base_manifest.json",
        tmp_path,
        cycles=2,
        games_per_cycle=3,
        fixed_suite_path="ml/eval/fixed_deals_100.json",
        fixed_suite_max_cases=4,
    )

    assert len(calls) == 2
    assert final_manifest.name == "cycle_002_manifest.json"
    summary = json.loads((tmp_path / "end_to_end_summary.json").read_text(encoding="utf-8"))
    assert len(summary["cycles"]) == 2
