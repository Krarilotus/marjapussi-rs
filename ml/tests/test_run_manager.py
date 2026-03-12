from __future__ import annotations

import json
from pathlib import Path

from ml import run_manager


def _configure_paths(monkeypatch, tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs"
    current = runs_dir / ".current_four_model_run.json"
    monkeypatch.setattr(run_manager, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(run_manager, "CURRENT_RUN_PATH", current)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def test_build_run_config_uses_expected_defaults(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    config = run_manager.build_run_config(
        run_name="demo_run",
        data="ml/data/human_dataset_canonical.ndjson",
        device="cuda",
        workers=0,
        selfplay_games_per_cycle=64,
        max_joint_attempts=16,
        fixed_suite="ml/eval/fixed_deals_100.json",
        fixed_suite_max_cases=16,
    )
    assert config.run_name == "demo_run"
    assert config.device == "cuda"
    assert config.workers == 0
    assert config.output_dir == str((runs_dir / "demo_run").as_posix())


def test_resolve_run_name_prefers_existing_current(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    (runs_dir / "older").mkdir()
    (runs_dir / "current_run").mkdir()
    run_manager.set_current_run("current_run")
    assert run_manager.resolve_run_name() == "current_run"


def test_resolve_run_name_falls_back_when_current_missing(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    missing = "missing_run"
    run_manager.set_current_run(missing)
    (runs_dir / "latest_real").mkdir()
    assert run_manager.resolve_run_name() == "latest_real"


def test_start_run_writes_config_and_current(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(run_manager, "_launch_process", lambda config: 4242)
    config = run_manager.build_run_config(
        run_name="managed_run",
        data="ml/data/human_dataset_canonical.ndjson",
        device="cuda",
        workers=0,
        selfplay_games_per_cycle=64,
        max_joint_attempts=16,
        fixed_suite="ml/eval/fixed_deals_100.json",
        fixed_suite_max_cases=16,
    )
    pid = run_manager.start_run(config)
    assert pid == 4242
    written = json.loads((runs_dir / "managed_run" / "run_config.json").read_text(encoding="utf-8"))
    assert written["run_name"] == "managed_run"
    assert json.loads(run_manager.CURRENT_RUN_PATH.read_text(encoding="utf-8"))["run_name"] == "managed_run"


def test_resume_run_reuses_saved_config(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    config = run_manager.build_run_config(
        run_name="resume_me",
        data="ml/data/human_dataset_canonical.ndjson",
        device="cuda",
        workers=0,
        selfplay_games_per_cycle=64,
        max_joint_attempts=16,
        fixed_suite="ml/eval/fixed_deals_100.json",
        fixed_suite_max_cases=16,
    )
    run_manager.write_run_config(config)
    launched = {}

    def _fake_launch(cfg):
        launched["config"] = cfg
        return 777

    monkeypatch.setattr(run_manager, "_launch_process", _fake_launch)
    pid = run_manager.resume_run("resume_me")
    assert pid == 777
    assert launched["config"].run_name == "resume_me"
    assert launched["config"].device == "cuda"


def test_load_run_config_falls_back_to_manifest_inference(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    run_dir = runs_dir / "legacy_run" / "human"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "human_pretrain_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_path": "ml/data/legacy.ndjson",
                "device": "cuda",
                "workers": 2,
            }
        ),
        encoding="utf-8",
    )
    config = run_manager.load_run_config("legacy_run")
    assert config.run_name == "legacy_run"
    assert config.data == "ml/data/legacy.ndjson"
    assert config.device == "cuda"
    assert config.workers == 2


def test_stop_run_invokes_taskkill_on_windows(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    run_dir = runs_dir / "stop_me"
    run_dir.mkdir()
    (run_dir / "launcher.pid").write_text("999", encoding="utf-8")
    monkeypatch.setattr(run_manager, "resolve_run_name", lambda run_name=None: "stop_me")
    monkeypatch.setattr(run_manager, "_resolve_pid", lambda run_name, run_dir: 999)
    monkeypatch.setattr(run_manager.os, "name", "nt")
    calls = []

    def _fake_run(args, check=False, **kwargs):
        calls.append(args)
        class _Result:
            returncode = 0
        return _Result()

    monkeypatch.setattr(run_manager.subprocess, "run", _fake_run)
    assert run_manager.stop_run() is True
    assert calls == [["taskkill", "/PID", "999", "/T", "/F"]]


def test_run_status_reports_pid_and_paths(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    run_dir = runs_dir / "status_me"
    run_dir.mkdir()
    monkeypatch.setattr(run_manager, "resolve_run_name", lambda run_name=None: "status_me")
    monkeypatch.setattr(run_manager, "_resolve_pid", lambda run_name, run_dir: 123)
    monkeypatch.setattr(run_manager, "_pid_alive", lambda pid: pid == 123)
    status = run_manager.run_status()
    assert status["run_name"] == "status_me"
    assert status["pid"] == 123
    assert status["alive"] is True
    assert status["stdout"].endswith("autorun.stdout.log")


def test_run_status_discovers_pid_for_legacy_run(monkeypatch, tmp_path):
    runs_dir = _configure_paths(monkeypatch, tmp_path)
    run_dir = runs_dir / "legacy_running"
    run_dir.mkdir()
    monkeypatch.setattr(run_manager, "resolve_run_name", lambda run_name=None: "legacy_running")
    monkeypatch.setattr(run_manager, "_discover_pid_for_run", lambda run_name: 456)
    monkeypatch.setattr(run_manager, "_pid_alive", lambda pid: pid == 456)
    status = run_manager.run_status()
    assert status["pid"] == 456
    assert status["alive"] is True
    assert (run_dir / "launcher.pid").read_text(encoding="utf-8").strip() == "456"
