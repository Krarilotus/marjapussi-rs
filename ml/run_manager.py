from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "ml" / "runs"
CURRENT_RUN_PATH = RUNS_DIR / ".current_four_model_run.json"


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    output_dir: str
    data: str
    device: str = "cuda"
    workers: int = 0
    selfplay_games_per_cycle: int = 64
    max_joint_attempts: int = 16
    fixed_suite: str = "ml/eval/fixed_deals_100.json"
    fixed_suite_max_cases: int = 16


def _timestamped_run_name(prefix: str = "four_model_run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _run_dir(run_name: str) -> Path:
    return RUNS_DIR / run_name


def _run_config_path(run_dir: Path) -> Path:
    return run_dir / "run_config.json"


def _run_pid_path(run_dir: Path) -> Path:
    return run_dir / "launcher.pid"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_config(config: RunConfig) -> Path:
    run_dir = _run_dir(config.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = _run_config_path(run_dir)
    _write_json(path, asdict(config))
    return path


def load_run_config(run_name: str) -> RunConfig:
    run_dir = _run_dir(run_name)
    config_path = _run_config_path(run_dir)
    if config_path.exists():
        payload = _load_json(config_path)
        return RunConfig(**payload)
    return infer_run_config(run_name)


def set_current_run(run_name: str) -> None:
    _write_json(CURRENT_RUN_PATH, {"run_name": run_name})


def get_current_run_name() -> str | None:
    if not CURRENT_RUN_PATH.exists():
        return None
    try:
        payload = _load_json(CURRENT_RUN_PATH)
    except json.JSONDecodeError:
        return None
    run_name = payload.get("run_name")
    return str(run_name) if run_name else None


def resolve_run_name(run_name: str | None = None) -> str:
    if run_name:
        return run_name
    current = get_current_run_name()
    if current and _run_dir(current).exists():
        return current
    candidates = sorted(
        [p for p in RUNS_DIR.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("no run directory found")
    return candidates[0].name


def build_run_config(
    *,
    run_name: str | None,
    data: str,
    device: str,
    workers: int,
    selfplay_games_per_cycle: int,
    max_joint_attempts: int,
    fixed_suite: str,
    fixed_suite_max_cases: int,
) -> RunConfig:
    effective_run_name = run_name or _timestamped_run_name("four_model_managed")
    return RunConfig(
        run_name=effective_run_name,
        output_dir=str(_run_dir(effective_run_name).as_posix()),
        data=data,
        device=device,
        workers=workers,
        selfplay_games_per_cycle=selfplay_games_per_cycle,
        max_joint_attempts=max_joint_attempts,
        fixed_suite=fixed_suite,
        fixed_suite_max_cases=fixed_suite_max_cases,
    )


def infer_run_config(run_name: str) -> RunConfig:
    run_dir = _run_dir(run_name)
    if not run_dir.exists():
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    data = "ml/data/human_dataset_canonical.ndjson"
    manifest_candidates = [
        run_dir / "human" / "human_pretrain_manifest.json",
        run_dir / "joint" / "joint_manifest.json",
    ]
    for manifest_path in manifest_candidates:
        if not manifest_path.exists():
            continue
        try:
            manifest = _load_json(manifest_path)
        except json.JSONDecodeError:
            continue
        data = str(manifest.get("data_path") or data)
        device = str(manifest.get("device") or "cuda")
        workers = int(manifest.get("workers") or 0)
        return RunConfig(
            run_name=run_name,
            output_dir=str(run_dir.as_posix()),
            data=data,
            device=device,
            workers=workers,
        )

    return RunConfig(
        run_name=run_name,
        output_dir=str(run_dir.as_posix()),
        data=data,
    )


def _autorun_command(config: RunConfig) -> list[str]:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.exists():
        python = ROOT / ".venv" / "bin" / "python"
    return [
        str(python),
        "-u",
        "ml/train_four_model_autorun.py",
        "--data",
        config.data,
        "--output-dir",
        config.output_dir,
        "--device",
        config.device,
        "--workers",
        str(config.workers),
        "--selfplay-games-per-cycle",
        str(config.selfplay_games_per_cycle),
        "--max-joint-attempts",
        str(config.max_joint_attempts),
        "--fixed-suite",
        config.fixed_suite,
        "--fixed-suite-max-cases",
        str(config.fixed_suite_max_cases),
    ]


def _launch_process(config: RunConfig) -> int:
    run_dir = _run_dir(config.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "autorun.stdout.log"
    stderr_path = run_dir / "autorun.stderr.log"
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    kwargs: dict = {
        "cwd": str(ROOT),
        "stdout": stdout_handle,
        "stderr": stderr_handle,
    }
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        proc = subprocess.Popen(_autorun_command(config), creationflags=creationflags, **kwargs)
    else:
        proc = subprocess.Popen(_autorun_command(config), start_new_session=True, **kwargs)
    _run_pid_path(run_dir).write_text(str(proc.pid), encoding="utf-8")
    return int(proc.pid)


def start_run(config: RunConfig) -> int:
    write_run_config(config)
    set_current_run(config.run_name)
    return _launch_process(config)


def resume_run(run_name: str | None = None) -> int:
    effective_run_name = resolve_run_name(run_name)
    config = load_run_config(effective_run_name)
    set_current_run(effective_run_name)
    return _launch_process(config)


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return str(pid) in result.stdout
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _discover_pid_for_run(run_name: str) -> int | None:
    if os.name == "nt":
        command = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*ml/train_four_model_autorun.py*' "
            f"-and $_.CommandLine -like '*{run_name}*' }} | "
            "Select-Object -First 1 -ExpandProperty ProcessId"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )
        text = result.stdout.strip()
        return int(text) if text.isdigit() else None
    result = subprocess.run(
        ["pgrep", "-af", "ml/train_four_model_autorun.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        if run_name not in line:
            continue
        pid_text = line.split(maxsplit=1)[0]
        if pid_text.isdigit():
            return int(pid_text)
    return None


def _resolve_pid(run_name: str, run_dir: Path) -> int | None:
    pid_path = _run_pid_path(run_dir)
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
        if pid is not None and _pid_alive(pid):
            return pid
    pid = _discover_pid_for_run(run_name)
    if pid is not None:
        pid_path.write_text(str(pid), encoding="utf-8")
    return pid


def stop_run(run_name: str | None = None) -> bool:
    effective_run_name = resolve_run_name(run_name)
    run_dir = _run_dir(effective_run_name)
    pid = _resolve_pid(effective_run_name, run_dir)
    if pid is None:
        return False
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    else:
        os.killpg(pid, signal.SIGTERM)
    return True


def run_status(run_name: str | None = None) -> dict:
    effective_run_name = resolve_run_name(run_name)
    run_dir = _run_dir(effective_run_name)
    pid = _resolve_pid(effective_run_name, run_dir)
    return {
        "run_name": effective_run_name,
        "run_dir": str(run_dir),
        "pid": pid,
        "alive": _pid_alive(pid) if pid is not None else False,
        "stdout": str(run_dir / "autorun.stdout.log"),
        "stderr": str(run_dir / "autorun.stderr.log"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    start_p = sub.add_parser("start")
    start_p.add_argument("--run-name", default=None)
    start_p.add_argument("--data", default="ml/data/human_dataset_canonical.ndjson")
    start_p.add_argument("--device", default="cuda")
    start_p.add_argument("--workers", type=int, default=0)
    start_p.add_argument("--selfplay-games-per-cycle", type=int, default=64)
    start_p.add_argument("--max-joint-attempts", type=int, default=16)
    start_p.add_argument("--fixed-suite", default="ml/eval/fixed_deals_100.json")
    start_p.add_argument("--fixed-suite-max-cases", type=int, default=16)

    resume_p = sub.add_parser("resume")
    resume_p.add_argument("--run-name", default=None)

    stop_p = sub.add_parser("stop")
    stop_p.add_argument("--run-name", default=None)

    status_p = sub.add_parser("status")
    status_p.add_argument("--run-name", default=None)

    current_p = sub.add_parser("current")
    current_p.add_argument("--run-name", default=None)

    args = parser.parse_args()
    if args.cmd == "start":
        config = build_run_config(
            run_name=args.run_name,
            data=args.data,
            device=args.device,
            workers=args.workers,
            selfplay_games_per_cycle=args.selfplay_games_per_cycle,
            max_joint_attempts=args.max_joint_attempts,
            fixed_suite=args.fixed_suite,
            fixed_suite_max_cases=args.fixed_suite_max_cases,
        )
        pid = start_run(config)
        print(json.dumps({"run_name": config.run_name, "pid": pid, "run_dir": config.output_dir}))
    elif args.cmd == "resume":
        pid = resume_run(args.run_name)
        print(json.dumps(run_status(args.run_name)))
    elif args.cmd == "stop":
        stopped = stop_run(args.run_name)
        print(json.dumps({"stopped": stopped, **run_status(args.run_name)}))
    elif args.cmd == "status":
        print(json.dumps(run_status(args.run_name)))
    elif args.cmd == "current":
        print(json.dumps({"run_name": resolve_run_name(args.run_name)}))


if __name__ == "__main__":
    main()
