from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from ml.four_model_manifest import (
        BeliefStageManifest,
        DecisionStageManifest,
        FourModelOutputs,
        load_four_model_manifest,
        validate_four_model_manifest,
        write_four_model_manifest,
    )
    from ml.checkpoint_utils import checkpoint_metadata
    from ml.four_model_phase import task_from_phase_name
    from ml.generate_four_model_selfplay import default_selfplay_mix, generate_selfplay_dataset
    from ml.train_belief_from_dataset import train as train_belief
    from ml.train_decision_from_dataset import train as train_decision
    from ml.train_four_model_human_pretrain import default_belief_stage, default_decision_stages
    from ml.train_four_model_joint import run_joint_training
except ModuleNotFoundError:
    from four_model_manifest import (
        BeliefStageManifest,
        DecisionStageManifest,
        FourModelOutputs,
        load_four_model_manifest,
        validate_four_model_manifest,
        write_four_model_manifest,
    )
    from checkpoint_utils import checkpoint_metadata
    from four_model_phase import task_from_phase_name
    from generate_four_model_selfplay import default_selfplay_mix, generate_selfplay_dataset
    from train_belief_from_dataset import train as train_belief
    from train_decision_from_dataset import train as train_decision
    from train_four_model_human_pretrain import default_belief_stage, default_decision_stages
    from train_four_model_joint import run_joint_training


@dataclass(frozen=True)
class TaskThresholds:
    bidding_acc: float = 0.18
    passing_acc: float = 0.18
    playing_acc: float = 0.52
    belief_hidden_acc: float = 0.38
    belief_consistency: float = 0.99
    min_selfplay_bidding_records: int = 8
    min_selfplay_passing_records: int = 8
    min_selfplay_playing_records: int = 32
    max_pass_game_rate: float = 0.40
    min_contract_made_rate: float = 0.10
    min_avg_bid: float = 125.0
    max_avg_bid: float = 210.0


@dataclass(frozen=True)
class PhaseAttemptResult:
    phase: str
    attempt: int
    metrics: dict[str, float]
    passed: bool


class PhaseValidationError(RuntimeError):
    pass


def _load_manifest_metrics(manifest_path: str | Path) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("metadata", {})


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _write_progress(root: Path, payload: dict) -> None:
    (root / "autorun_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_progress(root: Path) -> dict:
    path = root / "autorun_progress.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_phase_metric(checkpoint_path: Path, *keys: str) -> float | None:
    metadata = checkpoint_metadata(checkpoint_path)
    current: object = metadata
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _promote_checkpoint(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _write_phase_report(root: Path, phase: str, payload: dict) -> Path:
    phase_dir = root / "phase_reports"
    phase_dir.mkdir(parents=True, exist_ok=True)
    report_path = phase_dir / f"{phase}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _load_phase_report(root: Path, phase: str) -> dict | None:
    path = root / "phase_reports" / f"{phase}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _latest_passed_attempt(attempts: list[dict]) -> dict | None:
    passed = [entry for entry in attempts if bool(entry.get("passed", False))]
    if not passed:
        return None
    return max(passed, key=lambda entry: int(entry.get("attempt", 0)))


def _latest_attempt_checkpoint(phase_dir: Path, stem: str) -> tuple[int, Path] | None:
    attempt_dirs = sorted(phase_dir.glob("attempt_*"), reverse=True)
    for attempt_dir in attempt_dirs:
        try:
            attempt_num = int(attempt_dir.name.split("_")[-1])
        except ValueError:
            continue
        latest = attempt_dir / f"{stem}_latest.pt"
        if latest.exists():
            return attempt_num, latest
    return None


def _resume_validated_decision_checkpoint(root: Path, task: str, threshold: float) -> tuple[Path | None, list[PhaseAttemptResult]]:
    validated = root / "human" / task / f"{task}_validated.pt"
    report = _load_phase_report(root, task)
    if not validated.exists() or report is None:
        return None, []
    passed = _latest_passed_attempt(report.get("attempts", []))
    if passed is None:
        return None, []
    acc = float(passed.get("metrics", {}).get("accuracy", 0.0))
    if acc < threshold:
        return None, []
    attempts = [
        PhaseAttemptResult(
            phase=str(entry["phase"]),
            attempt=int(entry["attempt"]),
            metrics={k: float(v) for k, v in dict(entry.get("metrics", {})).items()},
            passed=bool(entry.get("passed", False)),
        )
        for entry in report.get("attempts", [])
    ]
    return validated, attempts


def _resume_validated_belief_checkpoint(root: Path, thresholds: TaskThresholds) -> tuple[Path | None, list[PhaseAttemptResult]]:
    validated = root / "human" / "belief" / "belief_validated.pt"
    report = _load_phase_report(root, "belief")
    if not validated.exists() or report is None:
        return None, []
    passed = _latest_passed_attempt(report.get("attempts", []))
    if passed is None:
        return None, []
    metrics = dict(passed.get("metrics", {}))
    if (
        float(metrics.get("card_owner_acc", 0.0)) < thresholds.belief_hidden_acc
        or float(metrics.get("constraint_consistency", 0.0)) < thresholds.belief_consistency
        or float(metrics.get("void_suit_acc", 0.0)) < 0.55
        or float(metrics.get("half_pair_acc", 0.0)) < 0.55
        or float(metrics.get("calibration_score", 0.0)) < 0.45
    ):
        return None, []
    attempts = [
        PhaseAttemptResult(
            phase=str(entry["phase"]),
            attempt=int(entry["attempt"]),
            metrics={k: float(v) for k, v in dict(entry.get("metrics", {})).items()},
            passed=bool(entry.get("passed", False)),
        )
        for entry in report.get("attempts", [])
    ]
    return validated, attempts


def _resume_human_manifest(root: Path) -> Path | None:
    manifest_path = root / "human" / "human_pretrain_manifest.json"
    if not manifest_path.exists():
        return None
    _manifest, errors = validate_four_model_manifest(manifest_path)
    if errors:
        return None
    return manifest_path


def _resume_joint_manifest(root: Path) -> Path | None:
    progress = _read_progress(root)
    current_manifest = progress.get("current_manifest")
    if isinstance(current_manifest, str) and Path(current_manifest).exists():
        return Path(current_manifest)
    manifest_path = root / "joint" / "joint_manifest.json"
    if manifest_path.exists():
        try:
            load_four_model_manifest(manifest_path)
        except Exception:
            return None
        return manifest_path
    return None


def _task_from_record_payload(record: dict) -> str | None:
    canonical = record.get("canonical_state")
    if isinstance(canonical, dict):
        global_state = canonical.get("global")
        if isinstance(global_state, dict):
            phase = global_state.get("phase")
            if isinstance(phase, str):
                return phase
    phase = record.get("phase")
    if isinstance(phase, str):
        return phase
    return None


def _summarize_selfplay_tasks(data_path: str | Path) -> dict[str, int]:
    counts = {"bidding": 0, "passing": 0, "playing": 0}
    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase_name = _task_from_record_payload(record)
            if not phase_name:
                continue
            task = task_from_phase_name(phase_name)
            counts[task] += 1
    return counts


def _validate_selfplay_coverage(data_path: str | Path, thresholds: TaskThresholds) -> tuple[bool, dict[str, float]]:
    counts = _summarize_selfplay_tasks(data_path)
    passed = (
        counts["bidding"] >= thresholds.min_selfplay_bidding_records
        and counts["passing"] >= thresholds.min_selfplay_passing_records
        and counts["playing"] >= thresholds.min_selfplay_playing_records
    )
    return passed, {key: float(value) for key, value in counts.items()}


def _train_decision_phase(
    *,
    data_path: str,
    root: Path,
    stage: DecisionStageManifest,
    device: str,
    workers: int,
    max_steps: int,
    no_amp: bool,
    min_epochs: int,
    target_acc_streak: int,
    thresholds: TaskThresholds,
    max_retries: int,
) -> tuple[Path, list[PhaseAttemptResult]]:
    phase_dir = root / stage.task
    attempts: list[PhaseAttemptResult] = []
    threshold_map = {
        "bidding": thresholds.bidding_acc,
        "passing": thresholds.passing_acc,
        "playing": thresholds.playing_acc,
    }
    required_acc = threshold_map[stage.task]
    resume_attempt = _latest_attempt_checkpoint(phase_dir, stage.task)
    start_attempt = resume_attempt[0] if resume_attempt is not None else 1
    resume_checkpoint = resume_attempt[1] if resume_attempt is not None else None
    for attempt in range(start_attempt, max_retries + 1):
        attempt_dir = phase_dir / f"attempt_{attempt:02d}"
        started_at = time.time()
        summary = train_decision(
            data_path=data_path,
            task=stage.task,
            epochs=stage.epochs,
            batch=stage.batch,
            device=device,
            workers=workers,
            checkpoints_dir=attempt_dir,
            max_steps=max_steps,
            min_epochs=min_epochs,
            target_acc=stage.target_acc,
            target_acc_streak=target_acc_streak,
            no_amp=no_amp,
            checkpoint=resume_checkpoint if attempt == start_attempt else None,
        )
        acc = float(summary["accuracy"])
        latest_checkpoint = attempt_dir / f"{stage.task}_latest.pt"
        best_checkpoint = attempt_dir / f"{stage.task}_best.pt"
        selected_checkpoint = best_checkpoint if best_checkpoint.exists() else latest_checkpoint
        selected_acc = _read_phase_metric(selected_checkpoint, "accuracy") or acc
        policy_loss = _read_phase_metric(selected_checkpoint, "policy_loss")
        duration_sec = time.time() - started_at
        passed = selected_acc >= required_acc
        attempts.append(
            PhaseAttemptResult(
                phase=stage.task,
                attempt=attempt,
                metrics={
                    "accuracy": selected_acc,
                    "policy_loss": policy_loss if policy_loss is not None else float(summary.get("policy_loss", 0.0)),
                    "duration_sec": duration_sec,
                },
                passed=passed,
            )
        )
        if passed:
            promoted_checkpoint = _promote_checkpoint(
                selected_checkpoint,
                phase_dir / f"{stage.task}_validated.pt",
            )
            _write_phase_report(
                root.parent,
                stage.task,
                {
                    "phase": stage.task,
                    "selected_checkpoint": str(promoted_checkpoint),
                    "attempts": [asdict(a) for a in attempts],
                    "target_accuracy": required_acc,
                },
            )
            return promoted_checkpoint, attempts
        resume_checkpoint = None
    _write_phase_report(
        root.parent,
        stage.task,
        {
            "phase": stage.task,
            "attempts": [asdict(a) for a in attempts],
            "target_accuracy": required_acc,
            "failed": True,
        },
    )
    raise PhaseValidationError(
        f"decision phase '{stage.task}' failed semantic validation after {max_retries} attempts"
    )


def _train_belief_phase(
    *,
    data_path: str,
    root: Path,
    stage: BeliefStageManifest,
    device: str,
    workers: int,
    max_steps: int,
    no_amp: bool,
    target_hidden_streak: int,
    thresholds: TaskThresholds,
    max_retries: int,
) -> tuple[Path, list[PhaseAttemptResult]]:
    phase_dir = root / "belief"
    attempts: list[PhaseAttemptResult] = []
    resume_attempt = _latest_attempt_checkpoint(phase_dir, "belief")
    start_attempt = resume_attempt[0] if resume_attempt is not None else 1
    resume_checkpoint = resume_attempt[1] if resume_attempt is not None else None
    for attempt in range(start_attempt, max_retries + 1):
        attempt_dir = phase_dir / f"attempt_{attempt:02d}"
        started_at = time.time()
        summary = train_belief(
            data_path=data_path,
            epochs=stage.epochs,
            batch=stage.batch,
            device=device,
            workers=workers,
            checkpoints_dir=attempt_dir,
            max_steps=max_steps,
            min_epochs=stage.min_epochs,
            target_hidden_acc=stage.target_hidden_acc,
            target_hidden_streak=target_hidden_streak,
            no_amp=no_amp,
            checkpoint=resume_checkpoint if attempt == start_attempt else None,
        )
        latest_checkpoint = attempt_dir / "belief_latest.pt"
        best_checkpoint = attempt_dir / "belief_best.pt"
        selected_checkpoint = best_checkpoint if best_checkpoint.exists() else latest_checkpoint
        hidden_acc = _read_phase_metric(selected_checkpoint, "belief_metrics", "card_owner_acc") or float(summary["card_owner_acc"])
        consistency = _read_phase_metric(selected_checkpoint, "belief_metrics", "constraint_consistency") or float(summary["constraint_consistency"])
        void_acc = _read_phase_metric(selected_checkpoint, "belief_metrics", "void_suit_acc") or float(summary.get("void_suit_acc", 0.0))
        half_pair_acc = _read_phase_metric(selected_checkpoint, "belief_metrics", "half_pair_acc") or float(summary.get("half_pair_acc", 0.0))
        calibration = _read_phase_metric(selected_checkpoint, "belief_metrics", "calibration_score") or float(summary.get("calibration_score", 0.0))
        duration_sec = time.time() - started_at
        passed = (
            hidden_acc >= thresholds.belief_hidden_acc
            and consistency >= thresholds.belief_consistency
            and void_acc >= 0.55
            and half_pair_acc >= 0.55
            and calibration >= 0.45
        )
        attempts.append(
            PhaseAttemptResult(
                phase="belief",
                attempt=attempt,
                metrics={
                    "card_owner_acc": hidden_acc,
                    "constraint_consistency": consistency,
                    "void_suit_acc": void_acc,
                    "half_pair_acc": half_pair_acc,
                    "calibration_score": calibration,
                    "duration_sec": duration_sec,
                },
                passed=passed,
            )
        )
        if passed:
            promoted_checkpoint = _promote_checkpoint(selected_checkpoint, phase_dir / "belief_validated.pt")
            _write_phase_report(
                root.parent,
                "belief",
                {
                    "phase": "belief",
                    "selected_checkpoint": str(promoted_checkpoint),
                    "attempts": [asdict(a) for a in attempts],
                    "targets": {
                        "card_owner_acc": thresholds.belief_hidden_acc,
                        "constraint_consistency": thresholds.belief_consistency,
                        "void_suit_acc": 0.55,
                        "half_pair_acc": 0.55,
                        "calibration_score": 0.45,
                    },
                },
            )
            return promoted_checkpoint, attempts
        resume_checkpoint = None
    _write_phase_report(
        root.parent,
        "belief",
        {
            "phase": "belief",
            "attempts": [asdict(a) for a in attempts],
            "failed": True,
        },
    )
    raise PhaseValidationError(f"belief phase failed semantic validation after {max_retries} attempts")


def _validate_joint_manifest(manifest_path: str | Path, thresholds: TaskThresholds) -> tuple[bool, dict[str, float]]:
    metadata = _load_manifest_metrics(manifest_path)
    eval_payload = metadata.get("last_fixed_suite_eval", {})
    summary = eval_payload.get("summary", {})
    metrics = {
        "pass_game_rate": float(summary.get("pass_game_rate", 1.0)),
        "contract_made_rate": float(summary.get("contract_made_rate", 0.0)),
        "avg_highest_bid": float(summary.get("avg_highest_bid", 0.0)),
        "point_diff": float(summary.get("avg_playing_margin_points", 0.0)),
    }
    passed = (
        metrics["pass_game_rate"] <= thresholds.max_pass_game_rate
        and metrics["contract_made_rate"] >= thresholds.min_contract_made_rate
        and thresholds.min_avg_bid <= metrics["avg_highest_bid"] <= thresholds.max_avg_bid
    )
    return passed, metrics


def _estimate_remaining_seconds(
    phase_history: list[dict],
    *,
    total_decision_phases: int,
    max_joint_attempts: int,
) -> float | None:
    def _durations(phase_names: set[str]) -> list[float]:
        return [
            float(entry.get("metrics", {}).get("duration_sec", 0.0))
            for entry in phase_history
            if entry.get("phase") in phase_names
            and float(entry.get("metrics", {}).get("duration_sec", 0.0)) > 0.0
        ]

    decision_durations = _durations({"bidding", "passing", "playing"})
    belief_durations = _durations({"belief"})
    joint_durations = _durations({"joint"})

    completed_decisions = len(
        {entry.get("phase") for entry in phase_history if entry.get("phase") in {"bidding", "passing", "playing"} and entry.get("passed")}
    )
    remaining_decisions = max(0, total_decision_phases - completed_decisions)
    remaining_belief = 0 if any(entry.get("phase") == "belief" and entry.get("passed") for entry in phase_history) else 1
    completed_joint_attempts = sum(1 for entry in phase_history if entry.get("phase") == "joint")
    remaining_joint = max(0, max_joint_attempts - completed_joint_attempts)

    avg_decision = (sum(decision_durations) / len(decision_durations)) if decision_durations else None
    avg_belief = (sum(belief_durations) / len(belief_durations)) if belief_durations else avg_decision
    avg_joint = (sum(joint_durations) / len(joint_durations)) if joint_durations else avg_decision

    if avg_decision is None and avg_belief is None and avg_joint is None:
        return None

    decision_eta = remaining_decisions * (avg_decision or 0.0)
    belief_eta = remaining_belief * (avg_belief or 0.0)
    joint_eta = remaining_joint * (avg_joint or avg_belief or avg_decision or 0.0)
    return decision_eta + belief_eta + joint_eta


def run_autorun(
    *,
    data_path: str,
    output_dir: str | Path,
    device: str,
    workers: int,
    no_amp: bool,
    decision_max_steps: int,
    belief_max_steps: int,
    selfplay_games_per_cycle: int,
    selfplay_seed_start: int,
    joint_cycles_per_attempt: int,
    max_joint_attempts: int,
    fixed_suite_path: str,
    fixed_suite_max_cases: int,
    strict_param_budget: int,
    max_phase_retries: int,
    thresholds: TaskThresholds,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_started_at = time.time()
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    decision_stages = default_decision_stages()
    belief_stage = default_belief_stage()
    progress = _read_progress(root)
    phase_history: list[dict] = list(progress.get("phase_history", []))

    outputs: dict[str, Path] = {}
    current_manifest = _resume_human_manifest(root)
    if current_manifest is not None:
        manifest = load_four_model_manifest(current_manifest)
        outputs = {
            "bidding": manifest.outputs.bidding,
            "passing": manifest.outputs.passing,
            "playing": manifest.outputs.playing,
            "belief": manifest.outputs.belief,
        }
    else:
        threshold_map = {
            "bidding": thresholds.bidding_acc,
            "passing": thresholds.passing_acc,
            "playing": thresholds.playing_acc,
        }
        for stage in decision_stages:
            resumed_ckpt, resumed_attempts = _resume_validated_decision_checkpoint(root, stage.task, threshold_map[stage.task])
            if resumed_ckpt is not None:
                outputs[stage.task] = resumed_ckpt
                if not any(entry.get("phase") == stage.task for entry in phase_history):
                    phase_history.extend(asdict(a) for a in resumed_attempts)
                continue
            ckpt, attempts = _train_decision_phase(
                data_path=data_path,
                root=root / "human",
                stage=stage,
                device=device,
                workers=workers,
                max_steps=decision_max_steps,
                no_amp=no_amp,
                min_epochs=stage.min_epochs,
                target_acc_streak=2,
                thresholds=thresholds,
                max_retries=max_phase_retries,
            )
            outputs[stage.task] = ckpt
            phase_history.extend(asdict(a) for a in attempts)
            _write_progress(root, {"phase_history": phase_history})
            eta = _estimate_remaining_seconds(
                phase_history,
                total_decision_phases=len(decision_stages),
                max_joint_attempts=max_joint_attempts,
            )
            print(f"[INFO] RunETA after {stage.task}: {_format_duration(eta)} | Elapsed: {_format_duration(time.time() - run_started_at)}")

        belief_ckpt, belief_attempts = _resume_validated_belief_checkpoint(root, thresholds)
        if belief_ckpt is not None:
            outputs["belief"] = belief_ckpt
            if not any(entry.get("phase") == "belief" for entry in phase_history):
                phase_history.extend(asdict(a) for a in belief_attempts)
        else:
            belief_ckpt, belief_attempts = _train_belief_phase(
                data_path=data_path,
                root=root / "human",
                stage=belief_stage,
                device=device,
                workers=workers,
                max_steps=belief_max_steps,
                no_amp=no_amp,
                target_hidden_streak=2,
                thresholds=thresholds,
                max_retries=max_phase_retries,
            )
            outputs["belief"] = belief_ckpt
            phase_history.extend(asdict(a) for a in belief_attempts)
            _write_progress(root, {"phase_history": phase_history})
            eta = _estimate_remaining_seconds(
                phase_history,
                total_decision_phases=len(decision_stages),
                max_joint_attempts=max_joint_attempts,
            )
            print(f"[INFO] RunETA after belief: {_format_duration(eta)} | Elapsed: {_format_duration(time.time() - run_started_at)}")

        current_manifest = write_four_model_manifest(
            root / "human" / "human_pretrain_manifest.json",
            data_path=data_path,
            device=device,
            workers=workers,
            decision_stages=decision_stages,
            belief_stage=belief_stage,
            outputs=FourModelOutputs(
                bidding=outputs["bidding"],
                passing=outputs["passing"],
                playing=outputs["playing"],
                belief=outputs["belief"],
            ),
            metadata={
                "training_stage": "human_pretrain",
                "phase_history": phase_history,
            },
        )

    resumed_joint_manifest = _resume_joint_manifest(root)
    if resumed_joint_manifest is not None:
        current_manifest = resumed_joint_manifest

    joint_passed = False
    completed_joint_attempts = sum(1 for entry in phase_history if entry.get("phase") == "joint")
    if completed_joint_attempts > 0 and current_manifest is not None:
        joint_passed, _existing_metrics = _validate_joint_manifest(current_manifest, thresholds)
        if joint_passed:
            _write_progress(
                root,
                {
                    "phase_history": phase_history,
                    "current_manifest": str(current_manifest),
                },
            )
            return current_manifest

    for attempt in range(completed_joint_attempts + 1, max_joint_attempts + 1):
        sim_data = data_root / f"joint_cycle_{attempt:03d}.ndjson"
        mix = default_selfplay_mix(selfplay_games_per_cycle)
        generate_selfplay_dataset(
            current_manifest,
            sim_data,
            games=selfplay_games_per_cycle,
            full_games=mix.full_games,
            bidding_games=mix.bidding_games,
            passing_games=mix.passing_games,
            seed_start=selfplay_seed_start + (attempt - 1) * selfplay_games_per_cycle,
            device=device,
        )
        coverage_ok, coverage_metrics = _validate_selfplay_coverage(sim_data, thresholds)
        if not coverage_ok:
            phase_history.append(
                asdict(
                    PhaseAttemptResult(
                        phase="joint_coverage",
                        attempt=attempt,
                        metrics=coverage_metrics,
                        passed=False,
                    )
                )
            )
            _write_progress(
                root,
                {
                    "phase_history": phase_history,
                    "current_manifest": str(current_manifest),
                },
            )
            eta = _estimate_remaining_seconds(
                phase_history,
                total_decision_phases=len(decision_stages),
                max_joint_attempts=max_joint_attempts,
            )
            print(f"[INFO] RunETA after joint coverage retry {attempt}: {_format_duration(eta)} | Elapsed: {_format_duration(time.time() - run_started_at)}")
            continue
        joint_started_at = time.time()
        current_manifest = run_joint_training(
            base_manifest_path=current_manifest,
            sim_data_path=str(sim_data),
            checkpoints_dir=root / "joint",
            cycles=joint_cycles_per_attempt,
            device=device,
            workers=workers,
            belief_epochs_per_update=1,
            decision_epochs_per_update=1,
            belief_max_steps=belief_max_steps,
            decision_max_steps=decision_max_steps,
            no_amp=no_amp,
            fixed_suite_path=fixed_suite_path,
            fixed_suite_max_cases=fixed_suite_max_cases,
            strict_param_budget=strict_param_budget,
        )
        passed, metrics = _validate_joint_manifest(current_manifest, thresholds)
        phase_history.append(
            asdict(
                PhaseAttemptResult(
                    phase="joint",
                    attempt=attempt,
                    metrics={**metrics, "duration_sec": time.time() - joint_started_at},
                    passed=passed,
                )
            )
        )
        _write_progress(
            root,
            {
                "phase_history": phase_history,
                "current_manifest": str(current_manifest),
            },
        )
        eta = _estimate_remaining_seconds(
            phase_history,
            total_decision_phases=len(decision_stages),
            max_joint_attempts=max_joint_attempts,
        )
        print(f"[INFO] RunETA after joint attempt {attempt}: {_format_duration(eta)} | Elapsed: {_format_duration(time.time() - run_started_at)}")
        if passed:
            joint_passed = True
            break

    if not joint_passed:
        raise PhaseValidationError(
            f"joint phase failed semantic validation after {max_joint_attempts} attempts"
        )

    return current_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="ml/runs/four_model_autorun")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--decision-max-steps", type=int, default=0)
    parser.add_argument("--belief-max-steps", type=int, default=0)
    parser.add_argument("--selfplay-games-per-cycle", type=int, default=64)
    parser.add_argument("--selfplay-seed-start", type=int, default=1)
    parser.add_argument("--joint-cycles-per-attempt", type=int, default=1)
    parser.add_argument("--max-joint-attempts", type=int, default=16)
    parser.add_argument("--fixed-suite", default="ml/eval/fixed_deals_100.json")
    parser.add_argument("--fixed-suite-max-cases", type=int, default=16)
    parser.add_argument("--strict-param-budget", type=int, default=28_000_000)
    parser.add_argument("--max-phase-retries", type=int, default=3)
    args = parser.parse_args()

    manifest = run_autorun(
        data_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        workers=args.workers,
        no_amp=args.no_amp,
        decision_max_steps=args.decision_max_steps,
        belief_max_steps=args.belief_max_steps,
        selfplay_games_per_cycle=args.selfplay_games_per_cycle,
        selfplay_seed_start=args.selfplay_seed_start,
        joint_cycles_per_attempt=args.joint_cycles_per_attempt,
        max_joint_attempts=args.max_joint_attempts,
        fixed_suite_path=args.fixed_suite,
        fixed_suite_max_cases=args.fixed_suite_max_cases,
        strict_param_budget=args.strict_param_budget,
        max_phase_retries=args.max_phase_retries,
        thresholds=TaskThresholds(),
    )
    print(f"Wrote autorun manifest: {manifest}")


if __name__ == "__main__":
    main()
