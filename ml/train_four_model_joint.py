from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    from ml.checkpoint_utils import checkpoint_metadata
    from ml.four_model_manifest import FourModelOutputs, load_four_model_manifest, write_four_model_manifest
    from ml.four_model_governance import evaluate_manifest_behavior, maybe_promote_best_manifest
    from ml.four_model_schedule import (
        BeliefMetricSnapshot,
        JointScheduleConfig,
        compute_joint_schedule,
    )
    from ml.train_belief_from_dataset import train as train_belief
    from ml.train_decision_from_dataset import train as train_decision
except ModuleNotFoundError:
    from checkpoint_utils import checkpoint_metadata
    from four_model_manifest import FourModelOutputs, load_four_model_manifest, write_four_model_manifest
    from four_model_governance import evaluate_manifest_behavior, maybe_promote_best_manifest
    from four_model_schedule import BeliefMetricSnapshot, JointScheduleConfig, compute_joint_schedule
    from train_belief_from_dataset import train as train_belief
    from train_decision_from_dataset import train as train_decision


@dataclass(frozen=True)
class JointCycleSummary:
    cycle_index: int
    schedule_phase: str
    belief_updates: int
    decision_updates: int
    belief_metrics: dict[str, float]


def _coerce_float(data: dict[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def belief_metrics_from_checkpoint(path: str | Path) -> tuple[BeliefMetricSnapshot, int, int]:
    metadata = checkpoint_metadata(path)
    metrics = dict(metadata.get("belief_metrics", {}))
    return (
        BeliefMetricSnapshot(
            card_owner_acc=_coerce_float(metrics, "card_owner_acc", _coerce_float(metadata, "hidden_accuracy", 0.0)),
            constraint_consistency=_coerce_float(metrics, "constraint_consistency", 1.0),
            void_suit_acc=_coerce_float(metrics, "void_suit_acc", 0.0),
            half_pair_acc=_coerce_float(metrics, "half_pair_acc", 0.0),
            calibration_score=_coerce_float(metrics, "calibration_score", 0.0),
        ),
        int(metadata.get("global_step", 0)),
        int(metadata.get("epochs_seen", metadata.get("epoch", 0))),
    )

def run_joint_training(
    base_manifest_path: str | Path,
    sim_data_path: str,
    checkpoints_dir: str | Path,
    *,
    cycles: int = 4,
    device: str | None = None,
    workers: int | None = None,
    belief_epochs_per_update: int = 1,
    decision_epochs_per_update: int = 1,
    belief_max_steps: int = 0,
    decision_max_steps: int = 0,
    no_amp: bool = False,
    schedule_cfg: JointScheduleConfig = JointScheduleConfig(),
    fixed_suite_path: str | None = None,
    fixed_suite_max_cases: int = 0,
    strict_param_budget: int = 28_000_000,
) -> Path:
    base_manifest = load_four_model_manifest(base_manifest_path)
    root = Path(checkpoints_dir)
    root.mkdir(parents=True, exist_ok=True)

    resolved_device = device or base_manifest.device
    resolved_workers = workers if workers is not None else base_manifest.workers
    current_outputs = FourModelOutputs(
        bidding=base_manifest.outputs.bidding,
        passing=base_manifest.outputs.passing,
        playing=base_manifest.outputs.playing,
        belief=base_manifest.outputs.belief,
    )

    history: list[JointCycleSummary] = []
    belief_metrics, belief_steps_seen, belief_epochs_seen = belief_metrics_from_checkpoint(
        current_outputs.belief
    )

    for cycle_idx in range(cycles):
        schedule = compute_joint_schedule(
            belief_metrics,
            steps_seen=belief_steps_seen,
            epochs_seen=belief_epochs_seen,
            cfg=schedule_cfg,
        )

        local_belief_dir = root / "belief"
        for _ in range(schedule.belief_updates):
            last_metrics = train_belief(
                data_path=sim_data_path,
                epochs=belief_epochs_per_update,
                batch=base_manifest.belief_stage.batch,
                device=resolved_device,
                workers=resolved_workers,
                checkpoints_dir=local_belief_dir,
                max_steps=belief_max_steps,
                min_epochs=1,
                target_hidden_acc=0.0,
                target_hidden_streak=1,
                no_amp=no_amp,
                checkpoint=current_outputs.belief,
            )
            current_outputs = FourModelOutputs(
                bidding=current_outputs.bidding,
                passing=current_outputs.passing,
                playing=current_outputs.playing,
                belief=local_belief_dir / "belief_latest.pt",
            )
            belief_metrics = BeliefMetricSnapshot(
                card_owner_acc=float(last_metrics["card_owner_acc"]),
                constraint_consistency=float(last_metrics["constraint_consistency"]),
                void_suit_acc=float(last_metrics["void_suit_acc"]),
                half_pair_acc=float(last_metrics["half_pair_acc"]),
                calibration_score=float(last_metrics["calibration_score"]),
            )
            belief_steps_seen = int(last_metrics["global_step"])
            belief_epochs_seen = int(last_metrics["epochs_seen"])

        task_outputs = {
            "bidding": current_outputs.bidding,
            "passing": current_outputs.passing,
            "playing": current_outputs.playing,
        }
        for _ in range(schedule.decision_updates):
            for stage in base_manifest.decision_stages:
                task_dir = root / stage.task
                train_decision(
                    data_path=sim_data_path,
                    task=stage.task,
                    epochs=decision_epochs_per_update,
                    batch=stage.batch,
                    device=resolved_device,
                    workers=resolved_workers,
                    checkpoints_dir=task_dir,
                    max_steps=decision_max_steps,
                    min_epochs=1,
                    target_acc=0.0,
                    target_acc_streak=1,
                    no_amp=no_amp,
                    checkpoint=task_outputs[stage.task],
                )
                task_outputs[stage.task] = task_dir / f"{stage.task}_latest.pt"

        current_outputs = FourModelOutputs(
            bidding=task_outputs["bidding"],
            passing=task_outputs["passing"],
            playing=task_outputs["playing"],
            belief=current_outputs.belief,
        )
        cycle_summary = JointCycleSummary(
            cycle_index=cycle_idx + 1,
            schedule_phase=schedule.phase_name,
            belief_updates=schedule.belief_updates,
            decision_updates=schedule.decision_updates,
            belief_metrics={
                "card_owner_acc": belief_metrics.card_owner_acc,
                "constraint_consistency": belief_metrics.constraint_consistency,
                "void_suit_acc": belief_metrics.void_suit_acc,
                "half_pair_acc": belief_metrics.half_pair_acc,
                "calibration_score": belief_metrics.calibration_score,
                "steps_seen": float(belief_steps_seen),
                "epochs_seen": float(belief_epochs_seen),
            },
        )
        history.append(cycle_summary)

        manifest_path = write_four_model_manifest(
            root / "joint_manifest.json",
            data_path=sim_data_path,
            device=resolved_device,
            workers=resolved_workers,
            decision_stages=base_manifest.decision_stages,
            belief_stage=base_manifest.belief_stage,
            outputs=current_outputs,
            metadata={
                "training_stage": "joint_simulated",
                "parent_manifest": str(Path(base_manifest_path).resolve()),
                "joint_cycles_completed": cycle_idx + 1,
                "last_schedule_phase": schedule.phase_name,
                "last_belief_metrics": cycle_summary.belief_metrics,
            },
        )
        (root / "joint_summary.json").write_text(
            json.dumps({"cycles": [asdict(entry) for entry in history]}, indent=2),
            encoding="utf-8",
        )
        if fixed_suite_path:
            eval_dir = root / "governance"
            eval_result = evaluate_manifest_behavior(
                manifest_path,
                suite_path=fixed_suite_path,
                device=resolved_device,
                max_cases=fixed_suite_max_cases,
                strict_param_budget=strict_param_budget,
                output_path=eval_dir / f"cycle_{cycle_idx + 1:03d}_fixed_eval.log",
            )
            promoted, best_manifest_path, governance_payload = maybe_promote_best_manifest(
                result=eval_result,
                governance_dir=eval_dir,
            )
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload.setdefault("metadata", {})
            manifest_payload["metadata"]["last_fixed_suite_eval"] = governance_payload
            manifest_payload["metadata"]["best_fixed_suite_manifest"] = str(best_manifest_path)
            manifest_payload["metadata"]["best_fixed_suite_promoted"] = promoted
            manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--sim-data", required=True)
    parser.add_argument("--checkpoints-dir", default="ml/checkpoints/four_model_joint")
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--belief-epochs-per-update", type=int, default=1)
    parser.add_argument("--decision-epochs-per-update", type=int, default=1)
    parser.add_argument("--belief-max-steps", type=int, default=0)
    parser.add_argument("--decision-max-steps", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--fixed-suite", default=None)
    parser.add_argument("--fixed-suite-max-cases", type=int, default=0)
    parser.add_argument("--strict-param-budget", type=int, default=28_000_000)
    args = parser.parse_args()

    manifest_path = run_joint_training(
        base_manifest_path=args.base_manifest,
        sim_data_path=args.sim_data,
        checkpoints_dir=args.checkpoints_dir,
        cycles=args.cycles,
        belief_epochs_per_update=args.belief_epochs_per_update,
        decision_epochs_per_update=args.decision_epochs_per_update,
        belief_max_steps=args.belief_max_steps,
        decision_max_steps=args.decision_max_steps,
        device=args.device,
        workers=args.workers,
        no_amp=args.no_amp,
        fixed_suite_path=args.fixed_suite,
        fixed_suite_max_cases=args.fixed_suite_max_cases,
        strict_param_budget=args.strict_param_budget,
    )
    print(f"Wrote joint manifest: {manifest_path}")
