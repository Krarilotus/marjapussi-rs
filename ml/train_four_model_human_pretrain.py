from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

try:
    from ml.four_model_manifest import (
        BeliefStageManifest,
        DecisionStageManifest,
        FourModelOutputs,
        write_four_model_manifest,
    )
    from ml.train_belief_from_dataset import train as train_belief
    from ml.train_decision_from_dataset import train as train_decision
except ModuleNotFoundError:
    from four_model_manifest import (
        BeliefStageManifest,
        DecisionStageManifest,
        FourModelOutputs,
        write_four_model_manifest,
    )
    from train_belief_from_dataset import train as train_belief
    from train_decision_from_dataset import train as train_decision


def default_decision_stages() -> tuple[DecisionStageManifest, ...]:
    return (
        DecisionStageManifest(task="bidding", epochs=12, batch=256, target_acc=0.45, min_epochs=4),
        DecisionStageManifest(task="passing", epochs=12, batch=256, target_acc=0.40, min_epochs=4),
        DecisionStageManifest(task="playing", epochs=12, batch=256, target_acc=0.42, min_epochs=4),
    )


def default_belief_stage() -> BeliefStageManifest:
    return BeliefStageManifest(epochs=12, batch=256, target_hidden_acc=0.70, min_epochs=4)


def run_human_pretraining(
    data_path: str,
    checkpoints_dir: str | Path,
    device: str = "cpu",
    workers: int = 4,
    max_steps_decision: int = 0,
    max_steps_belief: int = 0,
    no_amp: bool = False,
) -> Path:
    root = Path(checkpoints_dir)
    root.mkdir(parents=True, exist_ok=True)
    decision_stages = default_decision_stages()
    belief_stage = default_belief_stage()

    for stage in decision_stages:
        train_decision(
            data_path=data_path,
            task=stage.task,
            epochs=stage.epochs,
            batch=stage.batch,
            device=device,
            workers=workers,
            checkpoints_dir=root / stage.task,
            max_steps=max_steps_decision,
            min_epochs=stage.min_epochs,
            target_acc=stage.target_acc,
            target_acc_streak=2,
            no_amp=no_amp,
        )

    train_belief(
        data_path=data_path,
        epochs=belief_stage.epochs,
        batch=belief_stage.batch,
        device=device,
        workers=workers,
        checkpoints_dir=root / "belief",
        max_steps=max_steps_belief,
        min_epochs=belief_stage.min_epochs,
        target_hidden_acc=belief_stage.target_hidden_acc,
        target_hidden_streak=2,
        no_amp=no_amp,
    )

    return write_four_model_manifest(
        root / "human_pretrain_manifest.json",
        data_path=data_path,
        device=device,
        workers=workers,
        decision_stages=decision_stages,
        belief_stage=belief_stage,
        outputs=FourModelOutputs(
            bidding=root / "bidding" / "bidding_latest.pt",
            passing=root / "passing" / "passing_latest.pt",
            playing=root / "playing" / "playing_latest.pt",
            belief=root / "belief" / "belief_latest.pt",
        ),
        metadata={
            "training_stage": "human_pretrain",
            "human_data_path": data_path,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoints-dir", default="ml/checkpoints/four_model_human")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps-decision", type=int, default=0)
    parser.add_argument("--max-steps-belief", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    manifest_path = run_human_pretraining(
        data_path=args.data,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        workers=args.workers,
        max_steps_decision=args.max_steps_decision,
        max_steps_belief=args.max_steps_belief,
        no_amp=args.no_amp,
    )
    print(f"Wrote manifest: {manifest_path}")
