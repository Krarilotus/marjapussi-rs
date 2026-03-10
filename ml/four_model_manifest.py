from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecisionStageManifest:
    task: str
    epochs: int
    batch: int
    target_acc: float
    min_epochs: int = 1


@dataclass(frozen=True)
class BeliefStageManifest:
    epochs: int
    batch: int
    target_hidden_acc: float
    min_epochs: int = 1


@dataclass(frozen=True)
class FourModelOutputs:
    bidding: Path
    passing: Path
    playing: Path
    belief: Path


@dataclass(frozen=True)
class FourModelManifest:
    path: Path
    data_path: str
    device: str
    workers: int
    decision_stages: tuple[DecisionStageManifest, ...]
    belief_stage: BeliefStageManifest
    outputs: FourModelOutputs
    metadata: dict[str, Any]

    @property
    def root_dir(self) -> Path:
        return self.path.parent

    def missing_outputs(self) -> list[Path]:
        return [path for path in self.outputs.__dict__.values() if not path.exists()]


def write_four_model_manifest(
    path: str | Path,
    *,
    data_path: str,
    device: str,
    workers: int,
    decision_stages: tuple[DecisionStageManifest, ...] | list[DecisionStageManifest],
    belief_stage: BeliefStageManifest,
    outputs: FourModelOutputs,
    metadata: dict[str, Any] | None = None,
) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_root = manifest_path.parent.resolve()

    def _store_path(value: Path) -> str:
        path_value = Path(value)
        if not path_value.is_absolute():
            path_value = path_value.resolve()
        try:
            return str(path_value.relative_to(manifest_root))
        except ValueError:
            return str(path_value)

    payload = {
        "data_path": data_path,
        "device": device,
        "workers": workers,
        "decision_stages": [asdict(stage) for stage in decision_stages],
        "belief_stage": asdict(belief_stage),
        "outputs": {
            "bidding": _store_path(outputs.bidding),
            "passing": _store_path(outputs.passing),
            "playing": _store_path(outputs.playing),
            "belief": _store_path(outputs.belief),
        },
        "metadata": dict(metadata or {}),
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_four_model_manifest(path: str | Path) -> FourModelManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    decision_stages = tuple(
        DecisionStageManifest(
            task=str(stage["task"]),
            epochs=int(stage["epochs"]),
            batch=int(stage["batch"]),
            target_acc=float(stage["target_acc"]),
            min_epochs=int(stage.get("min_epochs", 1)),
        )
        for stage in payload.get("decision_stages", [])
    )
    belief_payload = payload["belief_stage"]
    outputs_payload = payload["outputs"]
    return FourModelManifest(
        path=manifest_path,
        data_path=str(payload["data_path"]),
        device=str(payload["device"]),
        workers=int(payload["workers"]),
        decision_stages=decision_stages,
        belief_stage=BeliefStageManifest(
            epochs=int(belief_payload["epochs"]),
            batch=int(belief_payload["batch"]),
            target_hidden_acc=float(belief_payload["target_hidden_acc"]),
            min_epochs=int(belief_payload.get("min_epochs", 1)),
        ),
        outputs=FourModelOutputs(
            bidding=_resolve_path(root, outputs_payload["bidding"]),
            passing=_resolve_path(root, outputs_payload["passing"]),
            playing=_resolve_path(root, outputs_payload["playing"]),
            belief=_resolve_path(root, outputs_payload["belief"]),
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def validate_four_model_manifest(path: str | Path) -> tuple[FourModelManifest, list[str]]:
    manifest = load_four_model_manifest(path)
    errors: list[str] = []
    seen_tasks = {stage.task for stage in manifest.decision_stages}
    for required in ("bidding", "passing", "playing"):
        if required not in seen_tasks:
            errors.append(f"missing decision stage '{required}'")
    for missing in manifest.missing_outputs():
        errors.append(f"missing output checkpoint: {missing}")
    return manifest, errors
