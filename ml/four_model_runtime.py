from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from ml.belief_decoder import decode_hidden_card_assignment
    from ml.belief_model import BeliefNet
    from ml.checkpoint_utils import load_model_checkpoint
    from ml.decision_model import BiddingNet, PassingNet, PlayingNet
    from ml.decision_state import build_decision_features_from_record, task_from_phase_name
    from ml.four_model_manifest import FourModelManifest, load_four_model_manifest
    from ml.neurosymbolic_dataset import NUM_OWNER_CLASSES, build_belief_features
    from ml.neurosymbolic_state import CanonicalState
except ModuleNotFoundError:
    from belief_decoder import decode_hidden_card_assignment
    from belief_model import BeliefNet
    from checkpoint_utils import load_model_checkpoint
    from decision_model import BiddingNet, PassingNet, PlayingNet
    from decision_state import build_decision_features_from_record, task_from_phase_name
    from four_model_manifest import FourModelManifest, load_four_model_manifest
    from neurosymbolic_dataset import NUM_OWNER_CLASSES, build_belief_features
    from neurosymbolic_state import CanonicalState


TASK_TO_MODEL = {
    "bidding": BiddingNet,
    "passing": PassingNet,
    "playing": PlayingNet,
}


@dataclass(frozen=True)
class FourModelBundle:
    manifest: FourModelManifest
    belief_model: BeliefNet
    decision_models: dict[str, torch.nn.Module]
    device: str


def normalize_runtime_record(payload: dict) -> dict:
    if "obs" in payload and "canonical_state" in payload:
        return payload
    if "phase" in payload and "legal_actions" in payload:
        record = {"obs": payload, "canonical_state": payload.get("canonical_state")}
        if payload.get("belief_targets") is not None:
            record["belief_targets"] = payload["belief_targets"]
        return record
    raise KeyError("runtime payload must contain either {'obs', 'canonical_state'} or an obs dict with canonical_state")


def load_belief_checkpoint(path: str | Path, device: str = "cpu") -> BeliefNet:
    model = BeliefNet().to(device)
    load_model_checkpoint(model, path, device=device)
    model.eval()
    return model


def load_decision_checkpoint(task: str, path: str | Path, device: str = "cpu") -> torch.nn.Module:
    if task not in TASK_TO_MODEL:
        raise ValueError(f"unsupported task '{task}'")
    model = TASK_TO_MODEL[task]().to(device)
    load_model_checkpoint(model, path, device=device, expected_task=task)
    model.eval()
    return model


def load_four_model_bundle(manifest_path: str | Path, device: str | None = None) -> FourModelBundle:
    manifest = load_four_model_manifest(manifest_path)
    resolved_device = device or manifest.device
    decision_models = {
        "bidding": load_decision_checkpoint("bidding", manifest.outputs.bidding, resolved_device),
        "passing": load_decision_checkpoint("passing", manifest.outputs.passing, resolved_device),
        "playing": load_decision_checkpoint("playing", manifest.outputs.playing, resolved_device),
    }
    belief_model = load_belief_checkpoint(manifest.outputs.belief, resolved_device)
    return FourModelBundle(
        manifest=manifest,
        belief_model=belief_model,
        decision_models=decision_models,
        device=resolved_device,
    )


def owner_classes_to_onehot(owner_classes: tuple[int, ...]) -> torch.Tensor:
    rows = []
    for owner_idx in owner_classes:
        row = [0.0] * NUM_OWNER_CLASSES
        if owner_idx >= 0:
            row[owner_idx] = 1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


@torch.no_grad()
def predict_belief_owner_onehot(
    state: CanonicalState,
    belief_model: torch.nn.Module,
    device: str = "cpu",
) -> torch.Tensor:
    feats = build_belief_features(state)
    outputs = belief_model(
        card_features=feats.card_features.unsqueeze(0).to(device),
        player_features=feats.player_features[1:].unsqueeze(0).to(device),
        global_features=feats.global_features.unsqueeze(0).to(device),
    )
    decoded = decode_hidden_card_assignment(outputs["card_logits"].squeeze(0).cpu(), state)
    return owner_classes_to_onehot(decoded.card_owner_classes)


def build_runtime_decision_features(
    record: dict,
    belief_model: torch.nn.Module,
    device: str = "cpu",
):
    record = normalize_runtime_record(record)
    state = CanonicalState.from_record(record)
    belief_owner = predict_belief_owner_onehot(state, belief_model, device=device)
    return build_decision_features_from_record(
        record,
        use_teacher_belief=False,
        belief_owner_override=belief_owner,
    )


@torch.no_grad()
def predict_decision_outputs(
    record: dict,
    model: torch.nn.Module,
    belief_model: torch.nn.Module,
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    record = normalize_runtime_record(record)
    features = build_runtime_decision_features(record, belief_model, device=device)
    outputs = model(
        card_features=features.card_features.unsqueeze(0).to(device),
        player_features=features.player_features.unsqueeze(0).to(device),
        global_features=features.global_features.unsqueeze(0).to(device),
        action_features=features.action_features.unsqueeze(0).to(device),
        action_mask=features.action_mask.unsqueeze(0).to(device),
    )
    return {
        "task": features.task,
        "policy_logits": outputs["policy_logits"].squeeze(0).cpu(),
        "value": outputs["value"].squeeze(0).cpu(),
        "aux": outputs["aux"].squeeze(0).cpu(),
    }


def task_from_record_phase(record: dict) -> str:
    record = normalize_runtime_record(record)
    phase = str(record["obs"]["phase"])
    task = task_from_phase_name(phase)
    if task is None:
        raise KeyError(f"no decision task mapped for phase '{phase}'")
    return task


def select_decision_model(bundle: FourModelBundle, record: dict) -> torch.nn.Module:
    record = normalize_runtime_record(record)
    task = task_from_record_phase(record)
    return bundle.decision_models[task]


@torch.no_grad()
def predict_with_bundle(bundle: FourModelBundle, record: dict) -> dict[str, torch.Tensor]:
    record = normalize_runtime_record(record)
    model = select_decision_model(bundle, record)
    return predict_decision_outputs(
        record,
        model,
        bundle.belief_model,
        device=bundle.device,
    )


@torch.no_grad()
def choose_action_pos_with_bundle(bundle: FourModelBundle, payload: dict) -> tuple[int, float]:
    outputs = predict_with_bundle(bundle, payload)
    logits = outputs["policy_logits"]
    probs = torch.softmax(logits, dim=-1)
    pos = int(torch.argmax(probs).item())
    conf = float(probs[pos].item())
    return pos, conf
