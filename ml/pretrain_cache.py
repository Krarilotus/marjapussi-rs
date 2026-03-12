from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from ml.decision_state import (
        build_decision_features_from_record,
        build_decision_targets_from_record,
    )
    from ml.neurosymbolic_dataset import (
        build_belief_features,
        build_belief_targets,
        load_canonical_state,
    )
except ModuleNotFoundError:
    from decision_state import (
        build_decision_features_from_record,
        build_decision_targets_from_record,
    )
    from neurosymbolic_dataset import (
        build_belief_features,
        build_belief_targets,
        load_canonical_state,
    )


CACHE_VERSION = "v2"
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "ml" / "data" / "cache"


def _data_signature(data_path: str | Path, suffix: str) -> str:
    path = Path(data_path)
    stat = path.stat()
    raw = f"{CACHE_VERSION}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{suffix}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(data_path: str | Path, kind: str, suffix: str) -> Path:
    path = Path(data_path)
    stem = path.stem
    sig = _data_signature(path, suffix)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{stem}.{kind}.{sig}.pt"


def _load_cached_payload(path: Path) -> dict | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _pad_vector(vec: torch.Tensor | None, size: int) -> torch.Tensor:
    if vec is None:
        return torch.zeros(size, dtype=torch.float32)
    vec = vec.to(dtype=torch.float32)
    if vec.shape[0] > size:
        return vec[:size]
    if vec.shape[0] < size:
        return F.pad(vec, (0, size - vec.shape[0]))
    return vec


def load_or_build_decision_cache(data_path: str | Path, task: str) -> tuple[Path, dict]:
    cache_path = _cache_path(data_path, "decision", task)
    cached = _load_cached_payload(cache_path)
    if cached is not None:
        return cache_path, cached

    card_features: list[torch.Tensor] = []
    player_features: list[torch.Tensor] = []
    global_features: list[torch.Tensor] = []
    action_features: list[torch.Tensor] = []
    action_masks: list[torch.Tensor] = []
    policy_targets: list[int] = []
    value_targets: list[float] = []
    aux_targets: list[torch.Tensor] = []
    sample_weights: list[float] = []
    teacher_policies: list[torch.Tensor | None] = []
    max_actions = 0

    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                features = build_decision_features_from_record(record, use_teacher_belief=True)
            except Exception:
                continue
            if features.task != task:
                continue
            try:
                targets = build_decision_targets_from_record(record)
            except Exception:
                continue
            card_features.append(features.card_features)
            player_features.append(features.player_features)
            global_features.append(features.global_features)
            action_features.append(features.action_features)
            action_masks.append(features.action_mask)
            policy_targets.append(int(targets.policy_idx))
            value_targets.append(float(targets.value_target))
            aux_targets.append(targets.aux_targets)
            sample_weights.append(float(targets.sample_weight))
            teacher_policies.append(targets.teacher_policy)
            max_actions = max(max_actions, int(features.action_features.shape[0]))

    if not card_features:
        raise RuntimeError(f"no decision samples found for task '{task}' in {data_path}")

    padded_action_features = []
    padded_action_masks = []
    padded_teacher_policies = []
    for feats, mask, teacher_policy in zip(action_features, action_masks, teacher_policies):
        padded_action_features.append(
            F.pad(feats, (0, 0, 0, max_actions - feats.shape[0]))
            if feats.shape[0] < max_actions
            else feats
        )
        padded_action_masks.append(
            F.pad(mask, (0, max_actions - mask.shape[0]), value=True)
            if mask.shape[0] < max_actions
            else mask
        )
        padded_teacher_policies.append(_pad_vector(teacher_policy, max_actions))

    payload = {
        "format": "packed_tensors",
        "sample_count": len(card_features),
        "max_actions": max_actions,
        "card_features": torch.stack(card_features),
        "player_features": torch.stack(player_features),
        "global_features": torch.stack(global_features),
        "action_features": torch.stack(padded_action_features),
        "action_mask": torch.stack(padded_action_masks),
        "policy_targets": torch.tensor(policy_targets, dtype=torch.long),
        "value_targets": torch.tensor(value_targets, dtype=torch.float32),
        "aux_targets": torch.stack(aux_targets),
        "sample_weights": torch.tensor(sample_weights, dtype=torch.float32),
        "teacher_policy": torch.stack(padded_teacher_policies),
    }

    torch.save(payload, cache_path)
    return cache_path, payload


def load_or_build_belief_cache(data_path: str | Path) -> tuple[Path, dict]:
    cache_path = _cache_path(data_path, "belief", "belief")
    cached = _load_cached_payload(cache_path)
    if cached is not None:
        return cache_path, cached

    card_features: list[torch.Tensor] = []
    player_features: list[torch.Tensor] = []
    global_features: list[torch.Tensor] = []
    card_targets: list[torch.Tensor] = []
    hidden_masks: list[torch.Tensor] = []
    candidate_masks: list[torch.Tensor] = []
    void_targets: list[torch.Tensor] = []
    half_targets: list[torch.Tensor] = []
    pair_targets: list[torch.Tensor] = []

    with Path(data_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                state = load_canonical_state(record)
                feats = build_belief_features(state)
                targets = build_belief_targets(state)
            except Exception:
                continue
            card_features.append(feats.card_features)
            player_features.append(feats.player_features[1:])
            global_features.append(feats.global_features)
            card_targets.append(torch.tensor(targets.card_owner_targets, dtype=torch.long))
            hidden_masks.append(torch.tensor(targets.hidden_card_mask, dtype=torch.bool))
            candidate_masks.append(torch.tensor(targets.owner_candidate_mask, dtype=torch.bool))
            void_targets.append(torch.tensor(targets.player_void_targets[1:], dtype=torch.float32))
            half_targets.append(torch.tensor(targets.player_has_half_targets[1:], dtype=torch.float32))
            pair_targets.append(torch.tensor(targets.player_has_pair_targets[1:], dtype=torch.float32))

    if not card_features:
        raise RuntimeError(f"no belief samples found in {data_path}")

    payload = {
        "format": "packed_tensors",
        "sample_count": len(card_features),
        "card_features": torch.stack(card_features),
        "player_features": torch.stack(player_features),
        "global_features": torch.stack(global_features),
        "card_targets": torch.stack(card_targets),
        "hidden_mask": torch.stack(hidden_masks),
        "candidate_mask": torch.stack(candidate_masks),
        "void_targets": torch.stack(void_targets),
        "half_targets": torch.stack(half_targets),
        "pair_targets": torch.stack(pair_targets),
    }

    torch.save(payload, cache_path)
    return cache_path, payload
