from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

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


CACHE_VERSION = "v1"
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


def load_or_build_decision_cache(data_path: str | Path, task: str) -> tuple[Path, list[dict]]:
    cache_path = _cache_path(data_path, "decision", task)
    cached = _load_cached_payload(cache_path)
    if cached is not None:
        return cache_path, list(cached["samples"])

    samples: list[dict] = []
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
            teacher_policy = targets.teacher_policy
            if teacher_policy is None:
                teacher_policy = torch.empty(0, dtype=torch.float32)
            samples.append(
                {
                    "card_features": features.card_features,
                    "player_features": features.player_features,
                    "global_features": features.global_features,
                    "action_features": features.action_features,
                    "action_mask": features.action_mask,
                    "policy_target": int(targets.policy_idx),
                    "value_target": float(targets.value_target),
                    "aux_targets": targets.aux_targets,
                    "sample_weight": float(targets.sample_weight),
                    "teacher_policy": teacher_policy,
                }
            )

    torch.save({"samples": samples}, cache_path)
    return cache_path, samples


def load_or_build_belief_cache(data_path: str | Path) -> tuple[Path, list[dict]]:
    cache_path = _cache_path(data_path, "belief", "belief")
    cached = _load_cached_payload(cache_path)
    if cached is not None:
        return cache_path, list(cached["samples"])

    samples: list[dict] = []
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
            samples.append(
                {
                    "card_features": feats.card_features,
                    "player_features": feats.player_features[1:],
                    "global_features": feats.global_features,
                    "card_targets": torch.tensor(targets.card_owner_targets, dtype=torch.long),
                    "hidden_mask": torch.tensor(targets.hidden_card_mask, dtype=torch.bool),
                    "candidate_mask": torch.tensor(targets.owner_candidate_mask, dtype=torch.bool),
                    "void_targets": torch.tensor(targets.player_void_targets[1:], dtype=torch.float32),
                    "half_targets": torch.tensor(targets.player_has_half_targets[1:], dtype=torch.float32),
                    "pair_targets": torch.tensor(targets.player_has_pair_targets[1:], dtype=torch.float32),
                }
            )

    torch.save({"samples": samples}, cache_path)
    return cache_path, samples
