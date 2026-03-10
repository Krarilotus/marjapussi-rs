from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint_payload(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        return {"state_dict": raw["state_dict"], "metadata": dict(raw.get("metadata", {}))}
    if isinstance(raw, dict):
        return {"state_dict": raw, "metadata": {}}
    raise TypeError(f"unsupported checkpoint payload type: {type(raw)!r}")


def checkpoint_metadata(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    return dict(load_checkpoint_payload(path, device=device).get("metadata", {}))


def load_model_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    device: str = "cpu",
    expected_task: str | None = None,
) -> dict[str, Any]:
    payload = load_checkpoint_payload(path, device=device)
    metadata = payload.get("metadata", {})
    if expected_task is not None:
        actual_task = metadata.get("task")
        if actual_task is not None and str(actual_task) != expected_task:
            raise ValueError(
                f"checkpoint task mismatch: expected '{expected_task}', got '{actual_task}'"
            )
    model.load_state_dict(payload["state_dict"])
    return payload
