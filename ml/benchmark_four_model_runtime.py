from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

try:
    from ml.four_model_runtime import build_runtime_decision_features, load_four_model_bundle, select_decision_model
except ModuleNotFoundError:
    from four_model_runtime import build_runtime_decision_features, load_four_model_bundle, select_decision_model


@dataclass(frozen=True)
class RuntimeTimingSummary:
    total_examples: int
    belief_ms_avg: float
    decision_ms_avg: float
    end_to_end_ms_avg: float
    end_to_end_ms_p95: float


def summarize_timings(belief_ms: list[float], decision_ms: list[float], total_ms: list[float]) -> RuntimeTimingSummary:
    if not total_ms:
        return RuntimeTimingSummary(0, 0.0, 0.0, 0.0, 0.0)
    ordered = sorted(total_ms)
    p95_idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return RuntimeTimingSummary(
        total_examples=len(total_ms),
        belief_ms_avg=sum(belief_ms) / len(belief_ms),
        decision_ms_avg=sum(decision_ms) / len(decision_ms),
        end_to_end_ms_avg=sum(total_ms) / len(total_ms),
        end_to_end_ms_p95=ordered[p95_idx],
    )


def iter_records(path: str | Path, max_records: int = 0) -> Iterable[dict]:
    seen = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            seen += 1
            if max_records > 0 and seen >= max_records:
                break


@torch.no_grad()
def benchmark_runtime(
    *,
    manifest: str | Path,
    data: str | Path,
    max_records: int = 128,
    device: str = "cpu",
) -> RuntimeTimingSummary:
    bundle = load_four_model_bundle(manifest, device=device)
    belief_ms: list[float] = []
    decision_ms: list[float] = []
    total_ms: list[float] = []

    for record in iter_records(data, max_records=max_records):
        t0 = time.perf_counter()
        features = build_runtime_decision_features(record, bundle.belief_model, device=bundle.device)
        t1 = time.perf_counter()
        model = select_decision_model(bundle, record)
        _ = model(
            card_features=features.card_features.unsqueeze(0).to(bundle.device),
            player_features=features.player_features.unsqueeze(0).to(bundle.device),
            global_features=features.global_features.unsqueeze(0).to(bundle.device),
            action_features=features.action_features.unsqueeze(0).to(bundle.device),
            action_mask=features.action_mask.unsqueeze(0).to(bundle.device),
        )
        if bundle.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        belief_ms.append((t1 - t0) * 1000.0)
        decision_ms.append((t2 - t1) * 1000.0)
        total_ms.append((t2 - t0) * 1000.0)

    return summarize_timings(belief_ms, decision_ms, total_ms)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = benchmark_runtime(
        manifest=args.manifest,
        data=args.data,
        max_records=args.max_records,
        device=args.device,
    )
    print(f"Examples:       {summary.total_examples}")
    print(f"Belief avg ms:  {summary.belief_ms_avg:.3f}")
    print(f"Decision avg ms:{summary.decision_ms_avg:.3f}")
    print(f"E2E avg ms:     {summary.end_to_end_ms_avg:.3f}")
    print(f"E2E p95 ms:     {summary.end_to_end_ms_p95:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
