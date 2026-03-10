from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from ml.behavior_score import (
        BehaviorEvalSummary,
        BehaviorScoreBreakdown,
        compute_behavior_score,
    )
    from ml.eval_fixed_deals import evaluate_checkpoint_suite
except ModuleNotFoundError:
    from behavior_score import BehaviorEvalSummary, BehaviorScoreBreakdown, compute_behavior_score
    from eval_fixed_deals import evaluate_checkpoint_suite


@dataclass(frozen=True)
class GovernanceEvalResult:
    manifest_path: str
    suite_path: str
    summary: dict[str, float | int | str]
    score: dict[str, float]


def _behavior_summary_from_eval(summary: dict) -> BehaviorEvalSummary:
    return BehaviorEvalSummary(
        point_diff=float(summary.get("avg_playing_margin_points", 0.0)),
        pass_game_rate=float(summary.get("pass_game_rate", 0.0)),
        avg_bid=float(summary.get("avg_highest_bid", 0.0)),
        contract_made_rate=float(summary.get("contract_made_rate", 0.0)),
        trump_call_rate=float(summary.get("pair_call_rate", 0.0)),
    )


def evaluate_manifest_behavior(
    manifest_path: str | Path,
    *,
    suite_path: str | Path,
    device: str = "cpu",
    max_cases: int = 0,
    strict_param_budget: int = 28_000_000,
    output_path: str | Path | None = None,
) -> GovernanceEvalResult:
    summary = evaluate_checkpoint_suite(
        checkpoint_path=manifest_path,
        suite_path=suite_path,
        device=device,
        strict_param_budget=strict_param_budget,
        max_cases=max_cases,
        echo=False,
        ansi=False,
        output_path=output_path,
    )
    score = compute_behavior_score(_behavior_summary_from_eval(summary))
    return GovernanceEvalResult(
        manifest_path=str(Path(manifest_path)),
        suite_path=str(Path(suite_path)),
        summary={
            key: value
            for key, value in summary.items()
            if key not in {"outcomes", "log_lines"}
        },
        score=asdict(score),
    )


def maybe_promote_best_manifest(
    *,
    result: GovernanceEvalResult,
    governance_dir: str | Path,
    best_name: str = "best_fixed_suite_manifest.json",
) -> tuple[bool, Path, dict]:
    root = Path(governance_dir)
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / "best_fixed_suite_score.json"
    best_manifest_path = root / best_name
    current_score = float(result.score["total"])
    previous_score = float("-inf")
    if score_path.exists():
        try:
            previous_score = float(json.loads(score_path.read_text(encoding="utf-8")).get("total", float("-inf")))
        except Exception:
            previous_score = float("-inf")

    promoted = current_score > previous_score
    payload = {
        "manifest_path": result.manifest_path,
        "suite_path": result.suite_path,
        "summary": result.summary,
        "score": result.score,
        "total": current_score,
        "promoted": promoted,
    }
    eval_path = root / "last_fixed_suite_eval.json"
    eval_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if promoted:
        shutil.copy2(result.manifest_path, best_manifest_path)
        score_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return promoted, best_manifest_path, payload
