from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from ml.generate_four_model_selfplay import (
        SelfPlaySummary,
        default_selfplay_mix,
        generate_selfplay_dataset,
    )
    from ml.train_four_model_joint import run_joint_training
except ModuleNotFoundError:
    from generate_four_model_selfplay import SelfPlaySummary, default_selfplay_mix, generate_selfplay_dataset
    from train_four_model_joint import run_joint_training


@dataclass(frozen=True)
class EndToEndCycleSummary:
    cycle_index: int
    manifest_path: str
    sim_data_path: str
    selfplay: dict[str, float | int]


def run_end_to_end_training(
    base_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    cycles: int = 4,
    games_per_cycle: int = 64,
    seed_start: int = 1,
    device: str | None = None,
    workers: int | None = None,
    belief_epochs_per_update: int = 1,
    decision_epochs_per_update: int = 1,
    belief_max_steps: int = 0,
    decision_max_steps: int = 0,
    no_amp: bool = False,
    fixed_suite_path: str | None = None,
    fixed_suite_max_cases: int = 0,
    strict_param_budget: int = 28_000_000,
) -> Path:
    root = Path(output_dir)
    data_dir = root / "data"
    ckpt_dir = root / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_manifest = Path(base_manifest_path)
    cycle_summaries: list[EndToEndCycleSummary] = []

    for cycle_idx in range(cycles):
        sim_data_path = data_dir / f"cycle_{cycle_idx + 1:03d}.ndjson"
        mix = default_selfplay_mix(games_per_cycle)
        selfplay_summary = generate_selfplay_dataset(
            current_manifest,
            sim_data_path,
            games=games_per_cycle,
            full_games=mix.full_games,
            bidding_games=mix.bidding_games,
            passing_games=mix.passing_games,
            seed_start=seed_start + cycle_idx * games_per_cycle,
        )
        current_manifest = run_joint_training(
            base_manifest_path=current_manifest,
            sim_data_path=str(sim_data_path),
            checkpoints_dir=ckpt_dir,
            cycles=1,
            device=device,
            workers=workers,
            belief_epochs_per_update=belief_epochs_per_update,
            decision_epochs_per_update=decision_epochs_per_update,
            belief_max_steps=belief_max_steps,
            decision_max_steps=decision_max_steps,
            no_amp=no_amp,
            fixed_suite_path=fixed_suite_path,
            fixed_suite_max_cases=fixed_suite_max_cases,
            strict_param_budget=strict_param_budget,
        )
        cycle_summaries.append(
            EndToEndCycleSummary(
                cycle_index=cycle_idx + 1,
                manifest_path=str(current_manifest),
                sim_data_path=str(sim_data_path),
                selfplay=asdict(selfplay_summary),
            )
        )
        (root / "end_to_end_summary.json").write_text(
            json.dumps({"cycles": [asdict(entry) for entry in cycle_summaries]}, indent=2),
            encoding="utf-8",
        )

    return current_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--output-dir", default="ml/runs/four_model_endtoend")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--games-per-cycle", type=int, default=64)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--belief-epochs-per-update", type=int, default=1)
    ap.add_argument("--decision-epochs-per-update", type=int, default=1)
    ap.add_argument("--belief-max-steps", type=int, default=0)
    ap.add_argument("--decision-max-steps", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--fixed-suite", default=None)
    ap.add_argument("--fixed-suite-max-cases", type=int, default=0)
    ap.add_argument("--strict-param-budget", type=int, default=28_000_000)
    args = ap.parse_args()

    manifest_path = run_end_to_end_training(
        base_manifest_path=args.base_manifest,
        output_dir=args.output_dir,
        cycles=args.cycles,
        games_per_cycle=args.games_per_cycle,
        seed_start=args.seed_start,
        device=args.device,
        workers=args.workers,
        belief_epochs_per_update=args.belief_epochs_per_update,
        decision_epochs_per_update=args.decision_epochs_per_update,
        belief_max_steps=args.belief_max_steps,
        decision_max_steps=args.decision_max_steps,
        no_amp=args.no_amp,
        fixed_suite_path=args.fixed_suite,
        fixed_suite_max_cases=args.fixed_suite_max_cases,
        strict_param_budget=args.strict_param_budget,
    )
    print(f"Wrote end-to-end manifest: {manifest_path}")


if __name__ == "__main__":
    main()
