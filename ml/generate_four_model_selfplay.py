from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ml.env import MarjapussiEnv
    from ml.four_model_phase import (
        phase_matches_generation_target,
        start_trick_for_generation_target,
        task_from_phase_name,
    )
    from ml.four_model_runtime import FourModelBundle, choose_action_pos_with_bundle, load_four_model_bundle
except ModuleNotFoundError:
    from env import MarjapussiEnv
    from four_model_phase import (
        phase_matches_generation_target,
        start_trick_for_generation_target,
        task_from_phase_name,
    )
    from four_model_runtime import FourModelBundle, choose_action_pos_with_bundle, load_four_model_bundle


@dataclass(frozen=True)
class SelfPlaySummary:
    games: int
    records: int
    avg_actions_per_game: float
    task_counts: dict[str, int]
    generated_by_target: dict[str, int]
    skipped_by_target: dict[str, int]


@dataclass(frozen=True)
class SelfPlayMix:
    full_games: int
    bidding_games: int
    passing_games: int


def default_selfplay_mix(total_games: int) -> SelfPlayMix:
    if total_games <= 0:
        return SelfPlayMix(full_games=0, bidding_games=0, passing_games=0)
    if total_games < 12:
        full_games = max(1, total_games // 2)
        remaining = total_games - full_games
        bidding_games = remaining // 2
        passing_games = remaining - bidding_games
        return SelfPlayMix(
            full_games=full_games,
            bidding_games=bidding_games,
            passing_games=passing_games,
        )
    top_up = max(8, total_games // 8)
    bidding_games = min(top_up, max(0, total_games - 2))
    passing_games = min(top_up, max(0, total_games - bidding_games - 1))
    full_games = max(1, total_games - bidding_games - passing_games)
    return SelfPlayMix(
        full_games=full_games,
        bidding_games=bidding_games,
        passing_games=passing_games,
    )


def _strip_obs_for_record(obs: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(obs)
    cleaned.pop("canonical_state", None)
    cleaned.pop("belief_targets", None)
    return cleaned


def _finalize_record(record: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    team_points = outcome.get("team_points", [0, 0])
    rel_pov = int(record["obs"].get("my_role", 0))
    team_idx = 0 if rel_pov % 2 == 0 else 1
    opp_idx = 1 - team_idx
    record["outcome_pts_my_team"] = float(team_points[team_idx])
    record["outcome_pts_opp"] = float(team_points[opp_idx])
    record["no_one_played"] = bool(outcome.get("no_one_played", False))
    record["contract_made"] = outcome.get("contract_made")
    record["highest_bid"] = outcome.get("highest_bid")
    record["playing_party"] = outcome.get("playing_party")
    record["playing_party_tricks"] = outcome.get("playing_party_tricks")
    record["defending_party_tricks"] = outcome.get("defending_party_tricks")
    record["playing_called_trumps"] = outcome.get("playing_called_trumps")
    record["playing_possible_pairs"] = outcome.get("playing_possible_pairs")
    record["playing_questions"] = outcome.get("playing_questions")
    record["defending_questions"] = outcome.get("defending_questions")
    record["first_trick_playing_won"] = outcome.get("first_trick_playing_won")
    return record


def _normalize_phase_to_task(phase_name: str) -> str:
    return task_from_phase_name(str(phase_name))


def _append_finished_records(
    *,
    handle,
    pending: list[dict[str, Any]],
    outcome: dict[str, Any],
    task_counts: dict[str, int],
) -> None:
    for record in pending:
        task = _normalize_phase_to_task(record["obs"].get("phase", ""))
        task_counts[task] += 1
        handle.write(json.dumps(_finalize_record(record, outcome), ensure_ascii=True) + "\n")


def _play_single_game(
    *,
    env: MarjapussiEnv,
    bundle: FourModelBundle,
    seed: int,
    start_trick: int | None,
    generation_target: str,
    max_steps: int,
) -> tuple[bool, int, int, list[dict[str, Any]], dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    obs = env.reset(seed=seed, start_trick=start_trick)
    if not phase_matches_generation_target(str(obs.get("phase", "")), generation_target):
        return False, 0, 0, pending, {}

    done = False
    steps = 0
    last_info: dict[str, Any] = {}
    while not done and steps < max_steps:
        active = int(obs.get("active_player", 0)) % 4
        seat_obs = env.observe_pov(active)
        legal = seat_obs.get("legal_actions", [])
        if not legal:
            break
        action_pos, _conf = choose_action_pos_with_bundle(bundle, seat_obs)
        action_pos = max(0, min(action_pos, len(legal) - 1))
        chosen = legal[action_pos]
        pending.append(
            {
                "obs": _strip_obs_for_record(seat_obs),
                "canonical_state": seat_obs.get("canonical_state"),
                "belief_targets": seat_obs.get("belief_targets"),
                "action_taken": action_pos,
                "pov_player_winrate": 0.5,
                "seed": seed,
                "seat": active,
                "generation_target": generation_target,
            }
        )
        obs, done, last_info = env.step(int(chosen.get("action_list_idx", action_pos)))
        steps += 1

    outcome = last_info if done else env.run_to_end("heuristic")
    return True, len(pending), steps, pending, outcome


def generate_selfplay_dataset(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    games: int | None = None,
    full_games: int | None = None,
    bidding_games: int = 0,
    passing_games: int = 0,
    seed_start: int = 1,
    max_steps: int = 300,
    max_seed_tries_per_target: int = 32,
) -> SelfPlaySummary:
    bundle = load_four_model_bundle(manifest_path, device="cpu")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    total_actions = 0
    task_counts = {"bidding": 0, "passing": 0, "playing": 0}
    generated_by_target = {"full": 0, "bidding": 0, "passing": 0}
    skipped_by_target = {"full": 0, "bidding": 0, "passing": 0}

    if full_games is None:
        if games is None:
            raise ValueError("either games or full_games must be provided")
        if bidding_games or passing_games:
            full_games = max(0, games - bidding_games - passing_games)
        else:
            full_games = games
    total_requested = full_games + bidding_games + passing_games
    seed_cursor = seed_start

    with output.open("w", encoding="utf-8") as handle:
        generation_specs = (
            ("full", full_games),
            ("bidding", bidding_games),
            ("passing", passing_games),
        )
        for generation_target, target_games in generation_specs:
            if target_games <= 0:
                continue
            accepted = 0
            tries = 0
            max_tries = max(target_games, 1) * max_seed_tries_per_target
            while accepted < target_games and tries < max_tries:
                tries += 1
                env = MarjapussiEnv(pov=0, include_labels=True)
                try:
                    ok, record_count, steps, pending, outcome = _play_single_game(
                        env=env,
                        bundle=bundle,
                        seed=seed_cursor,
                        start_trick=start_trick_for_generation_target(generation_target),
                        generation_target=generation_target,
                        max_steps=max_steps,
                    )
                finally:
                    env.close()
                seed_cursor += 1
                if not ok:
                    skipped_by_target[generation_target] += 1
                    continue
                _append_finished_records(
                    handle=handle,
                    pending=pending,
                    outcome=outcome,
                    task_counts=task_counts,
                )
                accepted += 1
                generated_by_target[generation_target] += 1
                total_records += record_count
                total_actions += steps

    total_generated_games = sum(generated_by_target.values())
    avg_actions = (total_actions / total_generated_games) if total_generated_games > 0 else 0.0
    return SelfPlaySummary(
        games=total_generated_games,
        records=total_records,
        avg_actions_per_game=avg_actions,
        task_counts=task_counts,
        generated_by_target=generated_by_target,
        skipped_by_target=skipped_by_target,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--full-games", type=int, default=None)
    ap.add_argument("--bidding-games", type=int, default=0)
    ap.add_argument("--passing-games", type=int, default=0)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-seed-tries-per-target", type=int, default=32)
    args = ap.parse_args()
    cli_full_games = args.full_games
    cli_bidding_games = args.bidding_games
    cli_passing_games = args.passing_games
    if cli_full_games is None and cli_bidding_games == 0 and cli_passing_games == 0:
        mix = default_selfplay_mix(args.games)
        cli_full_games = mix.full_games
        cli_bidding_games = mix.bidding_games
        cli_passing_games = mix.passing_games
    summary = generate_selfplay_dataset(
        args.manifest,
        args.output,
        games=args.games,
        full_games=cli_full_games,
        bidding_games=cli_bidding_games,
        passing_games=cli_passing_games,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        max_seed_tries_per_target=args.max_seed_tries_per_target,
    )
    print(
        json.dumps(
            {
                "games": summary.games,
                "records": summary.records,
                "avg_actions_per_game": summary.avg_actions_per_game,
                "task_counts": summary.task_counts,
                "generated_by_target": summary.generated_by_target,
                "skipped_by_target": summary.skipped_by_target,
            }
        )
    )


if __name__ == "__main__":
    main()
