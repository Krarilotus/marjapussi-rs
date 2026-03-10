from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ml.env import MarjapussiEnv
    from ml.four_model_runtime import FourModelBundle, choose_action_pos_with_bundle, load_four_model_bundle
except ModuleNotFoundError:
    from env import MarjapussiEnv
    from four_model_runtime import FourModelBundle, choose_action_pos_with_bundle, load_four_model_bundle


@dataclass(frozen=True)
class SelfPlaySummary:
    games: int
    records: int
    avg_actions_per_game: float


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


def generate_selfplay_dataset(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    games: int,
    seed_start: int = 1,
    max_steps: int = 300,
) -> SelfPlaySummary:
    bundle = load_four_model_bundle(manifest_path, device="cpu")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    total_actions = 0

    with output.open("w", encoding="utf-8") as handle:
        for game_idx in range(games):
            env = MarjapussiEnv(pov=0, include_labels=True)
            pending: list[dict[str, Any]] = []
            try:
                obs = env.reset(seed=seed_start + game_idx)
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
                            "seed": seed_start + game_idx,
                            "seat": active,
                        }
                    )
                    obs, done, last_info = env.step(int(chosen.get("action_list_idx", action_pos)))
                    steps += 1

                outcome = last_info if done else env.run_to_end("heuristic")
                for record in pending:
                    handle.write(json.dumps(_finalize_record(record, outcome), ensure_ascii=True) + "\n")
                total_records += len(pending)
                total_actions += steps
            finally:
                env.close()

    avg_actions = (total_actions / games) if games > 0 else 0.0
    return SelfPlaySummary(games=games, records=total_records, avg_actions_per_game=avg_actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()
    summary = generate_selfplay_dataset(
        args.manifest,
        args.output,
        games=args.games,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
    )
    print(
        json.dumps(
            {
                "games": summary.games,
                "records": summary.records,
                "avg_actions_per_game": summary.avg_actions_per_game,
            }
        )
    )


if __name__ == "__main__":
    main()
