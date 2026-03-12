import json
from pathlib import Path

from ml import generate_four_model_selfplay as mod


class FakeEnv:
    def __init__(self, pov=0, include_labels=False):
        self.done = False
        self._step = 0
        self.start_trick = None

    def reset(self, seed=None, start_trick=None):
        self.start_trick = start_trick
        phase = "Trick"
        if start_trick == -1:
            phase = "Bidding"
        elif start_trick == 0:
            phase = "PassingForth"
        return {
            "active_player": 0,
            "phase": phase,
            "legal_actions": [{"action_list_idx": 0, "action_token": 40, "card_idx": 0}],
            "canonical_state": {"schema_version": 1},
            "belief_targets": {"schema_version": 1},
            "my_role": 0,
        }

    def observe_pov(self, pov):
        phase = "Trick"
        if self.start_trick == -1:
            phase = "Bidding"
        elif self.start_trick == 0:
            phase = "PassingForth"
        return {
            "active_player": 0,
            "phase": phase,
            "legal_actions": [{"action_list_idx": 0, "action_token": 40, "card_idx": 0}],
            "canonical_state": {"schema_version": 1},
            "belief_targets": {"schema_version": 1},
            "my_role": 0,
        }

    def step(self, action_idx):
        self.done = True
        return (
            {
                "active_player": 0,
                "phase": "Trick",
                "legal_actions": [],
                "canonical_state": {"schema_version": 1},
                "belief_targets": {"schema_version": 1},
                "my_role": 0,
            },
            True,
            {"team_points": [120, 60], "no_one_played": False, "contract_made": True},
        )

    def run_to_end(self, policy):
        return {"team_points": [120, 60], "no_one_played": False, "contract_made": True}

    def close(self):
        return None


def test_generate_selfplay_dataset_writes_records(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mod, "MarjapussiEnv", FakeEnv)
    monkeypatch.setattr(mod, "load_four_model_bundle", lambda manifest_path, device="cpu": object())
    monkeypatch.setattr(mod, "choose_action_pos_with_bundle", lambda bundle, payload: (0, 1.0))

    out = tmp_path / "selfplay.ndjson"
    summary = mod.generate_selfplay_dataset("manifest.json", out, games=2)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert summary.games == 2
    assert summary.records == 2
    assert len(lines) == 2
    assert lines[0]["outcome_pts_my_team"] == 120.0


def test_default_selfplay_mix_reserves_phase_topups():
    mix = mod.default_selfplay_mix(64)
    assert mix.full_games == 48
    assert mix.bidding_games == 8
    assert mix.passing_games == 8


def test_generate_selfplay_dataset_balances_targets(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mod, "MarjapussiEnv", FakeEnv)
    monkeypatch.setattr(mod, "load_four_model_bundle", lambda manifest_path, device="cpu": object())
    monkeypatch.setattr(mod, "choose_action_pos_with_bundle", lambda bundle, payload: (0, 1.0))

    out = tmp_path / "selfplay_mix.ndjson"
    summary = mod.generate_selfplay_dataset(
        "manifest.json",
        out,
        games=12,
        full_games=4,
        bidding_games=4,
        passing_games=4,
    )
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    phases = [line["obs"]["phase"] for line in lines]
    assert summary.games == 12
    assert summary.generated_by_target["passing"] == 4
    assert "PassingForth" in phases
    assert summary.task_counts["passing"] >= 4


class NoPassingEnv(FakeEnv):
    def reset(self, seed=None, start_trick=None):
        payload = super().reset(seed=seed, start_trick=start_trick)
        if start_trick == 0:
            payload["phase"] = "Trick"
        return payload


def test_generate_selfplay_dataset_skips_unreachable_passing_targets(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(mod, "MarjapussiEnv", NoPassingEnv)
    monkeypatch.setattr(mod, "load_four_model_bundle", lambda manifest_path, device="cpu": object())
    monkeypatch.setattr(mod, "choose_action_pos_with_bundle", lambda bundle, payload: (0, 1.0))

    out = tmp_path / "selfplay_skip.ndjson"
    summary = mod.generate_selfplay_dataset(
        "manifest.json",
        out,
        games=8,
        full_games=4,
        passing_games=4,
        max_seed_tries_per_target=2,
    )
    assert summary.generated_by_target["passing"] == 0
    assert summary.skipped_by_target["passing"] > 0
