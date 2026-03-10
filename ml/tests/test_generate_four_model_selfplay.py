import json
from pathlib import Path

from ml import generate_four_model_selfplay as mod


class FakeEnv:
    def __init__(self, pov=0, include_labels=False):
        self.done = False
        self._step = 0

    def reset(self, seed=None):
        return {
            "active_player": 0,
            "phase": "Trick",
            "legal_actions": [{"action_list_idx": 0, "action_token": 40, "card_idx": 0}],
            "canonical_state": {"schema_version": 1},
            "belief_targets": {"schema_version": 1},
            "my_role": 0,
        }

    def observe_pov(self, pov):
        return {
            "active_player": 0,
            "phase": "Trick",
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
