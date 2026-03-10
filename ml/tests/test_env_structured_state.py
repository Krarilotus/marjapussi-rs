from ml.env import _merge_response_state


def test_merge_response_state_preserves_obs_and_structured_payloads():
    obs = {"phase": "Trick", "legal_actions": []}
    resp = {
        "canonical_state": {"schema_version": 1},
        "belief_targets": {"schema_version": 1},
    }
    merged = _merge_response_state(obs, resp)
    assert merged["phase"] == "Trick"
    assert merged["canonical_state"]["schema_version"] == 1
    assert merged["belief_targets"]["schema_version"] == 1
