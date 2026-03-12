import json
from pathlib import Path

from ml.four_model_governance import GovernanceEvalResult, maybe_promote_best_manifest
from ml.four_model_manifest import (
    BeliefStageManifest,
    DecisionStageManifest,
    FourModelOutputs,
    load_four_model_manifest,
    write_four_model_manifest,
)


def test_maybe_promote_best_manifest_promotes_higher_score(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"ok": True}), encoding="utf-8")
    result = GovernanceEvalResult(
        manifest_path=str(manifest),
        suite_path="ml/eval/fixed_deals_100.json",
        summary={"games": 8, "pass_game_rate": 0.0},
        score={"total": 12.5, "point_term": 1.0, "pass_penalty": 0.0, "minimal_bid_penalty": 0.0, "contract_bonus": 10.0, "trump_bonus": 1.5},
    )
    promoted, best_manifest, payload = maybe_promote_best_manifest(result=result, governance_dir=tmp_path / "gov")
    assert promoted is True
    assert best_manifest.exists()
    assert payload["total"] == 12.5


def test_maybe_promote_best_manifest_keeps_previous_better_score(tmp_path: Path):
    manifest_a = tmp_path / "a.json"
    manifest_b = tmp_path / "b.json"
    manifest_a.write_text(json.dumps({"a": True}), encoding="utf-8")
    manifest_b.write_text(json.dumps({"b": True}), encoding="utf-8")
    gov = tmp_path / "gov"
    first = GovernanceEvalResult(
        manifest_path=str(manifest_a),
        suite_path="suite.json",
        summary={},
        score={"total": 10.0, "point_term": 10.0, "pass_penalty": 0.0, "minimal_bid_penalty": 0.0, "contract_bonus": 0.0, "trump_bonus": 0.0},
    )
    second = GovernanceEvalResult(
        manifest_path=str(manifest_b),
        suite_path="suite.json",
        summary={},
        score={"total": 5.0, "point_term": 5.0, "pass_penalty": 0.0, "minimal_bid_penalty": 0.0, "contract_bonus": 0.0, "trump_bonus": 0.0},
    )
    maybe_promote_best_manifest(result=first, governance_dir=gov)
    promoted, best_manifest, payload = maybe_promote_best_manifest(result=second, governance_dir=gov)
    assert promoted is False
    assert json.loads(best_manifest.read_text(encoding="utf-8")) == {"a": True}
    assert payload["total"] == 5.0


def test_maybe_promote_best_manifest_rewrites_four_model_paths(tmp_path: Path):
    ckpt_root = tmp_path / "run" / "joint"
    (ckpt_root / "bidding").mkdir(parents=True)
    (ckpt_root / "passing").mkdir(parents=True)
    (ckpt_root / "playing").mkdir(parents=True)
    (ckpt_root / "belief").mkdir(parents=True)
    for rel in (
        ckpt_root / "bidding" / "bidding_latest.pt",
        ckpt_root / "passing" / "passing_latest.pt",
        ckpt_root / "playing" / "playing_latest.pt",
        ckpt_root / "belief" / "belief_latest.pt",
    ):
        rel.write_text("x", encoding="utf-8")

    manifest = write_four_model_manifest(
        ckpt_root / "joint_manifest.json",
        data_path="sim.ndjson",
        device="cpu",
        workers=0,
        decision_stages=(
            DecisionStageManifest(task="bidding", epochs=1, batch=8, target_acc=0.1),
            DecisionStageManifest(task="passing", epochs=1, batch=8, target_acc=0.1),
            DecisionStageManifest(task="playing", epochs=1, batch=8, target_acc=0.1),
        ),
        belief_stage=BeliefStageManifest(epochs=1, batch=8, target_hidden_acc=0.1),
        outputs=FourModelOutputs(
            bidding=ckpt_root / "bidding" / "bidding_latest.pt",
            passing=ckpt_root / "passing" / "passing_latest.pt",
            playing=ckpt_root / "playing" / "playing_latest.pt",
            belief=ckpt_root / "belief" / "belief_latest.pt",
        ),
        metadata={"training_stage": "joint_simulated"},
    )
    result = GovernanceEvalResult(
        manifest_path=str(manifest),
        suite_path="suite.json",
        summary={"games": 8},
        score={"total": 9.0, "point_term": 9.0, "pass_penalty": 0.0, "minimal_bid_penalty": 0.0, "contract_bonus": 0.0, "trump_bonus": 0.0},
    )
    promoted, best_manifest, _ = maybe_promote_best_manifest(result=result, governance_dir=tmp_path / "gov")
    assert promoted is True
    loaded = load_four_model_manifest(best_manifest)
    assert loaded.outputs.bidding.exists()
    assert loaded.outputs.passing.exists()
    assert loaded.outputs.playing.exists()
    assert loaded.outputs.belief.exists()
    assert "best_fixed_suite_snapshot" in str(loaded.outputs.bidding)
