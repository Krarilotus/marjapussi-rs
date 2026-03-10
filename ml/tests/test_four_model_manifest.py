import json
import os
from pathlib import Path

from ml.four_model_manifest import (
    BeliefStageManifest,
    DecisionStageManifest,
    FourModelOutputs,
    load_four_model_manifest,
    validate_four_model_manifest,
    write_four_model_manifest,
)


def test_load_four_model_manifest_resolves_outputs(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    for name in ("bidding_latest.pt", "passing_latest.pt", "playing_latest.pt", "belief_latest.pt"):
        (ckpt_dir / name).write_bytes(b"stub")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_path": "ml/data/human_dataset.ndjson",
                "device": "cpu",
                "workers": 2,
                "decision_stages": [
                    {"task": "bidding", "epochs": 1, "batch": 8, "target_acc": 0.5},
                    {"task": "passing", "epochs": 1, "batch": 8, "target_acc": 0.5},
                    {"task": "playing", "epochs": 1, "batch": 8, "target_acc": 0.5},
                ],
                "belief_stage": {
                    "epochs": 1,
                    "batch": 8,
                    "target_hidden_acc": 0.7,
                },
                "outputs": {
                    "bidding": "ckpts/bidding_latest.pt",
                    "passing": "ckpts/passing_latest.pt",
                    "playing": "ckpts/playing_latest.pt",
                    "belief": "ckpts/belief_latest.pt",
                },
                "metadata": {"training_stage": "human_pretrain"},
            }
        ),
        encoding="utf-8",
    )
    manifest = load_four_model_manifest(manifest_path)
    assert manifest.outputs.bidding.exists()
    assert manifest.outputs.belief.exists()
    assert manifest.metadata["training_stage"] == "human_pretrain"


def test_validate_four_model_manifest_reports_missing_outputs(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_path": "ml/data/human_dataset.ndjson",
                "device": "cpu",
                "workers": 2,
                "decision_stages": [{"task": "bidding", "epochs": 1, "batch": 8, "target_acc": 0.5}],
                "belief_stage": {
                    "epochs": 1,
                    "batch": 8,
                    "target_hidden_acc": 0.7,
                },
                "outputs": {
                    "bidding": "missing/bidding_latest.pt",
                    "passing": "missing/passing_latest.pt",
                    "playing": "missing/playing_latest.pt",
                    "belief": "missing/belief_latest.pt",
                },
            }
        ),
        encoding="utf-8",
    )
    _, errors = validate_four_model_manifest(manifest_path)
    assert any("missing decision stage 'passing'" in error for error in errors)
    assert any("missing output checkpoint" in error for error in errors)


def test_write_four_model_manifest_normalizes_output_paths_relative_to_manifest(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    manifest_dir = repo_root / "ml" / "checkpoints" / "bundle"
    out_dir = manifest_dir / "bidding"
    out_dir.mkdir(parents=True)
    passing_dir = manifest_dir / "passing"
    passing_dir.mkdir(parents=True)
    playing_dir = manifest_dir / "playing"
    playing_dir.mkdir(parents=True)
    belief_dir = manifest_dir / "belief"
    belief_dir.mkdir(parents=True)
    for path in (
        out_dir / "bidding_latest.pt",
        passing_dir / "passing_latest.pt",
        playing_dir / "playing_latest.pt",
        belief_dir / "belief_latest.pt",
    ):
        path.write_bytes(b"stub")

    monkeypatch.chdir(repo_root)
    manifest_path = write_four_model_manifest(
        manifest_dir / "manifest.json",
        data_path="ml/data/human_dataset.ndjson",
        device="cpu",
        workers=0,
        decision_stages=(
            DecisionStageManifest(task="bidding", epochs=1, batch=8, target_acc=0.5),
            DecisionStageManifest(task="passing", epochs=1, batch=8, target_acc=0.5),
            DecisionStageManifest(task="playing", epochs=1, batch=8, target_acc=0.5),
        ),
        belief_stage=BeliefStageManifest(epochs=1, batch=8, target_hidden_acc=0.7),
        outputs=FourModelOutputs(
            bidding=Path("ml/checkpoints/bundle/bidding/bidding_latest.pt"),
            passing=Path("ml/checkpoints/bundle/passing/passing_latest.pt"),
            playing=Path("ml/checkpoints/bundle/playing/playing_latest.pt"),
            belief=Path("ml/checkpoints/bundle/belief/belief_latest.pt"),
        ),
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["outputs"]["bidding"] == os.path.join("bidding", "bidding_latest.pt")
    manifest = load_four_model_manifest(manifest_path)
    assert manifest.outputs.bidding.exists()
