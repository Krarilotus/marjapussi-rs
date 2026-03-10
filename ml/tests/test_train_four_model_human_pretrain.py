import json
from pathlib import Path

import ml.train_four_model_human_pretrain as mod


def test_run_human_pretraining_writes_manifest(tmp_path: Path, monkeypatch):
    calls = []

    def fake_train_decision(**kwargs):
        calls.append(("decision", kwargs["task"], Path(kwargs["checkpoints_dir"])))
        Path(kwargs["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["checkpoints_dir"]) / f"{kwargs['task']}_latest.pt").write_text("ok", encoding="utf-8")

    def fake_train_belief(**kwargs):
        calls.append(("belief", None, Path(kwargs["checkpoints_dir"])))
        Path(kwargs["checkpoints_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["checkpoints_dir"]) / "belief_latest.pt").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(mod, "train_decision", fake_train_decision)
    monkeypatch.setattr(mod, "train_belief", fake_train_belief)

    manifest = mod.run_human_pretraining(
        data_path="ml/data/human_dataset.ndjson",
        checkpoints_dir=tmp_path / "pretrain",
        device="cpu",
        workers=0,
        max_steps_decision=1,
        max_steps_belief=1,
        no_amp=True,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [entry[1] for entry in calls[:3]] == ["bidding", "passing", "playing"]
    assert calls[3][0] == "belief"
    assert payload["outputs"]["belief"].endswith("belief\\belief_latest.pt") or payload["outputs"]["belief"].endswith("belief/belief_latest.pt")
