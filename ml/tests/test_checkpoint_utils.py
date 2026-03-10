from pathlib import Path

import pytest
import torch

from ml.checkpoint_utils import checkpoint_metadata, load_checkpoint_payload, load_model_checkpoint


def test_load_checkpoint_payload_supports_metadata(tmp_path: Path):
    path = tmp_path / "with_meta.pt"
    torch.save({"state_dict": {"weight": torch.ones(1)}, "metadata": {"task": "demo"}}, path)
    payload = load_checkpoint_payload(path)
    assert payload["metadata"]["task"] == "demo"


def test_checkpoint_metadata_reads_side_data(tmp_path: Path):
    path = tmp_path / "model.pt"
    torch.save({"state_dict": {"weight": torch.ones(1)}, "metadata": {"epochs_seen": 3}}, path)
    assert checkpoint_metadata(path)["epochs_seen"] == 3


def test_load_model_checkpoint_rejects_task_mismatch(tmp_path: Path):
    model = torch.nn.Linear(1, 1)
    path = tmp_path / "mismatch.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": {"task": "bidding"}}, path)
    with pytest.raises(ValueError):
        load_model_checkpoint(torch.nn.Linear(1, 1), path, expected_task="playing")
