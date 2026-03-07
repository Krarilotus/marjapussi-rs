import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import ACTION_FEAT_DIM
from train.pool import BatchInferenceServer


class _FakePolicy(torch.nn.Module):
    def forward(self, batch):
        bsz = batch["action_feats"].shape[0]
        device = batch["action_feats"].device
        logits = torch.tensor([[0.0, 10.0]], dtype=torch.float32, device=device).expand(bsz, -1).clone()
        card_logits = torch.zeros((bsz, 3, 36), dtype=torch.float32, device=device)
        pts_pred = torch.zeros((bsz, 2), dtype=torch.float32, device=device)
        value_pred = torch.zeros((bsz,), dtype=torch.float32, device=device)
        return logits, card_logits, pts_pred, value_pred


class BatchInferenceServerMaskTest(unittest.TestCase):
    def test_masked_actions_are_never_selected(self):
        model = _FakePolicy()
        server = BatchInferenceServer(model, "cpu", max_batch=1, greedy=True)
        try:
            obs = {
                "obs_a": {"dummy": torch.zeros((1, 1), dtype=torch.float32)},
                "token_ids": torch.zeros((1, 1), dtype=torch.long),
                "token_mask": torch.zeros((1, 1), dtype=torch.bool),
                "action_feats": torch.zeros((1, 2, ACTION_FEAT_DIM), dtype=torch.float32),
                "action_mask": torch.tensor([[False, True]], dtype=torch.bool),
            }
            action_idx, _value, _logp = server.infer(obs)
        finally:
            server.stop()

        self.assertEqual(action_idx, 0)


if __name__ == "__main__":
    unittest.main()
