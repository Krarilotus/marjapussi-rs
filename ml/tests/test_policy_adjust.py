import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.policy_adjust import apply_bid_consistency_adjustments


class PolicyAdjustTest(unittest.TestCase):
    def test_bid_soft_cap_penalizes_overreaching_bid_logits(self):
        logits = torch.zeros((1, 2), dtype=torch.float32)
        action_feats = torch.zeros((1, 2, 64), dtype=torch.float32)
        action_mask = torch.zeros((1, 2), dtype=torch.bool)
        pts_pred = torch.tensor([[0.1, 0.0]], dtype=torch.float32)  # ~42 points

        action_feats[0, 0, 1] = 1.0   # bid action
        action_feats[0, 0, 32] = (180.0 - 120.0) / 300.0

        adjusted, stats = apply_bid_consistency_adjustments(
            logits,
            action_feats,
            action_mask,
            pts_pred,
            bid_soft_cap_weight=1.0,
            bid_soft_cap_margin=0.0,
        )

        self.assertLess(adjusted[0, 0].item(), logits[0, 0].item())
        self.assertGreater(stats["bid_soft_cap_mean"].item(), 0.0)

    def test_stop_bid_penalty_hits_when_raise_is_makeable(self):
        logits = torch.zeros((1, 2), dtype=torch.float32)
        action_feats = torch.zeros((1, 2, 64), dtype=torch.float32)
        action_mask = torch.zeros((1, 2), dtype=torch.bool)
        pts_pred = torch.tensor([[0.5, 0.0]], dtype=torch.float32)  # ~210 points

        action_feats[0, 0, 2] = 1.0   # stop bidding
        action_feats[0, 1, 1] = 1.0   # legal raise
        action_feats[0, 1, 32] = (140.0 - 120.0) / 300.0

        adjusted, stats = apply_bid_consistency_adjustments(
            logits,
            action_feats,
            action_mask,
            pts_pred,
            stop_bid_penalty_weight=1.0,
            stop_bid_margin=0.0,
        )

        self.assertLess(adjusted[0, 0].item(), logits[0, 0].item())
        self.assertGreater(stats["stop_bid_penalty_mean"].item(), 0.0)
        self.assertGreater(stats["makeable_bid_rate"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
