import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_online import _is_advantage_target_action


class TrainOnlineAdvantageTargetTest(unittest.TestCase):
    def test_full_game_targets_bid_pass_and_info(self):
        self.assertTrue(_is_advantage_target_action(None, 3, is_bid=True, is_pass=False, is_info=False))
        self.assertTrue(_is_advantage_target_action(None, 3, is_bid=False, is_pass=True, is_info=False))
        self.assertTrue(_is_advantage_target_action(None, 3, is_bid=False, is_pass=False, is_info=True))
        self.assertFalse(_is_advantage_target_action(None, 3, is_bid=False, is_pass=False, is_info=False))

    def test_curriculum_targets_remain_unchanged(self):
        self.assertTrue(_is_advantage_target_action(-1, 1, is_bid=True, is_pass=False, is_info=False))
        self.assertFalse(_is_advantage_target_action(-1, 1, is_bid=False, is_pass=True, is_info=False))
        self.assertTrue(_is_advantage_target_action(0, 1, is_bid=False, is_pass=True, is_info=False))
        self.assertTrue(_is_advantage_target_action(4, 4, is_bid=False, is_pass=False, is_info=False))
        self.assertFalse(_is_advantage_target_action(4, 3, is_bid=False, is_pass=False, is_info=False))


if __name__ == "__main__":
    unittest.main()
