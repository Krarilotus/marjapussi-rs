import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_online import (
    _allocate_remaining_rounds_by_phase,
    _build_phase_plan,
    _phase_start_trick,
)


class TrainOnlineCurriculumTest(unittest.TestCase):
    def test_phase_plan_matches_40_5_5_50_split(self):
        phase_plan = _build_phase_plan(
            total_rounds=645,
            trick_frac=0.40,
            passing_frac=0.05,
            bidding_frac=0.05,
        )
        self.assertEqual(phase_plan["trick"], 258)
        self.assertEqual(phase_plan["passing"], 32)
        self.assertEqual(phase_plan["bidding_prop"], 32)
        self.assertEqual(phase_plan["full_game"], 323)

    def test_trick_target_cycles_through_1_to_9(self):
        seen = {_phase_start_trick("trick", rnd + 1) for rnd in range(18)}
        self.assertEqual(seen, set(range(1, 10)))
        self.assertEqual(_phase_start_trick("passing", 1), 0)
        self.assertEqual(_phase_start_trick("bidding_prop", 1), -1)
        self.assertIsNone(_phase_start_trick("full_game", 1))

    def test_remaining_round_allocation_extends_last_phase_after_early_transition(self):
        phase_plan = {
            "trick": 258,
            "passing": 32,
            "bidding_prop": 32,
            "full_game": 323,
        }
        alloc = _allocate_remaining_rounds_by_phase(
            phase_plan,
            current_phase_idx=3,
            current_phase_local_round=8,
            rounds_left=501,
        )
        self.assertEqual(sum(alloc.values()), 501)
        self.assertEqual(alloc["trick"], 0)
        self.assertEqual(alloc["passing"], 0)
        self.assertEqual(alloc["bidding_prop"], 0)
        self.assertEqual(alloc["full_game"], 501)

    def test_remaining_round_allocation_matches_standard_plan_without_early_transition(self):
        phase_plan = {
            "trick": 258,
            "passing": 32,
            "bidding_prop": 32,
            "full_game": 323,
        }
        alloc = _allocate_remaining_rounds_by_phase(
            phase_plan,
            current_phase_idx=0,
            current_phase_local_round=78,
            rounds_left=567,
        )
        self.assertEqual(sum(alloc.values()), 567)
        self.assertEqual(alloc["trick"], 180)
        self.assertEqual(alloc["passing"], 32)
        self.assertEqual(alloc["bidding_prop"], 32)
        self.assertEqual(alloc["full_game"], 323)


if __name__ == "__main__":
    unittest.main()
