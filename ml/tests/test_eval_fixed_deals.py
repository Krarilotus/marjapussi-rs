import unittest
import sys
from pathlib import Path
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_fixed_deals import (
    LoadedFourModel,
    load_policy_artifact,
    normalize_fixed_hands,
    parse_card_spec,
    resolve_start_hands,
    summarize_outcomes,
)
from belief_model import BeliefNet
from decision_model import BiddingNet, PassingNet, PlayingNet


class EvalFixedDealsHelpersTest(unittest.TestCase):
    def test_parse_card_spec_accepts_int_and_symbol(self):
        self.assertEqual(parse_card_spec(0), 0)
        self.assertEqual(parse_card_spec("g-6"), 0)
        self.assertEqual(parse_card_spec("r-a"), 35)
        self.assertEqual(parse_card_spec("s10"), 25)
        self.assertEqual(parse_card_spec("RK"), 33)
        self.assertEqual(parse_card_spec("SZ"), 25)
        self.assertEqual(parse_card_spec("SA"), 26)
        # Legacy alias compatibility (older custom fixed-deal files)
        self.assertEqual(parse_card_spec("AZ"), 25)

    def test_normalize_fixed_hands_requires_full_unique_deck(self):
        hands = [
            list(range(0, 9)),
            list(range(9, 18)),
            list(range(18, 27)),
            list(range(27, 36)),
        ]
        norm = normalize_fixed_hands(hands)
        self.assertEqual(len(norm), 4)
        self.assertEqual(sum(len(h) for h in norm), 36)

    def test_normalize_fixed_hands_accepts_compact_object_format(self):
        hands = {
            "p0_hand": "G6 G7 G8 G9 GU GO GK GZ GS",
            "p1_hand": "E6 E7 E8 E9 EU EO EK EZ ES",
            "p2_hand": "S6 S7 S8 S9 SU SO SK SZ SA",
            "p3_hand": "R6 R7 R8 R9 RU RO RK RZ RA",
        }
        norm = normalize_fixed_hands(hands)
        self.assertEqual(len(norm), 4)
        self.assertEqual(sum(len(h) for h in norm), 36)

    def test_summarize_outcomes_basic(self):
        out = summarize_outcomes(
            [
                {
                    "no_one_played": False,
                    "contract_made": True,
                    "highest_bid": 200,
                    "playing_party_tricks": 6,
                    "defending_party_tricks": 3,
                    "playing_called_trumps": 2,
                    "playing_possible_pairs": 3,
                    "playing_questions": 3,
                    "playing_party": 0,
                    "team_points": [160, 80],
                },
                {
                    "no_one_played": True,
                    "playing_party": None,
                    "team_points": [90, 70],
                },
            ]
        )
        self.assertEqual(out["games"], 2)
        self.assertAlmostEqual(out["pass_game_rate"], 0.5)
        self.assertAlmostEqual(out["contract_made_rate"], 1.0)
        self.assertAlmostEqual(out["pair_call_rate"], 2 / 3)
        self.assertEqual(out["taken_games"], 1)
        self.assertEqual(out["taken_games_won"], 1)
        self.assertAlmostEqual(out["taken_game_win_rate"], 1.0)
        self.assertEqual(out["overbid_games"], 0)
        self.assertEqual(out["questions_to_trump"], 2)
        self.assertEqual(out["questions_no_trump"], 1)

    def test_resolve_start_hands_prefers_case_hands(self):
        case_hands = [
            list(range(0, 9)),
            list(range(9, 18)),
            list(range(18, 27)),
            list(range(27, 36)),
        ]
        debug_hands = [[0] * 9, [1] * 9, [2] * 9, [3] * 9]
        out = resolve_start_hands(case_hands, debug_hands)
        self.assertEqual(out, case_hands)

    def test_resolve_start_hands_uses_debug_when_case_missing(self):
        debug_hands = [
            list(range(0, 9)),
            list(range(9, 18)),
            list(range(18, 27)),
            list(range(27, 36)),
        ]
        out = resolve_start_hands(None, debug_hands)
        self.assertEqual(out, debug_hands)

    def test_load_policy_artifact_accepts_four_model_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            belief_path = tmp_path / "belief.pt"
            bidding_path = tmp_path / "bidding.pt"
            passing_path = tmp_path / "passing.pt"
            playing_path = tmp_path / "playing.pt"
            manifest_path = tmp_path / "manifest.json"

            torch.save({"state_dict": BeliefNet().state_dict()}, belief_path)
            torch.save({"state_dict": BiddingNet().state_dict()}, bidding_path)
            torch.save({"state_dict": PassingNet().state_dict()}, passing_path)
            torch.save({"state_dict": PlayingNet().state_dict()}, playing_path)
            manifest_path.write_text(
                (
                    "{"
                    "\"data_path\":\"ml/data/human_dataset.ndjson\","
                    "\"device\":\"cpu\","
                    "\"workers\":1,"
                    "\"decision_stages\":["
                    "{\"task\":\"bidding\",\"epochs\":1,\"batch\":8,\"target_acc\":0.5},"
                    "{\"task\":\"passing\",\"epochs\":1,\"batch\":8,\"target_acc\":0.5},"
                    "{\"task\":\"playing\",\"epochs\":1,\"batch\":8,\"target_acc\":0.5}"
                    "],"
                    "\"belief_stage\":{\"epochs\":1,\"batch\":8,\"target_hidden_acc\":0.7},"
                    "\"outputs\":{"
                    f"\"bidding\":\"{bidding_path.as_posix()}\","
                    f"\"passing\":\"{passing_path.as_posix()}\","
                    f"\"playing\":\"{playing_path.as_posix()}\","
                    f"\"belief\":\"{belief_path.as_posix()}\""
                    "}"
                    "}"
                ),
                encoding="utf-8",
            )

            loaded = load_policy_artifact(manifest_path, torch.device("cpu"), strict_param_budget=0)
            self.assertIsInstance(loaded, LoadedFourModel)


if __name__ == "__main__":
    unittest.main()
