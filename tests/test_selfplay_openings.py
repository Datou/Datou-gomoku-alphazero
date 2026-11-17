import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from game import GomokuGame
from rif_openings import (CENTRE, RIF_OPENINGS, apply_symmetry, choose_opening,
                          opening_actions, sample_opening, validate_opening)


class TestRifOpenings(unittest.TestCase):
    def test_book_has_26_distinct_openings(self):
        self.assertEqual(len(RIF_OPENINGS), 26)
        self.assertEqual(len({tuple(opening) for opening in RIF_OPENINGS}), 26)
        direct = [opening for opening in RIF_OPENINGS if opening[1] == (7, 8)]
        indirect = [opening for opening in RIF_OPENINGS if opening[1] == (8, 8)]
        self.assertEqual(len(direct), 13)
        self.assertEqual(len(indirect), 13)

    def test_every_opening_is_legal_renju(self):
        for opening in RIF_OPENINGS:
            validate_opening(opening)
            game = GomokuGame(renju=True)
            for r, c in opening:
                self.assertIn((r, c), game.get_valid_moves())
                game.do_move(r * 15 + c)
                self.assertIsNone(game.get_game_ended())
            self.assertEqual(game.move_count, 3)
            self.assertEqual(game.current_player, -1)

    def test_first_move_is_centre_and_third_is_near(self):
        for opening in RIF_OPENINGS:
            self.assertEqual(opening[0], CENTRE)
            self.assertLessEqual(max(abs(opening[2][0] - 7), abs(opening[2][1] - 7)), 2)

    def test_symmetries_stay_legal(self):
        for opening in RIF_OPENINGS:
            for index in range(8):
                transformed = apply_symmetry(opening, index)
                self.assertEqual(transformed[0], CENTRE)
                self.assertEqual(len(set(transformed)), 3)
                game = GomokuGame(renju=True)
                for r, c in transformed:
                    self.assertTrue(0 <= r < 15 and 0 <= c < 15)
                    self.assertIn((r, c), game.get_valid_moves())
                    game.do_move(r * 15 + c)

    def test_sampling_returns_legal_three_stone_prefix(self):
        import numpy as np
        rng = np.random.default_rng(7)
        for _ in range(100):
            opening = sample_opening(rng)
            validate_opening(opening)
            game = GomokuGame(renju=True)
            for action in opening_actions(opening):
                self.assertIn((action // 15, action % 15), game.get_valid_moves())
                game.do_move(action)

    def test_free_opening_ratio_controls_prefix(self):
        import numpy as np
        rng = np.random.default_rng(11)
        self.assertTrue(all(len(choose_opening(rng, 0.0)) == 3 for _ in range(50)))
        self.assertTrue(all(choose_opening(rng, 1.0) == [CENTRE] for _ in range(50)))
        mixed = [choose_opening(rng, 0.5) for _ in range(200)]
        self.assertTrue(any(len(opening) == 1 for opening in mixed))
        self.assertTrue(any(len(opening) == 3 for opening in mixed))
        with self.assertRaises(ValueError):
            choose_opening(rng, 1.5)


if __name__ == '__main__':
    unittest.main()
