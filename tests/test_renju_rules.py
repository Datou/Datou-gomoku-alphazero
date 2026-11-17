import unittest
from unittest.mock import patch

import numpy as np

from config import config
from game import GomokuGame
from mcts import AlphaZeroMCTS
from renju_rules import RenjuRules


def position(black=(), white=()):
    game = GomokuGame(renju=True)
    for point in black:
        game.board[point] = 1
    for point in white:
        game.board[point] = -1
    game.move_count = int(np.count_nonzero(game.board))
    return game


class TestRenjuRules(unittest.TestCase):
    def test_overline_and_white_exemption(self):
        black = [(7, c) for c in (3, 4, 5, 7, 8)]
        game = position(black)
        self.assertEqual(game.forbidden_reason(7, 6), 'overline')
        self.assertNotIn((7, 6), game.get_valid_moves())
        game.current_player = -1
        self.assertIn((7, 6), game.get_valid_moves())
        game = position(white=black)
        game.current_player = -1
        game.do_move(7*15 + 6)
        self.assertEqual(game.get_game_ended(), -1)

    def test_double_three_and_four_three(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)])
        self.assertEqual(game.forbidden_reason(7, 7), 'double_three')
        game.board[5, 7] = 1
        self.assertIsNone(game.forbidden_reason(7, 7))  # 4-3 is allowed.

    def test_broken_three(self):
        game = position([(7, 5), (7, 8), (5, 7), (8, 7)])
        self.assertEqual(game.forbidden_reason(7, 7), 'double_three')

    def test_blocked_three_and_edge_are_false_threes(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)], [(7, 5), (7, 9)])
        self.assertIsNone(game.forbidden_reason(7, 7))
        game = position([(0, 1), (0, 3), (1, 2), (2, 2)])
        self.assertIsNone(game.forbidden_reason(0, 2))

    def test_open_four_is_one_four(self):
        game = position([(7, c) for c in (5, 6, 8)])
        self.assertIsNone(game.forbidden_reason(7, 7))
        game.board[7, 7] = 1
        fours = RenjuRules(game.board).fours((7, 7))
        self.assertEqual(len(fours), 1)
        self.assertEqual(len(next(iter(fours.values()))), 2)

    def test_cross_double_four(self):
        game = position([(7, 5), (7, 6), (7, 8), (5, 7), (6, 7), (8, 7)])
        self.assertEqual(game.forbidden_reason(7, 7), 'double_four')

    def test_same_line_double_four(self):
        # X.XX.X becomes X.XXX.X, two different four-stone sets.
        game = position([(7, c) for c in (4, 6, 8, 10)])
        self.assertEqual(game.forbidden_reason(7, 7), 'double_four')

    def test_overline_completion_does_not_count_as_four(self):
        game = position([(7, c) for c in (3, 4, 5, 8)], [(7, 9)])
        game.board[7, 7] = 1
        self.assertEqual(len(RenjuRules(game.board).fours((7, 7))), 0)

    def test_exact_five_precedes_double_three_and_four_and_overline(self):
        game = position([(7, c) for c in (3, 4, 5, 6)] +
                        [(r, 7) for r in (4, 5, 6, 8, 9)] + [(6, 6), (8, 8)])
        self.assertIsNone(game.forbidden_reason(7, 7))
        game.do_move(7*15 + 7)
        self.assertEqual(game.get_game_ended(), 1)

    def test_false_three_extension_is_overline(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)] +
                        [(r, 9) for r in (4, 5, 6, 8, 9)], [(7, 4)])
        game.board[7, 7] = 1
        self.assertEqual(game.forbidden_reason(7, 9), 'overline')
        self.assertIsNone(game.forbidden_reason(7, 7))

    def test_false_three_extension_is_double_four(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)] +
                        [(r, 9) for r in (5, 6, 8)], [(7, 4)])
        game.board[7, 7] = 1
        self.assertEqual(game.forbidden_reason(7, 9), 'double_four')
        self.assertIsNone(game.forbidden_reason(7, 7))

    def test_recursive_false_three_extension_is_double_three(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7),
                         (6, 9), (8, 9), (6, 10), (8, 8)], [(7, 4)])
        game.board[7, 7] = 1
        self.assertEqual(game.forbidden_reason(7, 9), 'double_three')
        self.assertIsNone(game.forbidden_reason(7, 7))

    def test_extension_making_five_elsewhere_is_not_a_three(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)] +
                        [(r, 9) for r in (4, 5, 6, 8)], [(7, 4)])
        self.assertIsNone(game.forbidden_reason(7, 7))

    def test_checks_restore_board_and_clone_keeps_rules(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)])
        before = game.board.copy()
        for _ in range(2):
            game.get_candidate_moves()
            game.get_valid_moves()
            np.testing.assert_array_equal(game.board, before)
        game.board[7, 7] = 1
        before = game.board.copy()
        self.assertTrue(game.check_forbidden(7, 7))
        np.testing.assert_array_equal(game.board, before)
        self.assertTrue(game.clone().renju)

    def test_played_black_foul_loses(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)])
        game.do_move(7*15 + 7)
        self.assertEqual(game.get_game_ended(), -1)

    def test_rotations_and_reflections_agree(self):
        for black, expected in [([(7, c) for c in (4, 6, 8, 10)], 'double_four'),
                                ([(7, 6), (7, 8), (6, 7), (8, 7)], 'double_three')]:
            for reflected in (False, True):
                for rotation in range(4):
                    game = position(black)
                    if reflected:
                        game.board = np.fliplr(game.board).copy()
                    game.board = np.rot90(game.board, rotation).copy()
                    self.assertEqual(game.forbidden_reason(7, 7), expected)

    def test_freestyle_still_allows_double_three_and_overline(self):
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)])
        game.renju = False
        self.assertIn((7, 7), game.get_valid_moves())
        game.do_move(7*15 + 7)
        self.assertIsNone(game.get_game_ended())

    def test_puct_masks_forbidden_prior_without_tactical_override(self):
        class Inference:
            def get_nowait(self):
                from queue import Empty
                raise Empty
            def put(self, request):
                pass
            def get(self, timeout=None):
                logits = np.zeros(225, dtype=np.float32)
                logits[7*15 + 7] = 100
                return logits, 0., None
        game = position([(7, 6), (7, 8), (6, 7), (8, 7)])
        engine = Inference()
        with patch.object(config, 'NUM_SIMULATIONS', 8):
            policy, _, action = AlphaZeroMCTS(0, engine, engine, False, False).search(game)
        self.assertEqual(policy[7*15 + 7], 0.)
        self.assertNotEqual(action, 7*15 + 7)
        self.assertIsNone(game.forbidden_reason(action//15, action%15))
        self.assertFalse(config.ENABLE_FORCED_MOVE_PRUNING)


if __name__ == '__main__':
    unittest.main()
