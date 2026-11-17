# tests/test_puct_search.py
#
# Regression tests for the AlphaZero search. These fail on the old Gumbel
# implementation, which had no terminal handling, no sign flip when backing a
# value up the tree, and no per-node legal moves.

import os
import sys
import unittest
from queue import Empty

import numpy as np
import torch
from unittest.mock import patch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config import config
from game import GomokuGame
from mcts import AlphaZeroMCTS, PUCTNode


class StubInference:
    """Synchronous stand-in for the inference server.

    ``prior_fn(obs) -> logits[225]`` and ``value_fn(obs) -> float``.
    """

    def __init__(self, prior_fn=None, value_fn=None):
        self.prior_fn = prior_fn or (lambda obs: np.zeros(config.ACTION_SPACE_SIZE))
        self.value_fn = value_fn or (lambda obs: 0.0)
        self.pending = []
        self.calls = 0
        self.seen = []

    def put(self, item):
        worker_id, kind, data = item
        assert kind == 'initial', kind
        logits = np.asarray(self.prior_fn(data), dtype=np.float64)
        self.pending.append((logits, float(self.value_fn(data)), None))
        self.calls += 1
        self.seen.append(np.array(data))

    def get(self, timeout=None):
        if not self.pending:
            raise Empty
        return self.pending.pop(0)

    def get_nowait(self):
        raise Empty


def make_game(black, white, to_move=1, last_move=None):
    g = GomokuGame()
    for r, c in black:
        g.board[r, c] = 1
    for r, c in white:
        g.board[r, c] = -1
    g.current_player = to_move
    g.move_count = len(black) + len(white)
    g.last_move = last_move
    return g


class TestPUCTSearch(unittest.TestCase):

    def test_search_uses_network_and_never_queries_tactical_override(self):
        server = StubInference()
        game = make_game(black=[(7, 3), (7, 4), (7, 5), (7, 6)],
                         white=[(6, 6)], to_move=1, last_move=(6, 6))
        with patch.object(GomokuGame, 'find_immediate_wins', side_effect=AssertionError('override called')):
            _, _, action = AlphaZeroMCTS(0, server, server, add_root_noise=False, sample_moves=False).search(game)
        self.assertGreater(server.calls, 0)
        self.assertIn(action, [107, 112])

    def test_backprop_flips_sign_at_every_edge(self):
        """A node value is always from the point of view of the player to move."""
        server = StubInference()
        mcts = AlphaZeroMCTS(0, server, server, add_root_noise=False, sample_moves=False)

        root = PUCTNode()
        root.actions = np.array([10, 11], dtype=np.int32)
        root.priors = np.array([0.5, 0.5], dtype=np.float32)
        root.child_visits = np.zeros(2, dtype=np.int32)
        root.child_values = np.zeros(2, dtype=np.float32)
        root.index = {10: 0, 11: 1}
        root.is_expanded = True

        child = PUCTNode(parent=root, action=10)
        child.is_terminal = True
        root.children[10] = child

        # The player to move at the child has lost.
        mcts._backprop(child, -1.0)

        self.assertAlmostEqual(child.value(), -1.0)
        # Q(root, 10) is +1 for the player to move at the root.
        self.assertAlmostEqual(root.child_values[0], 1.0)
        self.assertAlmostEqual(root.value(), 1.0)

    def test_search_finds_win_in_one(self):
        """Black has four in a row: the search must play the fifth stone."""
        server = StubInference()
        game = make_game(black=[(7, 3), (7, 4), (7, 5), (7, 6)],
                         white=[(6, 6)], to_move=1, last_move=(6, 6))
        self.assertIsNone(game.get_game_ended())

        mcts = AlphaZeroMCTS(0, server, server, add_root_noise=False, sample_moves=False)
        policy, value, action = mcts.search(game)

        winning = {(7, 2), (7, 7)}
        chosen = (action // config.BOARD_SIZE, action % config.BOARD_SIZE)
        self.assertIn(chosen, winning,
                      f"search played {chosen} instead of a winning move {winning}")
        self.assertGreater(value, 0.5, f"root value should be winning, got {value}")
        self.assertAlmostEqual(float(np.sum(policy)), 1.0, places=5)

    def test_search_blocks_an_open_four(self):
        """If white does not block, black wins immediately."""
        # Black is one move away from five at (7, 7) (the (7, 2) end is blocked
        # by white). White to move must play (7, 7).
        server = StubInference()
        game = make_game(black=[(7, 3), (7, 4), (7, 5), (7, 6)],
                         white=[(7, 2), (6, 6)], to_move=-1, last_move=(7, 6))
        self.assertIsNone(game.get_game_ended())

        mcts = AlphaZeroMCTS(0, server, server, add_root_noise=False, sample_moves=False)
        # A zero-value, uniform-prior network needs enough visits to discover
        # the opponent's winning reply in every losing branch. No tactical aid.
        with patch.object(config, 'NUM_SIMULATIONS', 4096):
            policy, value, action = mcts.search(game)

        self.assertEqual((action // config.BOARD_SIZE, action % config.BOARD_SIZE), (7, 7),
                         f"white failed to block the four, played {action}")
        # Blocking neutralises the four (both ends are now covered), so the
        # position is no longer lost for white.
        self.assertGreater(value, -0.5, f"white is not lost after blocking, got {value}")
        self.assertAlmostEqual(float(policy[7 * config.BOARD_SIZE + 7]), float(np.max(policy)), places=5)

    def test_search_reports_lost_position(self):
        """A position where the opponent already has five is scored as lost."""
        game = make_game(black=[(7, 3), (7, 4), (7, 5), (7, 6), (7, 7)],
                         white=[(6, 6)], to_move=-1, last_move=(7, 7))
        self.assertIsNotNone(game.get_game_ended())

        server = StubInference()
        mcts = AlphaZeroMCTS(0, server, server, add_root_noise=False, sample_moves=False)
        policy, value, action = mcts.search(game)
        self.assertEqual(action, -1)
        self.assertAlmostEqual(float(np.sum(policy)), 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
