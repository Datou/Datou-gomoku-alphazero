# tests/test_training_pipeline.py
#
# End-to-end checks for the data path that used to be broken:
#   1. a finished game must produce GameRecord/TrainingSlice objects,
#   2. the AlphaZero value target must be the game result z (not a bootstrapped
#      copy of the value head's own output).

import os
import sys
import unittest

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import loss as loss_module
import workers
from config import config
from data_structures import GameRecord, TrainingSlice
from game import GomokuGame
from network import GomokuNetEZ


def make_fake_game(num_moves=12, winner=1):
    """A finished game where black wins on the last move."""
    observations = [np.zeros((3, config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.float32)
                    for _ in range(num_moves)]
    actions = list(range(num_moves))
    rewards = np.zeros(num_moves, dtype=np.float32)
    if winner != 0:
        rewards[-1] = float(winner * (1 if (num_moves - 1) % 2 == 0 else -1))
    policies = []
    for i in range(num_moves):
        p = np.zeros(config.ACTION_SPACE_SIZE, dtype=np.float32)
        p[i] = 1.0
        policies.append(p)
    values = [float(winner * (1 if i % 2 == 0 else -1)) for i in range(num_moves)]
    moves_left = [num_moves - i for i in range(num_moves)]
    boards = [np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.int8) for _ in range(num_moves)]
    return observations, actions, rewards.tolist(), policies, values, moves_left, boards


class TestTrainingPipeline(unittest.TestCase):

    def test_game_record_and_slices_are_built(self):
        obs, actions, rewards, policies, values, moves_left, boards = make_fake_game()
        record, slices = workers.build_training_data(
            obs, actions, rewards, policies, values, moves_left, boards)

        self.assertEqual(len(record.observations), len(actions))
        self.assertEqual(list(record.values), values)
        self.assertEqual(len(slices), len(actions))
        self.assertIsInstance(slices[0], TrainingSlice)

        u = config.NUM_UNROLL_STEPS
        self.assertEqual(slices[0].observation.shape, (u + 1, 3, config.BOARD_SIZE, config.BOARD_SIZE))
        self.assertEqual(slices[0].policy_history.shape, (u + 1, config.ACTION_SPACE_SIZE))
        self.assertEqual(slices[0].value_history.shape, (u + 1,))
        self.assertEqual(slices[0].moves_left.shape, (u + 1,))

    def test_alphazero_value_target_is_the_game_result(self):
        obs, actions, rewards, policies, values, moves_left, boards = make_fake_game()
        _, slices = workers.build_training_data(
            obs, actions, rewards, policies, values, moves_left, boards)

        batch = slices[:4]
        obs_b = torch.from_numpy(np.stack([s.observation for s in batch])).to(config.DEVICE)
        act_b = torch.from_numpy(np.stack([s.action_history for s in batch])).to(config.DEVICE)
        rew_b = torch.from_numpy(np.stack([s.reward_history for s in batch])).to(config.DEVICE)
        pi_b = torch.from_numpy(np.stack([s.policy_history for s in batch])).to(config.DEVICE)
        val_b = torch.from_numpy(np.stack([s.value_history for s in batch])).to(config.DEVICE)
        mv_b = torch.from_numpy(np.stack([s.moves_left for s in batch])).to(config.DEVICE)

        model = GomokuNetEZ(config).to(config.DEVICE)
        target = GomokuNetEZ(config).to(config.DEVICE)
        target.load_state_dict(model.state_dict())

        captured = {}
        original = loss_module.scalar_to_support

        def spy(scalar, *a, **kw):
            captured.setdefault('scalar', scalar.detach().clone())
            return original(scalar, *a, **kw)

        loss_module.scalar_to_support = spy
        try:
            loss_value, logs = loss_module.calculate_loss(
                model, target, (obs_b, act_b, rew_b, pi_b, val_b, mv_b),
                torch.ones(len(batch), device=config.DEVICE))
        finally:
            loss_module.scalar_to_support = original

        # The value target sent to the loss must be z, the stored game result.
        np.testing.assert_allclose(
            captured['scalar'].flatten().cpu().numpy(), val_b[:, 0].cpu().numpy(), atol=1e-6)
        self.assertTrue(np.isfinite(loss_value.item()))

    def test_real_selfplay_records_terminal_outcome_without_tactical_override(self):
        from queue import Queue
        from threading import Event
        from types import SimpleNamespace
        from unittest.mock import patch

        for actions, expected in [
            ([108, 0, 109, 2, 110, 4, 111, 6, 112], [1, -1, 1, -1, 1, -1, 1, -1, 1]),
            ([0, 108, 2, 109, 4, 110, 6, 111, 30, 112], [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1]),
        ]:
            stopped = Event()
            records = []

            class PlannedSearch:
                def __init__(self, *args):
                    self.actions = iter(actions)

                def search(self, game):
                    action = next(self.actions)
                    policy = np.zeros(config.ACTION_SPACE_SIZE, dtype=np.float32)
                    policy[action] = 1
                    return policy, 0.0, action

            class Collect:
                def put(self, item):
                    records.append(item)
                    stopped.set()

            with patch.object(workers, 'AlphaZeroMCTS', PlannedSearch), \
                 patch.object(workers, 'DatabaseManager'), \
                 patch.object(GomokuGame, 'find_immediate_wins', side_effect=AssertionError('tactical override')):
                workers.universal_worker(0, SimpleNamespace(value=0), Collect(), Queue(),
                    Queue(), stopped, Queue(), Queue(), Queue(), Queue(),
                    SimpleNamespace(value=0), Queue(), Event())
            self.assertEqual(len(records), 1)
            record, slices, _ = records[0]
            self.assertEqual(record.values, expected)
            self.assertEqual([float(s.value_history[0]) for s in slices], expected)
            self.assertEqual(record.rewards[:-1], [0.0] * (len(actions) - 1))

    def test_draw_result_is_zero_for_both_players(self):
        self.assertEqual(workers.game_result_targets(0, [1, -1, 1, -1]), [0.0] * 4)
        with self.assertRaises(ValueError):
            workers.game_result_targets(None, [1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
