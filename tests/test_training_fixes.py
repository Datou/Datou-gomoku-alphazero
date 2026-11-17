# Regression tests for training-code review fixes.
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import workers
from config import config
from loss import augment_actions
from network import GomokuNetEZ
from utils import soft_update


class AugmentedActionTest(unittest.TestCase):
    def test_action_follows_rotated_board(self):
        h = w = config.BOARD_SIZE
        for r0, c0 in [(3, 7), (0, 14), (12, 1)]:
            board = torch.zeros(h, w)
            board[r0, c0] = 1
            for k in range(4):
                for flip in (False, True):
                    rotated = torch.rot90(board, k, dims=[0, 1])
                    if flip:
                        rotated = torch.flip(rotated, dims=[1])
                    action = int(augment_actions(torch.tensor(r0 * w + c0), k, flip, h, w))
                    self.assertEqual(action, int(rotated.argmax()), f'k={k} flip={flip}')


class SoftUpdateTest(unittest.TestCase):
    def test_target_moves_slowly_and_copies_buffers(self):
        source, target = torch.nn.BatchNorm1d(4), torch.nn.BatchNorm1d(4)
        with torch.no_grad():
            source.weight.fill_(1.0)
            target.weight.fill_(0.0)
            source.running_mean.fill_(3.0)
        soft_update(target, source, 0.995)
        self.assertAlmostEqual(target.weight[0].item(), 0.005, places=6)
        self.assertTrue(torch.equal(target.running_mean, source.running_mean))


class CheckpointGuardTest(unittest.TestCase):
    def test_non_finite_model_detected(self):
        model = GomokuNetEZ(config)
        self.assertTrue(workers.model_is_finite(model))
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = float('nan')
        self.assertFalse(workers.model_is_finite(model))


class LearningRateScheduleTest(unittest.TestCase):
    def test_warmup_cosine_then_hold_at_minimum(self):
        f = lambda t: workers.lr_factor(t, 100, 1000, 2e-4, 1e-7) * 2e-4
        self.assertAlmostEqual(f(0), 2e-6)
        self.assertAlmostEqual(f(100), 2e-4)
        self.assertAlmostEqual(f(550), (2e-4 + 1e-7) / 2)
        for t in (1000, 1500, 1900, 5000):
            self.assertAlmostEqual(f(t), 1e-7, msg=f'lr must not rise again at step {t}')

    def test_resume_sets_lr_from_step(self):
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda t: workers.lr_factor(t, 100, 1000, 2e-4, 1e-7))
        workers.set_scheduler_step(sched, 550)
        self.assertAlmostEqual(opt.param_groups[0]['lr'], (2e-4 + 1e-7) / 2)
        opt.step(); sched.step()
        self.assertEqual(sched.last_epoch, 551)


class TrimBufferTest(unittest.TestCase):
    def test_deletes_oldest_games_only_down_to_capacity(self):
        import tempfile
        from db_manager import DatabaseManager
        from data_structures import GameRecord, TrainingSlice
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(os.path.join(tmp, 'state.db'))
            for n in range(10):  # ten games of 10 positions each
                record = GameRecord([], list(range(10)), [], [], [], [], [])
                db.add_game_and_slices(record, [TrainingSlice(n, 0, 0, 0, 0, 0)] * 10, 0)
            db.trim_buffer(75)
            self.assertEqual(db.get_buffer_size(), 70)  # whole games: 3 removed
            conn = sqlite3_connect(db)
            remaining = [r[0] for r in conn.execute('SELECT game_id FROM games ORDER BY game_id')]
            self.assertEqual(remaining, list(range(4, 11)))
            db.trim_buffer(75)
            self.assertEqual(db.get_buffer_size(), 70)
            conn.close()
            import db_manager
            db_manager.thread_local.connections.pop(os.path.abspath(db.db_path)).close()


def sqlite3_connect(db):
    import sqlite3
    return sqlite3.connect(db.db_path)


if __name__ == '__main__':
    unittest.main()
