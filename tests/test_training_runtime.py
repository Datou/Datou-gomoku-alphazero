import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import json
import hashlib
import torch
import multiprocessing as mp
from unittest.mock import patch

from training_runtime import RunLock, training_budget_available, validate_run_rules, load_warm_start, atomic_json, worker_entry, assert_evaluation_allowed, resumed_sample_count
from db_manager import DatabaseManager


def fail_with_undrained_queue(queue):
    queue.put(b'x' * 1000000)
    raise RuntimeError('expected worker failure')


def increment_shared_counter(counter):
    for _ in range(100):
        with counter.get_lock():
            counter.value += 1


def read_single_writer_flags(mode, step, messages, actor):
    import time
    messages.put(('ready', actor, mode.value, step.value))
    deadline = time.monotonic() + 15
    while mode.value == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    messages.put(('updated', actor, mode.value, step.value))


class TestTrainingRuntime(unittest.TestCase):
    def test_single_writer_control_flags_broadcast_to_all_spawned_readers(self):
        import ctypes
        context = mp.get_context('spawn')
        mode, step = context.RawValue('i', 0), context.RawValue('i', 0)
        self.assertEqual(ctypes.sizeof(mode), 4)
        self.assertEqual(ctypes.addressof(mode) % ctypes.alignment(ctypes.c_int), 0)
        messages = context.Queue()
        children = [context.Process(target=read_single_writer_flags, args=(mode, step, messages, i)) for i in range(8)]
        try:
            for child in children:
                child.start()
            ready = [messages.get(timeout=30) for _ in children]
            self.assertEqual({row[1] for row in ready}, set(range(8)))
            self.assertTrue(all(row[0] == 'ready' and row[2:] == (0, 0) for row in ready))
            step.value = 15250
            mode.value = 1
            updated = [messages.get(timeout=15) for _ in children]
            self.assertEqual({row[1] for row in updated}, set(range(8)))
            self.assertTrue(all(row[0] == 'updated' and row[2:] == (1, 15250) for row in updated))
            for child in children:
                child.join(5)
                self.assertEqual(child.exitcode, 0)
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                    child.join(5)
            messages.close()

    def test_spawned_children_can_share_synchronized_value(self):
        context = mp.get_context('spawn')
        counter = context.Value('i', 0)
        children = [context.Process(target=increment_shared_counter, args=(counter,)) for _ in range(8)]
        try:
            for child in children:
                child.start()
            for child in children:
                child.join(timeout=30)
                self.assertFalse(child.is_alive(), 'Shared-value probe did not exit.')
                self.assertEqual(child.exitcode, 0)
            self.assertEqual(counter.value, 800)
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                    child.join(5)

    def test_crash_rollback_does_not_refund_consumed_samples(self):
        count = resumed_sample_count(15250, 1952000, {'step': 15434, 'samples_consumed': 1975552})
        self.assertEqual(count, 1975552)
        self.assertFalse(training_budget_available(246945, count, 128, 8))
        self.assertTrue(training_budget_available(246961, count, 128, 8))

    def test_old_or_lower_status_cannot_reduce_saved_sample_count(self):
        self.assertEqual(resumed_sample_count(15250, 1975552, {}), 1975552)
        self.assertEqual(resumed_sample_count(15250, 1975552,
                         {'step': 15250, 'samples_consumed': 1952000}), 1975552)
        self.assertEqual(resumed_sample_count(15250, 1975552,
                         {'step': 15000, 'samples_consumed': 2000000}), 1975552)

    def test_invalid_sample_high_water_mark_is_rejected(self):
        for value in (float('nan'), -1, True, '1975552'):
            with self.assertRaises(ValueError):
                resumed_sample_count(15250, 1952000, {'step': 15434, 'samples_consumed': value})

    def test_existing_database_opens_without_writer_lock_or_game_scan(self):
        import sqlite3
        from concurrent.futures import ThreadPoolExecutor
        from db_manager import get_db_connection, thread_local
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'state.db')
            DatabaseManager(path)
            writer = sqlite3.connect(path)
            writer.execute('BEGIN IMMEDIATE')
            writer.execute("UPDATE training_counters SET value=12345 WHERE key='positions'")

            def reader():
                connection = get_db_connection(path)
                connection.execute('PRAGMA busy_timeout=100')
                statements = []
                connection.set_trace_callback(statements.append)
                try:
                    manager = DatabaseManager(path)
                    return manager.get_total_positions(), statements
                finally:
                    thread_local.connections.pop(os.path.abspath(path)).close()

            try:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(lambda _: reader(), range(8)))
                for total, statements in results:
                    self.assertEqual(total, 0)  # sees only the committed counter
                    self.assertFalse(any('INSERT' in sql.upper() or 'CREATE' in sql.upper()
                                         or 'SUM(' in sql.upper() for sql in statements), statements)
            finally:
                writer.rollback()
                writer.close()
                thread_local.connections.pop(os.path.abspath(path)).close()

    def test_missing_counter_migrates_once_from_existing_games(self):
        from db_manager import get_db_connection, thread_local
        from data_structures import GameRecord
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'state.db')
            manager = DatabaseManager(path)
            manager.add_game_and_slices(GameRecord([], [1, 2, 3], [], [], [], [], []), [], 0)
            connection = get_db_connection(path)
            try:
                with connection:
                    connection.execute("DELETE FROM training_counters WHERE key='positions'")
                self.assertEqual(DatabaseManager(path).get_total_positions(), 3)
                with connection:
                    connection.execute('DELETE FROM games')
                self.assertEqual(DatabaseManager(path).get_total_positions(), 3)
            finally:
                thread_local.connections.pop(os.path.abspath(path)).close()

    def test_review_evaluation_never_overrides_user_pause_or_replaced_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'STOP').write_text('STAGE_REVIEW_PAUSE:test-token')
            atomic_json(output/'review_pause.json', {'active': True, 'status': 'stage_review_paused',
                        'allow_evaluation': True, 'stop_token': 'test-token'})
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                assert_evaluation_allowed(output)
            assert_evaluation_allowed(output, review_only=True)
            atomic_json(output/'user_pause.json', {'status': 'user_paused'})
            with self.assertRaisesRegex(RuntimeError, 'user pause'):
                assert_evaluation_allowed(output, review_only=True)
            atomic_json(output/'user_pause.json', {'status': 'resumed_by_user'})
            (output/'STOP').write_text('USER PAUSE')
            with self.assertRaisesRegex(RuntimeError, 'STOP'):
                assert_evaluation_allowed(output, review_only=True)
            (output/'STOP').unlink()
            with self.assertRaisesRegex(RuntimeError, 'remain present'):
                assert_evaluation_allowed(output, review_only=True)

    def test_transient_windows_read_lock_does_not_lose_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            atomic_json(path, {'step': 478})
            real_replace = os.replace
            calls = []
            def sharing_violation(source, target):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError('simulated Windows sharing violation')
                return real_replace(source, target)
            with patch('training_runtime.os.replace', side_effect=sharing_violation), \
                 patch('training_runtime.time.sleep'):
                atomic_json(path, {'step': 479})
            self.assertEqual(json.loads(path.read_text())['step'], 479)
            self.assertEqual(len(calls), 3)

    def test_worker_error_visible_even_when_telemetry_queue_is_not_drained(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'GOMOKU_OUTPUT_DIR': ''}):
            os.environ['GOMOKU_OUTPUT_DIR'] = directory
            context = mp.get_context('spawn')
            queue = context.Queue()
            process = context.Process(target=worker_entry, args=(fail_with_undrained_queue, (queue,)))
            process.start()
            process.join(timeout=20)
            if process.is_alive():
                process.terminate(); process.join(5)
                self.fail('Failed target remained alive waiting for a queue feeder.')
            self.assertNotEqual(process.exitcode, 0)
            state = json.loads((Path(directory)/'worker_status'/f'{process.pid}.json').read_text())
            self.assertEqual(state['phase'], 'failed')
            self.assertIn('expected worker failure', state['traceback'])
            queue.close()

    def test_rules_change_rejected_before_overwriting_old_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run_config.json'
            original = json.dumps({'config': {'ENABLE_RENJU': False}})
            path.write_text(original, encoding='utf-8')
            validate_run_rules(directory, 'freestyle_v1')
            with self.assertRaisesRegex(RuntimeError, 'separate output'):
                validate_run_rules(directory, 'renju_forbidden_v1')
            self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_warm_start_transfers_only_model_and_validates_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
            path = Path(directory) / 'seed.pt'
            torch.save(source.state_dict(), path)
            manifest = {'target_ruleset_id': 'renju_forbidden_v1',
                        'seed_checkpoint': 'seed.pt',
                        'seed_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            (Path(directory)/'warmstart.json').write_text(json.dumps(manifest), encoding='utf-8')
            optimizer = torch.optim.Adam(target.parameters())
            self.assertTrue(load_warm_start(target, directory, 'renju_forbidden_v1'))
            self.assertFalse(optimizer.state)
            for key in source.state_dict():
                self.assertTrue(torch.equal(source.state_dict()[key], target.state_dict()[key]))
            with self.assertRaisesRegex(RuntimeError, 'rules'):
                load_warm_start(target, directory, 'freestyle_v1')
            path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                load_warm_start(target, directory, 'renju_forbidden_v1')

    def test_sample_budget_counts_samples_not_optimization_steps(self):
        self.assertTrue(training_budget_available(100, 672, 128, 8))
        self.assertFalse(training_budget_available(100, 673, 128, 8))

    def test_second_process_cannot_acquire_project_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'training.lock'
            with RunLock(path):
                result = subprocess.run([sys.executable, '-c',
                    'from training_runtime import RunLock; import sys; RunLock(sys.argv[1]).__enter__()', str(path)],
                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Another training process', result.stderr)
            with RunLock(path):
                pass

    def test_database_isolation_cascade_and_lifetime_position_counter(self):
        from data_structures import GameRecord
        from db_manager import get_db_connection, thread_local
        with tempfile.TemporaryDirectory() as directory:
            paths = [os.path.join(directory, name) for name in ['a.db', 'b.db']]
            try:
                first, second = [DatabaseManager(path) for path in paths]
                record = GameRecord([], [1, 2, 3], [], [], [], [], [])
                game_id = first.add_game_and_slices(record, [1, 2, 3], 0)
                self.assertEqual(first.get_total_positions(), 3)
                self.assertEqual(first.get_buffer_size(), 3)
                self.assertEqual(second.get_buffer_size(), 0)
                connection = get_db_connection(paths[0])
                with connection:
                    connection.execute('DELETE FROM games WHERE game_id = ?', (game_id,))
                self.assertEqual(first.get_buffer_size(), 0)
                self.assertEqual(first.get_total_positions(), 3)
            finally:
                for path in paths:
                    connection = thread_local.connections.pop(os.path.abspath(path), None)
                    if connection:
                        connection.close()

