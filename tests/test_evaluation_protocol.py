import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from checkpoint_metadata import checkpoint_parameters, read_checkpoint_parameters
from config import config
from tools import evaluate_progress as arena


class TestEvaluationProtocol(unittest.TestCase):
    def setUp(self):
        self.previous_config = vars(config).copy()

    def tearDown(self):
        vars(config).clear()
        vars(config).update(self.previous_config)

    def test_export_retains_renju_and_missing_rules_cannot_silently_be_freestyle(self):
        params = checkpoint_parameters(config)
        params.update(ENABLE_RENJU=True, RULESET_ID='renju_forbidden_v1', RULESET='renju')
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output/'model_weights'/'model_weights_step_500.pt'
            checkpoint.parent.mkdir()
            (output/'run_config.json').write_text(json.dumps({'config': params}))
            sidecar = checkpoint.with_suffix('.json')
            sidecar.write_text(json.dumps(params))
            self.assertTrue(read_checkpoint_parameters(checkpoint)['ENABLE_RENJU'])
            del params['RULESET_ID']
            del params['ENABLE_RENJU']
            sidecar.write_text(json.dumps(params))
            with self.assertRaisesRegex(ValueError, 'explicit rule metadata'):
                read_checkpoint_parameters(checkpoint)

    def test_frozen_openings_are_legal_and_nonterminal(self):
        self.assertEqual(hashlib.sha256(arena.CONFIRMATION.read_bytes()).hexdigest(), arena.CONFIRMATION_SHA256)
        frozen = json.loads(arena.CONFIRMATION.read_text())
        self.assertEqual(len(frozen['openings']), 50)
        for actions in frozen['openings']:
            game = arena.opening_game(actions, True)
            self.assertIsNone(game.get_game_ended())
            self.assertEqual(game.current_player, 1)
        with self.assertRaises(ValueError):
            arena.opening_game([112, 112], True)

    def test_color_pair_statistics_do_not_mistake_black_advantage_for_progress(self):
        games = [{'opening': i, 'ai_color': color, 'ai_result': color} for i in range(50) for color in (1, -1)]
        result = arena.summarize(games)
        self.assertEqual((result['wins'], result['draws'], result['losses']), (50, 0, 50))
        self.assertEqual(result['score_rate'], 0.5)
        self.assertEqual(result['paired_bootstrap_95_score_interval'], [0.5, 0.5])
        self.assertEqual(result['by_color']['-1']['score_rate'], 0)

    def test_pause_resumes_completed_games_and_reuses_final_result(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'model_weights'/'model_weights_step_500.pt'
            checkpoint.parent.mkdir()
            params = checkpoint_parameters(config)
            params.update(ENABLE_RENJU=True, RULESET_ID='renju_forbidden_v1', RULESET='renju')
            checkpoint.with_suffix('.json').write_text(json.dumps(params))
            torch.save(torch.nn.Linear(1, 1).state_dict(), checkpoint)
            actions = [112, 0, 113, 2, 114, 4, 115, 6, 116]
            searches = []

            class PlannedEngine:
                def __init__(self, *a, **kw):
                    pass

                def search(self, game):
                    searches.append(game.move_count)
                    return None, None, actions[game.move_count]

                def choose(self, game):
                    searches.append(game.move_count)
                    return actions[game.move_count]

            real_atomic = arena.atomic_json
            stop = Path(directory)/'STOP'

            def pause_after_first(path, value):
                real_atomic(path, value)
                if len(value['games']) == 1:
                    stop.write_text('user pause')

            with patch.object(arena, 'OPENINGS', [[(7, 7), (0, 0), (7, 8), (0, 2)]]), \
                 patch.object(arena, 'GomokuNetEZ', lambda _: torch.nn.Linear(1, 1)), \
                 patch.object(arena, 'AlphaZeroMCTS', PlannedEngine), \
                 patch.object(arena, 'GreedyBot', lambda **kw: PlannedEngine()), \
                 patch.object(arena, 'atomic_json', side_effect=pause_after_first):
                with self.assertRaisesRegex(RuntimeError, 'STOP'):
                    arena.evaluate(checkpoint, pairs=1)
            self.assertEqual(len(searches), 5)
            stop.unlink()
            with patch.object(arena, 'OPENINGS', [[(7, 7), (0, 0), (7, 8), (0, 2)]]), \
                 patch.object(arena, 'GomokuNetEZ', lambda _: torch.nn.Linear(1, 1)), \
                 patch.object(arena, 'AlphaZeroMCTS', PlannedEngine), \
                 patch.object(arena, 'GreedyBot', lambda **kw: PlannedEngine()):
                result = arena.evaluate(checkpoint, pairs=1)
                self.assertEqual(len(searches), 10)
                self.assertEqual((result['wins'], result['losses']), (1, 1))
                reused = arena.evaluate(checkpoint, pairs=1)
                self.assertEqual(reused, result)
                self.assertEqual(len(searches), 10)
            result['checkpoint_sha256'] = 'changed'
            arena.result_path(checkpoint, 128, 1).write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                arena.evaluate(checkpoint, pairs=1)
