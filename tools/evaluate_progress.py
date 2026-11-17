"""Resumable paired-opening arena. Evaluation never supplies training actions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from contextlib import ExitStack
from types import SimpleNamespace

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checkpoint_metadata import read_checkpoint_parameters
from config import config
from game import GomokuGame
from network import GomokuNetEZ
from mcts import AlphaZeroMCTS
from tools.arena import LocalInference
from tools.greedy_bot import GreedyBot
from training_runtime import atomic_json, RunLock, assert_evaluation_allowed

OPENINGS = [
    [(7, 7), (7, 8), (8, 7), (6, 6)],
    [(7, 7), (6, 8), (8, 6), (8, 8)],
    [(7, 7), (8, 8), (6, 8), (7, 9)],
    [(7, 7), (9, 7), (7, 8), (5, 5)],
]
CONFIRMATION = ROOT / 'tests' / 'fixtures' / 'renju_confirmation_openings_v1.json'
CONFIRMATION_SHA256 = 'a72c65950374917829deb4750d9f59daaea44253c10b40e9ec4afa73dfdc6583'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def opening_game(actions, renju):
    game = GomokuGame(renju=renju)
    for action in actions:
        legal = {(r * game.board_size + c) for r, c in game.get_valid_moves()}
        if game.get_game_ended() is not None or action not in legal:
            raise ValueError('Illegal or already terminal evaluation opening/trajectory.')
        game.do_move(action)
    return game


def summarize(games):
    def stats(rows):
        w, d, l = [sum(g['ai_result'] == result for g in rows) for result in (1, 0, -1)]
        return {'wins': w, 'draws': d, 'losses': l, 'n': len(rows),
                'score_rate': (w + 0.5*d) / len(rows) if rows else None}
    result = stats(games)
    result['by_color'] = {str(c): stats([g for g in games if g['ai_color'] == c]) for c in (1, -1)}
    groups = {}
    for g in games:
        groups.setdefault(g['opening'], []).append((g['ai_result'] + 1) / 2)
    paired = np.array([np.mean(values) for values in groups.values() if len(values) == 2])
    # Resample openings with both colors together. This is a descriptive interval
    # for this opening distribution, not an independent human Elo estimate.
    if len(paired) >= 2:
        rng = np.random.default_rng(20261109)
        means = rng.choice(paired, size=(10000, len(paired)), replace=True).mean(axis=1)
        result['paired_bootstrap_95_score_interval'] = np.quantile(means, [0.025, 0.975]).tolist()
    return result


def result_path(checkpoint, sims, pairs=4, opponent='greedy', suite='progress'):
    suffix = f'sims{sims}_pairs{pairs}' if suite == 'progress' else f'sims{sims}_confirmation50'
    if opponent != 'greedy':
        suffix += '_vs_initial'
    root = checkpoint.parent.parent
    run_dir = root / 'training' if (root / 'training').is_dir() else root
    return run_dir / 'evaluations' / f'{checkpoint.stem}_{suffix}.json'


def evaluate(checkpoint, sims=128, pairs=4, opponent='greedy', suite='progress', review_only=False):
    checkpoint = Path(checkpoint).resolve()
    params = read_checkpoint_parameters(checkpoint)
    renju = bool(params.get('ENABLE_RENJU'))
    if suite == 'confirmation' and not renju:
        raise ValueError('Confirmation openings are for Renju only.')
    config.DEVICE = torch.device('cpu')
    torch.set_num_threads(1)
    config.NUM_SIMULATIONS = sims
    config.ENABLE_RENJU = renju
    config.ENABLE_FORCED_MOVE_PRUNING = False
    for key in ('C_PUCT', 'ACTION_RADIUS'):
        if key in params:
            setattr(config, key, params[key])
    if suite == 'confirmation':
        if digest(CONFIRMATION) != CONFIRMATION_SHA256:
            raise ValueError('Frozen confirmation opening set changed.')
        frozen = json.loads(CONFIRMATION.read_text(encoding='utf-8'))
        if frozen['ruleset_id'] != 'renju_forbidden_v1' or len(frozen['openings']) != 50:
            raise ValueError('Invalid frozen confirmation set.')
        openings = frozen['openings']
    else:
        openings = [[r * config.BOARD_SIZE + c for r, c in opening] for opening in OPENINGS[:pairs]]
    opening_sha = hashlib.sha256(json.dumps(openings, separators=(',', ':')).encode()).hexdigest()

    def engine(path):
        engine_params = read_checkpoint_parameters(path)
        model = GomokuNetEZ(SimpleNamespace(**engine_params))
        state = torch.load(path, map_location='cpu', weights_only=True)
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError('Non-finite evaluation weights.')
        model.load_state_dict(state, strict=True)
        model.eval()
        server = LocalInference(model)
        return AlphaZeroMCTS(0, server, server, add_root_noise=False, sample_moves=False)

    protocol = ('fifty_fixed_openings_swapped_colors_renju_v1' if suite == 'confirmation' else
                'four_fixed_openings_swapped_colors_renju_v1' if renju else
                'four_fixed_openings_swapped_colors_v1')
    opponent_sha = digest(ROOT/'tools'/'greedy_bot.py')
    opponent_path = None
    if opponent == 'initial':
        root = checkpoint.parent.parent
        run_dir = root / 'training' if (root / 'training').is_dir() else root
        opponent_path = root / 'weights' / 'initial_model.pt'
        if not opponent_path.exists():
            opponent_path = run_dir / 'weights' / 'initial_model.pt'
        if not opponent_path.exists():
            opponent_path = root / 'initial_model.pt'
        manifest_path = root / 'weights' / 'initial_model_manifest.json'
        if not manifest_path.exists():
            manifest_path = run_dir / 'initial_model_manifest.json'
        if not manifest_path.exists():
            # Fresh runs have no weight transfer; the immutable reference is the
            # random step-0 model described by initial_model_manifest.json.
            manifest_path = root / 'initial_model_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        opponent_sha = digest(opponent_path)
        if opponent_sha != manifest['seed_sha256']:
            raise ValueError('Initial reference model checksum changed.')
    identity = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                'checkpoint_metadata_sha256': digest(checkpoint.with_suffix('.json')),
                'simulations': sims, 'forced_move_pruning': False, 'cpu_threads': 1,
                'opponent': 'fixed_greedy_reference_v1' if opponent == 'greedy' else 'immutable_initial_model',
                'opponent_sha256': opponent_sha, 'protocol': protocol,
                'ruleset_id': 'renju_forbidden_v1' if renju else 'freestyle_v1',
                'openings_sha256': opening_sha, 'expected_games': len(openings) * 2,
                'engine_source_sha256': {name: digest(ROOT/name) for name in
                                         ('mcts.py', 'game.py', 'renju_rules.py', 'network.py')},
                'search': {'c_puct': config.C_PUCT, 'action_radius': config.ACTION_RADIUS,
                           'root_noise': False, 'sample_moves': False}}
    path = result_path(checkpoint, sims, pairs, opponent, suite)
    partial = path.with_suffix('.partial.json')
    previous = path if path.exists() else partial
    games = []
    if previous.exists():
        saved = json.loads(previous.read_text(encoding='utf-8'))
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError('Existing evaluation identity differs; preserve and investigate it.')
        games = saved['games']
        if len(games) > identity['expected_games']:
            raise ValueError('Too many saved arena games.')
        for index, row in enumerate(games):
            if row['opening'] != index//2 or row['ai_color'] != (1 if index%2 == 0 else -1):
                raise ValueError('Saved arena order changed.')
            replay = opening_game(row['actions'], renju)
            winner = replay.get_game_ended()
            if winner is None or winner != row['winner'] or row['actions'][:len(openings[index//2])] != openings[index//2]:
                raise ValueError('Saved arena trajectory failed replay.')
            expected = 0 if winner == 0 else 1 if winner == row['ai_color'] else -1
            if row['ai_result'] != expected:
                raise ValueError('Saved arena outcome is inconsistent.')
        if path.exists():
            if saved.get('status') != 'complete' or len(games) != identity['expected_games']:
                raise ValueError('Final arena result is incomplete.')
            return saved

    candidate = engine(checkpoint)
    reference = engine(opponent_path) if opponent_path else None
    for index in range(len(games), len(openings) * 2):
        assert_evaluation_allowed(checkpoint.parent.parent, review_only)
        opening_index, ai_color = index//2, 1 if index%2 == 0 else -1
        actions = list(openings[opening_index])
        game = opening_game(actions, renju)
        bot = GreedyBot(seed=20261031 + opening_index)
        while game.get_game_ended() is None:
            assert_evaluation_allowed(checkpoint.parent.parent, review_only)
            if game.current_player == ai_color:
                action = candidate.search(game)[2]
            else:
                action = reference.search(game)[2] if reference else bot.choose(game)
            if action < 0 or (action//game.board_size, action%game.board_size) not in game.get_valid_moves():
                raise RuntimeError('Evaluation engine returned an invalid move.')
            actions.append(int(action))
            game.do_move(action)
        winner = int(game.get_game_ended())
        games.append({'opening': opening_index, 'ai_color': ai_color, 'winner': winner,
                      'ai_result': 0 if winner == 0 else 1 if winner == ai_color else -1, 'actions': actions})
        atomic_json(partial, {**identity, 'status': 'in_progress', 'evaluated_at': time.time(),
                              'games': games, **summarize(games)})
        print(json.dumps({'game': len(games), 'of': len(openings)*2,
                          'result': games[-1]['ai_result'], 'moves': len(actions)}), flush=True)
    result = {**identity, 'status': 'complete', 'evaluated_at': time.time(), 'games': games, **summarize(games)}
    if suite == 'confirmation':
        interval = result.get('paired_bootstrap_95_score_interval', [0, 1])
        result['target_passed'] = (result['score_rate'] >= (0.7 if opponent == 'greedy' else 0.6)
                                   and all(s['score_rate'] >= 0.5 for s in result['by_color'].values())
                                   and interval[0] > 0.5)
    atomic_json(path, result)
    partial.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--sims', type=int, default=128)
    parser.add_argument('--pairs', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--opponent', choices=['greedy', 'initial'], default='greedy')
    parser.add_argument('--suite', choices=['progress', 'confirmation'], default='progress')
    parser.add_argument('--review-only', action='store_true', help='Evaluate under the explicit stage-review hold; never bypass a user STOP.')
    args = parser.parse_args()
    if args.sims < 1:
        parser.error('--sims must be positive')
    if args.suite == 'confirmation' and args.sims != 256:
        parser.error('Confirmation protocol uses exactly 256 simulations per model move.')
    checkpoint = args.checkpoint.resolve()
    path = result_path(checkpoint, args.sims, args.pairs, args.opponent, args.suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as locks:
        if args.review_only:
            locks.enter_context(RunLock(ROOT / '.training.lock'))
        locks.enter_context(RunLock(path.parent / '.evaluation.lock'))
        assert_evaluation_allowed(checkpoint.parent.parent, args.review_only)
        result = evaluate(checkpoint, args.sims, args.pairs, args.opponent, args.suite, args.review_only)
    print(json.dumps({k: v for k, v in result.items() if k != 'games'}, indent=2))


if __name__ == '__main__':
    main()
