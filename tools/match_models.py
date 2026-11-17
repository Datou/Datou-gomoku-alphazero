"""Play two AlphaZero checkpoints against each other on fixed RIF openings.

Each opening is played twice with colours swapped, so a 10-opening match is 20
games. Search is deterministic (no root noise, no temperature sampling), which
makes the result a direct comparison of the two networks at a fixed budget.

    python tools/match_models.py \
        --a weights/alphazero_20500.pt \
        --b weights/alphazero_27702.pt \
        --openings 10 --sims 64 --device cpu
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import config
from game import GomokuGame
from rif_openings import RIF_OPENINGS, opening_actions
from tools.arena import make_engine


def play_game(black_engine, white_engine, opening, max_moves=225):
    game = GomokuGame(renju=True)
    for action in opening_actions(opening):
        if game.get_game_ended() is not None:
            raise ValueError('Opening reached a terminal position.')
        game.do_move(action)
    while game.get_game_ended() is None and game.move_count < max_moves:
        engine = black_engine if game.current_player == 1 else white_engine
        _, _, action = engine.search(game)
        if action is None or action < 0:
            break
        game.do_move(int(action))
    winner = game.get_game_ended()
    return (0 if winner is None else int(winner)), game.move_count


def summarize(games):
    def stats(rows):
        wins = sum(g['a_result'] == 1 for g in rows)
        draws = sum(g['a_result'] == 0 for g in rows)
        losses = sum(g['a_result'] == -1 for g in rows)
        n = len(rows)
        return {'wins': wins, 'draws': draws, 'losses': losses, 'n': n,
                'score_rate': (wins + 0.5 * draws) / n if n else None}

    result = stats(games)
    result['by_color'] = {
        str(color): stats([g for g in games if g['a_color'] == color])
        for color in (1, -1)
    }
    result['by_opening'] = {
        str(opening): stats([g for g in games if g['opening'] == opening])
        for opening in sorted({g['opening'] for g in games})
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True)
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--openings', type=int, default=10, choices=range(1, 27))
    parser.add_argument('--start-opening', type=int, default=0, choices=range(0, 26))
    parser.add_argument('--sims', type=int, default=64)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    config.DEVICE = torch.device(args.device)
    torch.set_num_threads(1)
    config.NUM_SIMULATIONS = args.sims
    config.ENABLE_RENJU = True
    config.ENABLE_FORCED_MOVE_PRUNING = False

    a = make_engine(str(args.a.resolve()))
    b = make_engine(str(args.b.resolve()))
    openings = RIF_OPENINGS[args.start_opening:args.start_opening + args.openings]

    games = []
    for local_index, opening in enumerate(openings):
        opening_index = args.start_opening + local_index
        for a_color in (1, -1):
            black, white = (a, b) if a_color == 1 else (b, a)
            started = time.time()
            winner, moves = play_game(black, white, opening)
            a_result = 0 if winner == 0 else (1 if winner == a_color else -1)
            games.append({
                'opening': opening_index,
                'opening_stones': [list(p) for p in opening],
                'a_color': a_color,
                'winner': winner,
                'a_result': a_result,
                'moves': moves,
                'seconds': round(time.time() - started, 2),
            })
            print(json.dumps({'game': len(games), 'opening': opening_index,
                              'a_color': a_color, 'a_result': a_result,
                              'moves': moves}), flush=True)

    payload = {
        'a': str(args.a.resolve()),
        'b': str(args.b.resolve()),
        'simulations': args.sims,
        'device': args.device,
        'openings': args.openings,
        'start_opening': args.start_opening,
        'ruleset_id': 'renju_forbidden_v1',
        'protocol': 'rif_openings_swapped_colors_deterministic_v1',
        'games': games,
        **summarize(games),
    }
    output = args.output or (ROOT / 'training' / 'evaluations' /
                             f'match_{args.a.stem}_vs_{args.b.stem}_sims{args.sims}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k != 'games'}, indent=2, ensure_ascii=False))
    print('saved', output)


if __name__ == '__main__':
    main()
