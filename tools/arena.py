# tools/arena.py
#
# Play two checkpoints against each other with the real search. Used to answer
# "is the new model actually stronger?" without the web UI.
#
#     python tools/arena.py --a outputs/model_weights/model_weights_step_8250.pt \
#                           --b outputs/model_weights/model_weights_step_0.pt \
#                           --games 10 --sims 64
#
import argparse
import json
import os
import sys
from pathlib import Path
from queue import Empty
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import config
from game import GomokuGame
from mcts import AlphaZeroMCTS
from network import GomokuNetEZ


class LocalInference:
    """Synchronous inference for one model (same protocol as the server)."""

    def __init__(self, model):
        self.model = model
        self.result = None

    def put(self, request):
        _, kind, data = request
        with torch.no_grad():
            if kind != 'initial':
                raise ValueError(f'Unsupported inference request: {kind}')
            obs = torch.from_numpy(data).unsqueeze(0).to(config.DEVICE)
            p, v, _ = self.model.initial_inference(obs)
            self.result = (p.cpu().numpy()[0], v.cpu().numpy()[0, 0], None)

    def get(self, timeout=None):
        if self.result is None:
            raise Empty
        out, self.result = self.result, None
        return out

    def get_nowait(self):
        raise Empty


def load_model(path):
    params = json.loads(Path(path).with_suffix('.json').read_text(encoding='utf-8'))
    config.ENABLE_RENJU = bool(params.get('ENABLE_RENJU', config.ENABLE_RENJU))
    config.RULESET_ID = params.get('RULESET_ID', config.RULESET_ID)
    model = GomokuNetEZ(SimpleNamespace(**params)).to(config.DEVICE)
    model.load_state_dict(torch.load(path, map_location=config.DEVICE, weights_only=True))
    model.eval()
    return model


def make_engine(path):
    if path == "greedy":
        from tools.greedy_bot import GreedyBot

        class GreedyPlayer:
            def __init__(self):
                self.bot = GreedyBot()

            def search(self, game):
                return None, 0.0, self.bot.choose(game)

        return GreedyPlayer()
    server = LocalInference(load_model(path))
    return AlphaZeroMCTS(0, server, server, add_root_noise=False, sample_moves=False)


def play(black_engine, white_engine, max_moves=225):
    game = GomokuGame()
    while game.get_game_ended() is None and game.move_count < max_moves:
        engine = black_engine if game.current_player == 1 else white_engine
        _, _, action = engine.search(game)
        if action is None or action < 0:
            break
        game.do_move(int(action))
    return game.get_game_ended()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=64)
    ap.add_argument("--radius", type=int, default=None, help="override ACTION_RADIUS for the search")
    args = ap.parse_args()

    config.NUM_SIMULATIONS = args.sims
    if args.radius is not None:
        config.ACTION_RADIUS = args.radius
    engine_a = make_engine(args.a)
    engine_b = make_engine(args.b)

    wins_a = wins_b = draws = 0
    for i in range(args.games):
        a_is_black = (i % 2 == 0)
        black, white = (engine_a, engine_b) if a_is_black else (engine_b, engine_a)
        winner = play(black, white)
        if winner is None or winner == 0:
            draws += 1
            tag = "draw"
        else:
            a_won = (winner == 1) == a_is_black
            if a_won:
                wins_a += 1
                tag = "A wins"
            else:
                wins_b += 1
                tag = "B wins"
        print(f"game {i + 1}/{args.games}: {tag} (A={os.path.basename(args.a)}, B={os.path.basename(args.b)})")

    n = args.games
    print(f"\nA: {wins_a}W {wins_b}L {draws}D  ({wins_a / n * 100:.0f}% win rate, sims={args.sims})")


if __name__ == "__main__":
    main()
