# tools/greedy_bot.py
#
# A fixed, hand-written opponent used as an absolute yardstick. It is roughly a
# fixed reference: it takes a five, blocks a five, and otherwise
# prefers moves that build/deny open threes and fours. No search, no learning.
#
#     from tools.greedy_bot import GreedyBot
#     bot.choose(game) -> action index
#
import numpy as np

from config import config

DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _run_info(board, r, c, dr, dc, player):
    """Length of the run through (r,c) and whether each end is open."""
    size = board.shape[0]
    count = 1
    open_ends = 0
    for sign in (1, -1):
        i = 1
        while True:
            nr, nc = r + sign * i * dr, c + sign * i * dc
            if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == player:
                count += 1
                i += 1
            else:
                if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                    open_ends += 1
                break
    return count, open_ends


def _move_score(board, r, c, player):
    """Heuristic value of placing `player` at (r,c)."""
    board[r, c] = player
    try:
        score = 0
        for dr, dc in DIRS:
            count, open_ends = _run_info(board, r, c, dr, dc, player)
            if count >= 5:
                score += 1_000_000
            elif count == 4:
                score += 100_000 if open_ends == 2 else 10_000
            elif count == 3:
                score += 5_000 if open_ends == 2 else 500
            elif count == 2:
                score += 100 if open_ends == 2 else 20
            else:
                score += 1
        return score
    finally:
        board[r, c] = 0


class GreedyBot:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def choose(self, game):
        me = game.current_player
        board = game.board

        my_wins = game.find_immediate_wins(me)
        if my_wins:
            r, c = my_wins[0]
            return r * config.BOARD_SIZE + c

        blocks = game.find_immediate_wins(-me)
        if game.renju and me == 1:
            blocks = [(r, c) for r, c in blocks if not game.check_forbidden(r, c)]
        if blocks:
            best, best_score = None, -1e18
            for r, c in blocks:
                s = _move_score(board, r, c, me) + self.rng.random()
                if s > best_score:
                    best, best_score = (r, c), s
            return best[0] * config.BOARD_SIZE + best[1]

        candidates = game.get_candidate_moves(radius=2)
        if not candidates:
            return -1
        best, best_score = None, -1e18
        for r, c in candidates:
            mine = _move_score(board, r, c, me)
            theirs = _move_score(board, r, c, -me)
            s = mine + 0.9 * theirs + self.rng.random()
            if s > best_score:
                best, best_score = (r, c), s
        return best[0] * config.BOARD_SIZE + best[1]
