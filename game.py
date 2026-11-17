import numpy as np
from config import config
from renju_rules import RenjuRules

def _shift(mask, dr, dc):
    """Shift a boolean mask by (dr, dc), filling with False."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    src_r = slice(max(0, -dr), h - max(0, dr))
    dst_r = slice(max(0, dr), h - max(0, -dr))
    src_c = slice(max(0, -dc), w - max(0, dc))
    dst_c = slice(max(0, dc), w - max(0, -dc))
    out[dst_r, dst_c] = mask[src_r, src_c]
    return out

class GomokuGame:
    def __init__(self, board_size=config.BOARD_SIZE, n_in_row=config.N_IN_ROW, renju=None):
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.renju = config.ENABLE_RENJU if renju is None else bool(renju)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1 # 1: Black, -1: White
        self.last_move = None
        self.move_count = 0

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.last_move = None
        self.move_count = 0
        return self

    def clone(self):
        """Fast copy used by MCTS: the search must never mutate the real game."""
        g = GomokuGame.__new__(GomokuGame)
        g.board_size = self.board_size
        g.n_in_row = self.n_in_row
        g.renju = self.renju
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.last_move = self.last_move
        g.move_count = self.move_count
        return g

    def get_board_state(self, player, last_move):
        # 3 planes: [player stones, opponent stones, last move]
        board_state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        # Canonical form for NN: Plane 0 is ALWAYS current player, Plane 1 is opponent
        if player == 1:
            board_state[0] = (self.board == 1)
            board_state[1] = (self.board == -1)
        else:
            board_state[0] = (self.board == -1)
            board_state[1] = (self.board == 1)
            
        if last_move is not None:
            board_state[2, last_move[0], last_move[1]] = 1
        return board_state

    def get_valid_moves(self):
        """All legal moves for the player to move."""
        if not self.renju:
            rows, cols = np.where(self.board == 0)
            return list(zip(rows.tolist(), cols.tolist()))

        valid_indices = []
        rules = RenjuRules(self.board)
        rows, cols = np.where(self.board == 0)
        for r, c in zip(rows, cols):
            if self.current_player == 1:  # Black: Check forbidden moves
                if rules.forbidden_reason(r, c) is None:
                    valid_indices.append((r, c))
            else:  # White: No restrictions
                valid_indices.append((r, c))
        return valid_indices

    def get_candidate_moves(self, radius=None):
        """Legal moves worth searching: empty points near existing stones.

        In Gomoku a move far away from every stone is essentially never best,
        so restricting the search to a neighbourhood removes 80% of the action
        space. This is a search-time optimisation only - the rules still allow
        every empty point.
        """
        if radius is None:
            radius = getattr(config, "ACTION_RADIUS", 2)

        if self.move_count == 0:
            return [(self.board_size // 2, self.board_size // 2)]

        occupied = self.board != 0
        if radius is not None and radius >= 0 and occupied.any():
            grown = occupied.copy()
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    grown |= _shift(occupied, dr, dc)
            mask = (~occupied) & grown
        else:
            mask = ~occupied

        rows, cols = np.where(mask)
        if len(rows) == 0:
            return self.get_valid_moves()
        if self.renju and self.current_player == 1:
            rules = RenjuRules(self.board)
            moves = [(r, c) for r, c in zip(rows.tolist(), cols.tolist())
                     if rules.forbidden_reason(r, c) is None]
            return moves or self.get_valid_moves()
        return list(zip(rows.tolist(), cols.tolist()))

    def do_move(self, move_idx):
        if not 0 <= move_idx < self.board_size * self.board_size:
            raise ValueError('Move is outside the board.')
        r, c = move_idx // self.board_size, move_idx % self.board_size
        if self.board[r, c] != 0:
            raise ValueError('Intersection is occupied.')
        self.board[r, c] = self.current_player
        self.last_move = (r, c)
        self.current_player = -self.current_player
        self.move_count += 1
    
    def check_win(self, move=None):
        if move is None:
            if self.last_move is None: return False
            r, c = self.last_move
        else:
            r, c = move
        
        player = self.board[r, c]
        if player == 0: return False

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Forward
            for i in range(1, self.n_in_row + 2):
                nr, nc = r + i*dr, c + i*dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                else: break
            # Backward
            for i in range(1, self.n_in_row + 2):
                nr, nc = r - i*dr, c - i*dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and self.board[nr, nc] == player:
                    count += 1
                else: break
            
            if count >= self.n_in_row:
                # Free-style Gomoku: five or more in a row wins for both sides.
                # Renju (black exactly five, white five-or-more) is only applied
                # when this game uses the Renju rule engine.
                if not self.renju:
                    return True
                if player == 1:
                    if count == self.n_in_row:
                        return True
                else:
                    return True
        
        return False
    
    def get_game_ended(self):
        # Returns: 1 if Black wins, -1 if White wins, 0 if Draw, None if not ended
        if self.last_move and self.check_win(): 
            return self.board[self.last_move[0], self.last_move[1]]

        if self.renju and self.last_move and self.board[self.last_move] == 1:
            if self.check_forbidden(*self.last_move):
                return -1
        
        if self.move_count >= self.board_size * self.board_size: return 0
        return None

    def find_immediate_wins(self, player):
        """All empty points where ``player`` would make five right now.

        Used only by evaluation opponents and diagnostics. AlphaZero search
        does not call this helper or receive its suggested moves.
        """
        wins = []
        rows, cols = np.where(self.board == 0)
        for r, c in zip(rows.tolist(), cols.tolist()):
            self.board[r, c] = player
            won = self.check_win(move=(r, c))
            self.board[r, c] = 0
            if won:
                wins.append((r, c))
        return wins

    def forbidden_reason(self, r, c):
        return RenjuRules(self.board).forbidden_reason(r, c)

    def check_forbidden(self, r, c):
        """Black rule legality, for an empty point or the last placed black stone."""
        return self.forbidden_reason(r, c) is not None

    def check_overline(self, r, c):
        return max(RenjuRules(self.board).lengths((r, c))) > 5

    def check_double_four(self, r, c):
        return len(RenjuRules(self.board).fours((r, c))) >= 2

    def check_double_three(self, r, c):
        return len(RenjuRules(self.board).threes((r, c))) >= 2
