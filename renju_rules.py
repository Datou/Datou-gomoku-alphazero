"""Black forbidden points, following RIF rules 3 and 9.1–9.3.

Only rule legality is computed here; this module never chooses an AI move.
The recursive test is necessary: a three whose straight-four extensions are
all forbidden is a false three. Every recursive call adds a stone, so there
is no depth cutoff or pattern-only approximation.
"""

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class RenjuRules:
    def __init__(self, board):
        self.board = board
        self.size = len(board)
        self._memo = {}

    def stone(self, r, c):
        return int(self.board[r, c]) if 0 <= r < self.size and 0 <= c < self.size else 2

    def line_length(self, point, direction):
        r, c = point
        dr, dc = direction
        player = self.stone(r, c)
        count = 1
        for sign in (-1, 1):
            k = 1
            while self.stone(r + sign*k*dr, c + sign*k*dc) == player:
                count += 1
                k += 1
        return count

    def lengths(self, point):
        return [self.line_length(point, direction) for direction in DIRECTIONS]

    def forbidden_reason(self, r, c):
        """Inspect an empty point or an already placed black stone; restore it."""
        old = self.stone(r, c)
        if old not in (0, 1):
            raise ValueError('Forbidden-point checks require an empty or black point.')
        self.board[r, c] = 1
        try:
            return self._placed_reason((r, c))
        finally:
            self.board[r, c] = old

    def _placed_reason(self, point):
        key = (self.board.tobytes(), point)
        if key in self._memo:
            return self._memo[key]
        lengths = self.lengths(point)
        # RIF 9.2: attaining an exact five takes precedence over all fouls.
        if 5 in lengths:
            reason = None
        elif max(lengths) > 5:
            reason = 'overline'
        elif len(self.fours(point)) >= 2:
            reason = 'double_four'
        elif len(self.threes(point)) >= 2:
            reason = 'double_three'
        else:
            reason = None
        self._memo[key] = reason
        return reason

    def fours(self, point):
        """Distinct four-stone sets meeting at point, including same-line 4-4.

        An open four's two winning endpoints belong to one four, not two.
        A completion that produces six along this line does not make a four.
        """
        r, c = point
        found = {}
        for dr, dc in DIRECTIONS:
            for start in range(-4, 1):
                cells = [(r + k*dr, c + k*dc) for k in range(start, start + 5)]
                values = [self.stone(*p) for p in cells]
                if values.count(1) != 4 or values.count(0) != 1:
                    continue
                if self.stone(r + (start-1)*dr, c + (start-1)*dc) == 1:
                    continue
                if self.stone(r + (start+5)*dr, c + (start+5)*dc) == 1:
                    continue
                stones = frozenset(p for p, value in zip(cells, values) if value == 1)
                found.setdefault(stones, set()).add(cells[values.index(0)])
        return found

    def threes(self, point):
        """Distinct real threes meeting at point, with legal straight-four extensions."""
        r, c = point
        candidates = {}
        for dr, dc in DIRECTIONS:
            # Six-cell windows .XXXX. with the point inside the four.
            for start in range(-4, 0):
                cells = [(r + k*dr, c + k*dc) for k in range(start, start + 6)]
                values = [self.stone(*p) for p in cells]
                if values[0] != 0 or values[-1] != 0:
                    continue
                body = values[1:-1]
                if body.count(1) != 3 or body.count(0) != 1:
                    continue
                # Both endpoints must complete exact five, rather than overline.
                if self.stone(r + (start-1)*dr, c + (start-1)*dc) == 1:
                    continue
                if self.stone(r + (start+6)*dr, c + (start+6)*dc) == 1:
                    continue
                stones = frozenset(p for p, value in zip(cells[1:-1], body) if value == 1)
                extension = cells[1 + body.index(0)]
                candidates.setdefault(stones, set()).add(extension)

        found = set()
        for stones, extensions in candidates.items():
            for er, ec in extensions:
                self.board[er, ec] = 1
                try:
                    # A move making five is not a three-to-straight-four extension.
                    if 5 not in self.lengths((er, ec)) and self._placed_reason((er, ec)) is None:
                        found.add(stones)
                        break
                finally:
                    self.board[er, ec] = 0
            if len(found) >= 2:
                break
        return found
