// Renju (forbidden-move Gomoku) rules and game state, browser implementation.
// Mirrors renju_rules.py / game.py. Black must make exactly five; overline,
// double-four and double-three lose. White wins with five or more.
const SIZE = 15;

const DIRS = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

class RenjuRules {
  constructor(board) {
    this.board = board;
    this.memo = new Map();
  }

  at(r, c) {
    if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) return 2;
    return this.board[r * SIZE + c];
  }

  lineLength(r, c, dr, dc) {
    const player = this.at(r, c);
    let count = 1;
    for (const sign of [-1, 1]) {
      for (let k = 1; ; k += 1) {
        if (this.at(r + sign * k * dr, c + sign * k * dc) === player) count += 1;
        else break;
      }
    }
    return count;
  }

  lengths(r, c) {
    return DIRS.map(([dr, dc]) => this.lineLength(r, c, dr, dc));
  }

  forbiddenReason(r, c) {
    const index = r * SIZE + c;
    const previous = this.board[index];
    if (previous !== 0 && previous !== 1) throw new Error('forbidden checks need an empty or black point');
    this.board[index] = 1;
    try {
      return this._placedReason(r, c);
    } finally {
      this.board[index] = previous;
    }
  }

  _placedReason(r, c) {
    const key = this.board.join('') + '|' + r + ',' + c;
    if (this.memo.has(key)) return this.memo.get(key);
    const lengths = this.lengths(r, c);
    let reason = null;
    if (lengths.includes(5)) reason = null;
    else if (Math.max(...lengths) > 5) reason = 'overline';
    else if (this.fours(r, c).size >= 2) reason = 'double_four';
    else if (this.threes(r, c).size >= 2) reason = 'double_three';
    this.memo.set(key, reason);
    return reason;
  }

  fours(r, c) {
    const found = new Map();
    for (const [dr, dc] of DIRS) {
      for (let start = -4; start <= 0; start += 1) {
        const cells = [];
        let ones = 0;
        let zeros = 0;
        let zeroIndex = -1;
        for (let k = 0; k < 5; k += 1) {
          const rr = r + (start + k) * dr;
          const cc = c + (start + k) * dc;
          const value = this.at(rr, cc);
          cells.push([rr, cc]);
          if (value === 1) ones += 1;
          else if (value === 0) {
            zeros += 1;
            zeroIndex = k;
          }
        }
        if (ones !== 4 || zeros !== 1) continue;
        if (this.at(r + (start - 1) * dr, c + (start - 1) * dc) === 1) continue;
        if (this.at(r + (start + 5) * dr, c + (start + 5) * dc) === 1) continue;
        const stones = cells
          .filter((_, k) => k !== zeroIndex)
          .map(([rr, cc]) => rr * SIZE + cc)
          .sort((a, b) => a - b)
          .join(',');
        const emptyPoint = cells[zeroIndex][0] * SIZE + cells[zeroIndex][1];
        if (!found.has(stones)) found.set(stones, new Set());
        found.get(stones).add(emptyPoint);
      }
    }
    return found;
  }

  threes(r, c) {
    const candidates = new Map();
    for (const [dr, dc] of DIRS) {
      for (let start = -4; start <= -1; start += 1) {
        const cells = [];
        const values = [];
        for (let k = 0; k < 6; k += 1) {
          const rr = r + (start + k) * dr;
          const cc = c + (start + k) * dc;
          cells.push([rr, cc]);
          values.push(this.at(rr, cc));
        }
        if (values[0] !== 0 || values[5] !== 0) continue;
        const body = values.slice(1, 5);
        let ones = 0;
        let zeros = 0;
        let zeroIndex = -1;
        for (let k = 0; k < 4; k += 1) {
          if (body[k] === 1) ones += 1;
          else if (body[k] === 0) {
            zeros += 1;
            zeroIndex = k;
          }
        }
        if (ones !== 3 || zeros !== 1) continue;
        if (this.at(r + (start - 1) * dr, c + (start - 1) * dc) === 1) continue;
        if (this.at(r + (start + 6) * dr, c + (start + 6) * dc) === 1) continue;
        const stones = cells
          .slice(1, 5)
          .filter((_, k) => k !== zeroIndex)
          .map(([rr, cc]) => rr * SIZE + cc)
          .sort((a, b) => a - b)
          .join(',');
        const extension = cells[1 + zeroIndex];
        if (!candidates.has(stones)) candidates.set(stones, new Set());
        candidates.get(stones).add(extension[0] * SIZE + extension[1]);
      }
    }
    const found = new Set();
    for (const [stones, extensions] of candidates) {
      for (const point of extensions) {
        const er = Math.floor(point / SIZE);
        const ec = point % SIZE;
        this.board[point] = 1;
        try {
          const lengths = this.lengths(er, ec);
          if (!lengths.includes(5) && this._placedReason(er, ec) === null) {
            found.add(stones);
            break;
          }
        } finally {
          this.board[point] = 0;
        }
      }
      if (found.size >= 2) break;
    }
    return found;
  }
}

class RenjuGame {
  constructor() {
    this.board = new Int8Array(SIZE * SIZE);
    this.currentPlayer = 1;
    this.lastMove = null;
    this.moveCount = 0;
  }

  clone() {
    const copy = Object.create(RenjuGame.prototype);
    copy.board = this.board.slice();
    copy.currentPlayer = this.currentPlayer;
    copy.lastMove = this.lastMove ? [...this.lastMove] : null;
    copy.moveCount = this.moveCount;
    return copy;
  }

  at(r, c) {
    if (r < 0 || r >= SIZE || c < 0 || c >= SIZE) return 2;
    return this.board[r * SIZE + c];
  }

  doMove(index) {
    if (index < 0 || index >= SIZE * SIZE || this.board[index] !== 0) {
      throw new Error('illegal move');
    }
    const r = Math.floor(index / SIZE);
    const c = index % SIZE;
    this.board[index] = this.currentPlayer;
    this.lastMove = [r, c];
    this.currentPlayer = -this.currentPlayer;
    this.moveCount += 1;
  }

  checkWin(r, c) {
    const player = this.board[r * SIZE + c];
    if (player === 0) return false;
    for (const [dr, dc] of DIRS) {
      let count = 1;
      for (const sign of [-1, 1]) {
        for (let k = 1; ; k += 1) {
          if (this.at(r + sign * k * dr, c + sign * k * dc) === player) count += 1;
          else break;
        }
      }
      if (player === 1 && count === 5) return true;
      if (player === -1 && count >= 5) return true;
    }
    return false;
  }

  getGameEnded() {
    if (this.lastMove) {
      const [r, c] = this.lastMove;
      const stone = this.board[r * SIZE + c];
      if (this.checkWin(r, c)) return stone;
      if (stone === 1 && this.forbiddenReason(r, c) !== null) return -1;
    }
    if (this.moveCount >= SIZE * SIZE) return 0;
    return null;
  }

  forbiddenReason(r, c) {
    return new RenjuRules(this.board).forbiddenReason(r, c);
  }

  getValidMoves() {
    const moves = [];
    const rules = this.currentPlayer === 1 ? new RenjuRules(this.board) : null;
    for (let index = 0; index < SIZE * SIZE; index += 1) {
      if (this.board[index] !== 0) continue;
      if (rules && rules.forbiddenReason(Math.floor(index / SIZE), index % SIZE) !== null) continue;
      moves.push(index);
    }
    return moves;
  }

  getCandidateMoves(radius = 2) {
    if (this.moveCount === 0) return [7 * SIZE + 7];
    const candidates = new Set();
    for (let index = 0; index < SIZE * SIZE; index += 1) {
      if (this.board[index] === 0) continue;
      const r = Math.floor(index / SIZE);
      const c = index % SIZE;
      for (let dr = -radius; dr <= radius; dr += 1) {
        for (let dc = -radius; dc <= radius; dc += 1) {
          const rr = r + dr;
          const cc = c + dc;
          if (rr < 0 || rr >= SIZE || cc < 0 || cc >= SIZE) continue;
          const target = rr * SIZE + cc;
          if (this.board[target] === 0) candidates.add(target);
        }
      }
    }
    let moves = [...candidates];
    if (this.currentPlayer === 1) {
      const rules = new RenjuRules(this.board);
      moves = moves.filter((index) => rules.forbiddenReason(
        Math.floor(index / SIZE), index % SIZE) === null);
    }
    return moves.length ? moves : this.getValidMoves();
  }

  getBoardState(player, lastMove) {
    const state = new Float32Array(3 * SIZE * SIZE);
    for (let index = 0; index < SIZE * SIZE; index += 1) {
      const value = this.board[index];
      if (player === 1) {
        state[index] = value === 1 ? 1 : 0;
        state[SIZE * SIZE + index] = value === -1 ? 1 : 0;
      } else {
        state[index] = value === -1 ? 1 : 0;
        state[SIZE * SIZE + index] = value === 1 ? 1 : 0;
      }
    }
    if (lastMove) state[2 * SIZE * SIZE + lastMove[0] * SIZE + lastMove[1]] = 1;
    return state;
  }
}

if (typeof module !== 'undefined') {
  module.exports = { RenjuRules, RenjuGame, SIZE };
}
