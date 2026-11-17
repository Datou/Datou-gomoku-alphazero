// Regression test for the browser MCTS sign convention.
// Run with: node tests/test_web_mcts.js
const assert = require('assert');
const { AlphaZeroMCTS } = require('../docs/js/mcts.js');
const { RenjuGame } = require('../docs/js/renju.js');

function singleWinningPoint() {
  const game = new RenjuGame();
  // Black: (7,3) (7,4) (7,5) (7,6). White blocks (7,2), so only (7,7)
  // completes an exact five.
  for (const column of [3, 4, 5, 6]) game.board[7 * 15 + column] = 1;
  game.board[7 * 15 + 2] = -1;
  game.currentPlayer = 1;
  game.lastMove = [7, 6];
  game.moveCount = 5;
  return game;
}

(async () => {
  const session = { inputNames: ['obs'], outputNames: ['policy', 'value'] };
  const mcts = new AlphaZeroMCTS(session, { simulations: 128 });
  // Uniform policy and neutral value: only the terminal +1 can select the win.
  mcts.infer = async () => ({ policy: new Float32Array(225), value: 0 });
  const result = await mcts.search(singleWinningPoint(), 128);
  assert.strictEqual(result.action, 112, 'search must take the only immediate five');
  assert.ok(result.value > 0, 'a winning position should have a positive root value');
  console.log('OK: browser MCTS takes the immediate five');
})().catch((error) => {
  console.error(error);
  process.exit(1);
});

