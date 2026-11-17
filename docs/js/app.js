(() => {
  const SIZE = 15;
  const BOARD_PX = 600;
  const MARGIN = 38;
  const GRID = BOARD_PX - MARGIN * 2;
  const CELL = GRID / (SIZE - 1);
  const STAR_POINTS = [[3, 3], [3, 11], [11, 3], [11, 11], [7, 7]];
  const LETTERS = 'ABCDEFGHIJKLMNO';

  const MODELS = {
    '20500': { url: 'models/alphazero_20500.onnx', label: 'alphazero_20500', size: 10178067 },
    '27702': { url: 'models/alphazero_27702.onnx', label: 'alphazero_27702', size: 10178067 },
  };

  const REASON_TEXT = {
    overline: '长连禁手',
    double_four: '四四禁手',
    double_three: '三三禁手',
  };

  const canvas = document.getElementById('board');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('status');
  const statusSubEl = document.getElementById('status-sub');
  const progressEl = document.getElementById('load-progress');
  const progressTrackEl = document.getElementById('load-progress-track');
  const progressBarEl = document.getElementById('load-progress-bar');
  const progressTextEl = document.getElementById('load-progress-text');
  const backendEl = document.getElementById('backend');
  const simsLabelEl = document.getElementById('sims-label');
  const modelSelect = document.getElementById('model-select');
  const colorSelect = document.getElementById('color-select');
  const simsSelect = document.getElementById('sims-select');
  const newGameButton = document.getElementById('new-game');
  const confirmModal = document.getElementById('confirm-modal');
  const confirmYes = document.getElementById('confirm-yes');
  const confirmNo = document.getElementById('confirm-no');

  const state = {
    game: new RenjuGame(),
    humanColor: 1,
    simulations: 128,
    modelKey: '20500',
    session: null,
    mcts: null,
    backend: '',
    thinking: false,
    over: false,
    generation: 0,
  };

  function setStatus(main, sub = '', cls = '') {
    statusEl.className = 'status' + (cls ? ' ' + cls : '');
    statusEl.textContent = main;
    if (sub) {
      const span = document.createElement('span');
      span.className = 'sub';
      span.textContent = sub;
      statusEl.appendChild(span);
    }
    updateTurnHighlight();
  }

  function updateTurnHighlight() {
    const isYourTurn = Boolean(state.session) && !state.over && !state.thinking
      && state.game.currentPlayer === state.humanColor;
    statusEl.classList.toggle('your-turn', isYourTurn);
  }

  function setBusy(busy) {
    state.thinking = busy;
    canvas.classList.toggle('locked', busy);
    modelSelect.disabled = busy;
    colorSelect.disabled = busy;
    simsSelect.disabled = busy;
    newGameButton.disabled = busy;
    updateTurnHighlight();
  }

  function setProgress(fraction, text, indeterminate = false) {
    if (fraction === null && !indeterminate) {
      progressEl.hidden = true;
      progressEl.classList.remove('indeterminate');
      progressBarEl.style.width = '0%';
      progressTrackEl.setAttribute('aria-valuenow', '0');
      return;
    }
    progressEl.hidden = false;
    progressEl.classList.toggle('indeterminate', indeterminate);
    if (indeterminate) {
      progressBarEl.style.width = '35%';
      progressTrackEl.removeAttribute('aria-valuenow');
    } else {
      const percent = Math.max(0, Math.min(1, fraction));
      progressBarEl.style.width = `${percent * 100}%`;
      progressTrackEl.setAttribute('aria-valuenow', String(Math.round(percent * 100)));
    }
    progressTextEl.textContent = text || '';
  }

  async function fetchModelWithProgress(url, expectedTotal, onProgress) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`模型下载失败（HTTP ${response.status}）`);
    }
    const headerTotal = Number(response.headers.get('Content-Length')) || 0;
    // GitHub Pages may omit Content-Length or report the compressed transfer
    // size while fetch hands us decompressed bytes. Prefer the known ONNX size.
    const total = expectedTotal || headerTotal || 0;
    if (!response.body || total <= 0) {
      onProgress(null, 0, 0);
      const buffer = await response.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      onProgress(1, bytes.byteLength, bytes.byteLength);
      return bytes;
    }
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(Math.min(received / total, 0.99), received, total);
    }
    const bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.length;
    }
    onProgress(1, received, total);
    return bytes;
  }

  function updateMeta() {
    backendEl.textContent = state.backend || '—';
    simsLabelEl.textContent = String(state.simulations);
  }

  function setupCanvas() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(BOARD_PX * dpr);
    canvas.height = Math.round(BOARD_PX * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    render();
  }

  function gridX(column) {
    return MARGIN + column * CELL;
  }

  function gridY(row) {
    return MARGIN + row * CELL;
  }

  function drawBoard() {
    const background = ctx.createLinearGradient(0, 0, BOARD_PX, BOARD_PX);
    background.addColorStop(0, '#e7c37c');
    background.addColorStop(0.55, '#dcae5c');
    background.addColorStop(1, '#c99545');
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, BOARD_PX, BOARD_PX);

    ctx.save();
    ctx.strokeStyle = 'rgba(109, 75, 28, 0.85)';
    ctx.lineWidth = 1;
    for (let i = 0; i < SIZE; i += 1) {
      const x = gridX(i);
      const y = gridY(i);
      ctx.beginPath();
      ctx.moveTo(gridX(0), y);
      ctx.lineTo(gridX(SIZE - 1), y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, gridY(0));
      ctx.lineTo(x, gridY(SIZE - 1));
      ctx.stroke();
    }
    ctx.lineWidth = 1.8;
    ctx.strokeStyle = 'rgba(92, 62, 22, 0.95)';
    ctx.strokeRect(gridX(0), gridY(0), GRID, GRID);

    ctx.fillStyle = 'rgba(92, 62, 22, 0.95)';
    for (const [row, column] of STAR_POINTS) {
      ctx.beginPath();
      ctx.arc(gridX(column), gridY(row), 3.2, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.fillStyle = 'rgba(92, 62, 22, 0.7)';
    ctx.font = '11px "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < SIZE; i += 1) {
      ctx.fillText(LETTERS[i], gridX(i), BOARD_PX - 13);
      ctx.fillText(String(i + 1), 13, gridY(i));
    }
    ctx.restore();
  }

  function drawStone(row, column, player, alpha = 1) {
    const x = gridX(column);
    const y = gridY(row);
    const radius = CELL * 0.43;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.shadowColor = 'rgba(35, 24, 8, 0.45)';
    ctx.shadowBlur = 6;
    ctx.shadowOffsetY = 2.5;
    const gradient = ctx.createRadialGradient(
      x - radius * 0.35,
      y - radius * 0.4,
      radius * 0.12,
      x,
      y,
      radius,
    );
    if (player === 1) {
      gradient.addColorStop(0, '#5b6672');
      gradient.addColorStop(0.45, '#252c35');
      gradient.addColorStop(1, '#05080b');
    } else {
      gradient.addColorStop(0, '#ffffff');
      gradient.addColorStop(0.55, '#f2f4f6');
      gradient.addColorStop(1, '#c3c9d0');
    }
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.lineWidth = 1;
    ctx.strokeStyle = player === 1 ? 'rgba(0, 0, 0, 0.55)' : 'rgba(90, 98, 108, 0.45)';
    ctx.stroke();
    ctx.restore();
  }

  function drawLastMove(row, column) {
    const x = gridX(column);
    const y = gridY(row);
    ctx.save();
    ctx.strokeStyle = '#e5484d';
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.arc(x, y, CELL * 0.16, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function render() {
    ctx.clearRect(0, 0, BOARD_PX, BOARD_PX);
    drawBoard();
    const board = state.game.board;
    for (let index = 0; index < board.length; index += 1) {
      if (board[index] === 0) continue;
      const row = Math.floor(index / SIZE);
      const column = index % SIZE;
      drawStone(row, column, board[index]);
    }
    if (state.game.lastMove) drawLastMove(state.game.lastMove[0], state.game.lastMove[1]);
  }

  function pointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (BOARD_PX / rect.width);
    const y = (event.clientY - rect.top) * (BOARD_PX / rect.height);
    const column = Math.round((x - MARGIN) / CELL);
    const row = Math.round((y - MARGIN) / CELL);
    if (row < 0 || row >= SIZE || column < 0 || column >= SIZE) return null;
    const dx = x - gridX(column);
    const dy = y - gridY(row);
    if (Math.hypot(dx, dy) > CELL * 0.5) return null;
    return row * SIZE + column;
  }

  function turnLabel() {
    if (state.over) return '对局结束';
    if (state.thinking) return 'AI 思考中…';
    if (state.game.currentPlayer === state.humanColor) {
      return `轮到你（${state.humanColor === 1 ? '黑棋' : '白棋'}）`;
    }
    return 'AI 回合';
  }

  function finish(winner) {
    state.over = true;
    state.thinking = false;
    canvas.classList.remove('locked');
    let text = '平局';
    let cls = '';
    if (winner === state.humanColor) {
      text = '你赢了！';
      cls = 'win';
    } else if (winner === -state.humanColor) {
      text = 'AI 获胜';
    }
    setStatus(text, `共 ${state.game.moveCount} 手`, cls);
    updateMeta();
  }

  async function aiTurn() {
    if (state.over || state.thinking || !state.mcts) return;
    setProgress(null);
    const generation = state.generation;
    const started = performance.now();
    setBusy(true);
    setStatus('AI 思考中…', `${state.simulations} 次模拟`);
    try {
      const result = await state.mcts.search(state.game, state.simulations);
      if (generation !== state.generation || state.over) return;
      if (result.action < 0) {
        finish(0);
        return;
      }
      state.game.doMove(result.action);
      render();
      const ended = state.game.getGameEnded();
      if (ended !== null) {
        finish(ended);
        return;
      }
      const seconds = ((performance.now() - started) / 1000).toFixed(1);
      setBusy(false);
      setStatus(turnLabel(), `AI 用时 ${seconds}s · 局面估值 ${result.value.toFixed(2)}`);
    } catch (error) {
      console.error(error);
      if (generation === state.generation) {
        setStatus('AI 推理失败', String(error && error.message ? error.message : error), 'error');
      }
    } finally {
      if (generation === state.generation) setBusy(false);
    }
  }

  function onBoardClick(event) {
    if (!state.session || state.over || state.thinking) return;
    if (state.game.currentPlayer !== state.humanColor) return;
    const index = pointFromEvent(event);
    if (index === null || state.game.board[index] !== 0) return;
    const row = Math.floor(index / SIZE);
    const column = index % SIZE;
    let forbidden = null;
    if (state.humanColor === 1) forbidden = state.game.forbiddenReason(row, column);
    state.game.doMove(index);
    render();
    if (forbidden) {
      state.over = true;
      setStatus(`黑方${REASON_TEXT[forbidden] || '禁手'}，白方获胜`, '禁手落子立即判负', 'error');
      updateMeta();
      return;
    }
    const ended = state.game.getGameEnded();
    if (ended !== null) {
      finish(ended);
      return;
    }
    aiTurn();
  }

  function newGame() {
    closeConfirmModal();
    setProgress(null);
    state.generation += 1;
    state.game = new RenjuGame();
    state.over = false;
    state.thinking = false;
    render();
    updateMeta();
    if (!state.session) {
      setStatus('正在加载模型…', '首次加载约 10–30 秒');
      return;
    }
    state.mcts = new AlphaZeroMCTS(state.session, { simulations: state.simulations });
    setStatus(turnLabel(), state.humanColor === 1 ? '你执黑先行' : 'AI 执黑先行');
    if (state.humanColor === -1) aiTurn();
  }

  function openConfirmModal() {
    confirmModal.hidden = false;
    confirmNo.focus();
  }

  function closeConfirmModal() {
    confirmModal.hidden = true;
  }

  function requestNewGame() {
    if (!state.session || state.over || state.game.moveCount === 0) {
      newGame();
      return;
    }
    openConfirmModal();
  }

  async function loadModel(key) {
    const config = MODELS[key];
    if (!config) return;
    state.modelKey = key;
    state.generation += 1;
    state.session = null;
    state.mcts = null;
    state.over = false;
    state.thinking = false;
    setBusy(true);
    setStatus('正在加载模型…', `${config.label} · 准备下载`);
    setProgress(0, '准备下载模型…');
    let lastPercent = -1;
    try {
      if (typeof ort === 'undefined') {
        throw new Error('ONNX Runtime Web 加载失败，请检查静态资源是否完整');
      }
      if (ort.env && ort.env.wasm) {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
        ort.env.wasm.wasmPaths = new URL('vendor/', window.location.href).href;
      }
      if (ort.env) ort.env.logLevel = 'error';

      const modelBytes = await fetchModelWithProgress(config.url, config.size, (fraction, received, total) => {
        if (fraction === null || !total) {
          setProgress(null, `下载 ${config.label}…`, true);
          return;
        }
        const percent = Math.round(fraction * 100);
        if (percent === lastPercent) return;
        lastPercent = percent;
        const receivedMb = (received / 1048576).toFixed(1);
        const totalMb = (total / 1048576).toFixed(1);
        setStatus('正在加载模型…', `${config.label} · ${receivedMb} / ${totalMb} MB`);
        setProgress(fraction, `下载模型 ${receivedMb} / ${totalMb} MB · ${percent}%`);
      });

      setStatus('正在加载模型…', `${config.label} · 正在初始化推理引擎`);
      setProgress(null, '正在初始化推理引擎…', true);

      let session = null;
      let backend = '';
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        try {
          session = await ort.InferenceSession.create(modelBytes, {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
          });
          backend = 'WebGPU';
        } catch (error) {
          console.warn('WebGPU 不可用，回退到 WASM', error);
          session = null;
        }
      }
      if (!session) {
        session = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        backend = 'WASM（单线程）';
      }
      state.session = session;
      state.backend = backend;
      state.mcts = new AlphaZeroMCTS(session, { simulations: state.simulations });
      updateMeta();
      setProgress(1, '模型加载完成');
      newGame();
    } catch (error) {
      console.error(error);
      state.session = null;
      state.mcts = null;
      backendEl.textContent = '—';
      setProgress(null);
      setStatus('模型加载失败', String(error && error.message ? error.message : error), 'error');
    } finally {
      if (!state.thinking) setBusy(false);
    }
  }

  canvas.addEventListener('click', onBoardClick);
  window.addEventListener('resize', setupCanvas);
  newGameButton.addEventListener('click', requestNewGame);
  confirmYes.addEventListener('click', () => {
    closeConfirmModal();
    newGame();
  });
  confirmNo.addEventListener('click', closeConfirmModal);
  confirmModal.addEventListener('click', (event) => {
    if (event.target === confirmModal) closeConfirmModal();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !confirmModal.hidden) closeConfirmModal();
  });
  simsSelect.addEventListener('change', () => {
    state.simulations = Number(simsSelect.value);
    if (state.mcts) state.mcts.simulations = state.simulations;
    updateMeta();
    setStatus(turnLabel(), '模拟次数将在下一步生效');
  });
  colorSelect.addEventListener('change', () => {
    state.humanColor = Number(colorSelect.value);
    newGame();
  });
  modelSelect.addEventListener('change', () => {
    loadModel(modelSelect.value);
  });

  setupCanvas();
  updateMeta();
  loadModel(state.modelKey);
})();
