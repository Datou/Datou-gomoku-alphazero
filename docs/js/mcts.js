// AlphaZero PUCT search for the browser player. Inference is asynchronous
// (ONNX Runtime Web), so each simulation awaits one forward pass.
const BOARD_AREA = 15 * 15;

class MCTSNode {
  constructor(parent, action, game) {
    this.parent = parent;
    this.action = action;
    this.game = game;
    this.expanded = false;
    this.terminal = false;
    this.terminalValue = 0;
    this.initialValue = 0;
    this.actions = null;
    this.priors = null;
    this.policyIndex = null;
    this.children = new Map();
    this.childVisits = null;
    this.childValues = null;
    this.visits = 0;
    this.valueSum = 0;
  }

  meanValue() {
    return this.visits > 0 ? this.valueSum / this.visits : 0;
  }
}

class AlphaZeroMCTS {
  constructor(session, options = {}) {
    this.session = session;
    this.simulations = options.simulations || 64;
    this.cPuct = options.cPuct || 2.0;
    this.radius = options.radius === undefined ? 2 : options.radius;
    this.inputName = session.inputNames ? session.inputNames[0] : 'obs';
    this.policyName = null;
    this.valueName = null;
    if (session.outputNames) {
      for (const name of session.outputNames) {
        if (name.toLowerCase().includes('policy')) this.policyName = name;
        if (name.toLowerCase().includes('value')) this.valueName = name;
      }
    }
  }

  async infer(game) {
    const observation = game.getBoardState(game.currentPlayer, game.lastMove);
    const tensor = new ort.Tensor('float32', observation, [1, 3, 15, 15]);
    const results = await this.session.run({ [this.inputName]: tensor });
    const policyTensor = this.policyName ? results[this.policyName] : results.policy;
    const valueTensor = this.valueName ? results[this.valueName] : results.value;
    if (!policyTensor || !valueTensor) throw new Error('model did not return policy and value');
    const policy = policyTensor.data;
    const value = Number(valueTensor.data[0]);
    return { policy, value: Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0 };
  }

  _expand(node, policy) {
    if (node.terminal) return;
    const moves = node.game.getCandidateMoves(this.radius);
    if (!moves.length) {
      node.terminal = true;
      node.terminalValue = 0;
      node.expanded = true;
      return;
    }
    let maxLogit = -Infinity;
    for (const action of moves) {
      const logit = policy[action];
      if (Number.isFinite(logit) && logit > maxLogit) maxLogit = logit;
    }
    if (!Number.isFinite(maxLogit)) maxLogit = 0;
    const priors = new Float64Array(moves.length);
    let sum = 0;
    for (let i = 0; i < moves.length; i += 1) {
      const logit = policy[moves[i]];
      const weight = Number.isFinite(logit) ? Math.exp(logit - maxLogit) : 0;
      priors[i] = weight;
      sum += weight;
    }
    if (!(sum > 0)) {
      for (let i = 0; i < moves.length; i += 1) priors[i] = 1 / moves.length;
    } else {
      for (let i = 0; i < moves.length; i += 1) priors[i] /= sum;
    }
    node.expanded = true;
    node.actions = moves;
    node.priors = priors;
    node.policyIndex = new Map();
    for (let i = 0; i < moves.length; i += 1) node.policyIndex.set(moves[i], i);
    node.childVisits = new Int32Array(moves.length);
    node.childValues = new Float64Array(moves.length);
  }

  _select(node) {
    const total = Math.max(1, node.visits);
    const sqrtTotal = Math.sqrt(total);
    let bestIndex = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < node.actions.length; i += 1) {
      const visits = node.childVisits[i];
      const q = visits > 0 ? node.childValues[i] / visits : 0;
      const u = (this.cPuct * node.priors[i] * sqrtTotal) / (1 + visits);
      const score = q + u;
      if (score > bestScore) {
        bestScore = score;
        bestIndex = i;
      }
    }
    return node.actions[bestIndex];
  }

  async search(game, simulations) {
    const budget = simulations || this.simulations;
    const emptyPolicy = new Float32Array(BOARD_AREA);
    if (game.getGameEnded() !== null) {
      return { action: -1, policy: emptyPolicy, value: 0, simulations: 0 };
    }
    const root = new MCTSNode(null, -1, game.clone());
    const first = await this.infer(root.game);
    this._expand(root, first.policy);
    root.initialValue = first.value;
    if (root.terminal || !root.actions) {
      return { action: -1, policy: emptyPolicy, value: 0, simulations: 0 };
    }
    for (let simulation = 0; simulation < budget; simulation += 1) {
      let node = root;
      const path = [];
      while (node.expanded && !node.terminal) {
        const action = this._select(node);
        path.push(node);
        let child = node.children.get(action);
        if (!child) {
          const next = node.game.clone();
          next.doMove(action);
          child = new MCTSNode(node, action, next);
          node.children.set(action, child);
          const ended = next.getGameEnded();
          if (ended !== null) {
            child.terminal = true;
            child.terminalValue = ended * next.currentPlayer;
            child.expanded = true;
          } else {
            const inference = await this.infer(next);
            this._expand(child, inference.policy);
            child.initialValue = inference.value;
          }
          node = child;
          break;
        }
        node = child;
      }
      if (node !== root) path.push(node);
      let value;
      if (node.terminal) value = node.terminalValue;
      else if (node.visits > 0) value = node.meanValue();
      else value = node.initialValue;
      for (let i = path.length - 1; i >= 0; i -= 1) {
        const current = path[i];
        current.visits += 1;
        current.valueSum += value;
        if (current.parent) {
          const index = current.parent.policyIndex.get(current.action);
          if (index !== undefined) {
            current.parent.childVisits[index] += 1;
            current.parent.childValues[index] += value;
          }
        }
      }
      if (simulation % 8 === 7) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const policy = new Float64Array(BOARD_AREA);
    let total = 0;
    let bestAction = root.actions[0];
    let bestVisits = -1;
    for (let i = 0; i < root.actions.length; i += 1) {
      const visits = root.childVisits[i];
      total += visits;
      policy[root.actions[i]] = visits;
      if (visits > bestVisits) {
        bestVisits = visits;
        bestAction = root.actions[i];
      }
    }
    if (total > 0) {
      for (let i = 0; i < BOARD_AREA; i += 1) if (policy[i]) policy[i] /= total;
    } else {
      for (let i = 0; i < root.actions.length; i += 1) policy[root.actions[i]] = root.priors[i];
    }
    return {
      action: bestAction,
      policy,
      value: root.meanValue(),
      simulations: budget,
    };
  }
}

if (typeof module !== 'undefined') {
  module.exports = { AlphaZeroMCTS, MCTSNode };
}
