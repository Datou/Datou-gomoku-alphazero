# mcts.py
#
# AlphaZero PUCT search.
#
# Design rules:
#   1. Values are always stored from the point of view of the player to move at
#      that node. When a value travels up one edge, its sign is flipped.
#   2. The Q value of an edge is the negated value of the child node.
#   3. A position where the game is already over is never expanded: it is
#      scored -1 for the player to move (or 0 for a draw) and backed up.
#   4. Legal moves are computed from the position at each node, not once at the
#      root.

import logging
from abc import ABC, abstractmethod
from queue import Empty

import numpy as np

from config import config
from game import GomokuGame


class MCTS(ABC):
    def __init__(self, worker_id, request_queue, result_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.logger = logging.getLogger(f"MCTS-{self.__class__.__name__}-{worker_id}")

    @abstractmethod
    def search(self, game: GomokuGame):
        """Return (policy, value, action)."""


# =================================================================
# AlphaZero (PUCT) search
# =================================================================
class PUCTNode:
    """A node of a PUCT tree.

    ``value_sum / visit_count`` is the mean value of the position *for the
    player to move there*. ``child_values[i]`` is the mean value of taking
    ``actions[i]`` for the player to move at this node.
    """

    __slots__ = ('parent', 'action', 'actions', 'priors', 'index',
                 'children', 'visit_count', 'value_sum',
                 'child_visits', 'child_values', 'is_expanded', 'is_terminal')

    def __init__(self, parent=None, action=-1):
        self.parent = parent
        self.action = action
        self.actions = None
        self.priors = None
        self.index = None
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.child_visits = None
        self.child_values = None
        self.is_expanded = False
        self.is_terminal = False

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class AlphaZeroMCTS(MCTS):
    """Standard AlphaZero: PUCT selection, terminal values, sign-correct backup."""

    def __init__(self, worker_id, request_queue, result_queue, add_root_noise=True, sample_moves=True):
        super().__init__(worker_id, request_queue, result_queue)
        self.c_puct = float(getattr(config, 'C_PUCT', 2.0))
        self.dirichlet_alpha = float(getattr(config, 'ROOT_DIRICHLET_ALPHA', 0.3))
        self.dirichlet_eps = float(getattr(config, 'ROOT_DIRICHLET_EPSILON', 0.25))
        self.noise_top_k = int(getattr(config, 'ROOT_NOISE_TOP_K', 32))
        self.play_temperature_moves = int(getattr(config, 'PLAY_TEMPERATURE_MOVES', 20))
        self.action_radius = getattr(config, 'ACTION_RADIUS', 2)
        # Dirichlet noise and temperature sampling exist to make *self-play*
        # explore. Using them when playing a real opponent (arena / web UI)
        # randomises the opening and makes the engine measurably weaker.
        self.add_root_noise = bool(add_root_noise)
        self.sample_moves = bool(sample_moves)

    # --- Network communication (same protocol as the inference server) ---
    def _remote_initial_inference(self, obs):
        self.request_queue.put((self.worker_id, 'initial', obs))
        return self.result_queue.get(timeout=20)

    # --- Tree operations ---
    def _expand(self, node: PUCTNode, game: GomokuGame, policy_logits, candidates=None):
        moves = candidates if candidates is not None else game.get_candidate_moves(self.action_radius)
        if not moves:
            node.is_terminal = True
            return

        board_size = game.board_size
        actions = np.fromiter((r * board_size + c for r, c in moves),
                              dtype=np.int32, count=len(moves))
        logits = np.asarray(policy_logits, dtype=np.float64)[actions]
        logits = logits - logits.max()
        priors = np.exp(logits)
        total = priors.sum()
        if not np.isfinite(total) or total <= 0:
            priors = np.full(len(actions), 1.0 / len(actions))
        else:
            priors = priors / total

        node.actions = actions
        node.priors = priors.astype(np.float32)
        node.child_visits = np.zeros(len(actions), dtype=np.int32)
        node.child_values = np.zeros(len(actions), dtype=np.float32)
        node.index = {int(a): i for i, a in enumerate(actions)}
        node.is_expanded = True

        if node.parent is None and self.add_root_noise:
            self._add_root_noise(node)

    def _add_root_noise(self, node: PUCTNode):
        k = min(self.noise_top_k, len(node.priors))
        if k <= 1 or self.dirichlet_eps <= 0.0:
            return
        top = np.argsort(node.priors)[-k:]
        noise = np.random.dirichlet([self.dirichlet_alpha] * k).astype(np.float32)
        node.priors[top] = (1.0 - self.dirichlet_eps) * node.priors[top] + self.dirichlet_eps * noise
        node.priors /= node.priors.sum()

    def _select(self, node: PUCTNode) -> int:
        visits = node.child_visits
        q = node.child_values / np.maximum(visits, 1)
        sqrt_n = np.sqrt(max(node.visit_count, 1))
        u = self.c_puct * node.priors * sqrt_n / (1.0 + visits)
        return int(np.argmax(q + u))

    def _backprop(self, node: PUCTNode, value: float):
        v = float(value)
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
            parent = node.parent
            if parent is None:
                break
            # The parent is the other player: its value is the opposite.
            v = -v
            idx = parent.index.get(node.action)
            if idx is not None:
                parent.child_visits[idx] += 1
                parent.child_values[idx] += v
            node = parent

    def _simulate(self, root: PUCTNode, game: GomokuGame) -> bool:
        g = game.clone()
        node = root
        while node.is_expanded and not node.is_terminal:
            action = int(node.actions[self._select(node)])
            g.do_move(action)
            child = node.children.get(action)
            if child is None:
                child = PUCTNode(parent=node, action=action)
                node.children[action] = child
            node = child
            winner = g.get_game_ended()
            if winner is not None:
                node.is_terminal = True
                break

        if node.is_terminal:
            winner = g.get_game_ended()
            # The player to move at this node has no move left: they lost.
            value = float(winner * g.current_player) if winner is not None else -1.0
        else:
            obs = g.get_board_state(g.current_player, g.last_move)
            try:
                policy_logits, value, _ = self._remote_initial_inference(obs)
            except Empty:
                self.logger.warning(f"Worker {self.worker_id}: inference timeout in MCTS.")
                return False
            value = float(np.clip(value, -1.0, 1.0))
            self._expand(node, g, policy_logits)

        self._backprop(node, value)
        return True

    # --- Public API ---
    def search(self, game: GomokuGame):
        action_size = config.ACTION_SPACE_SIZE
        empty_policy = np.zeros(action_size, dtype=np.float32)

        if game.get_game_ended() is not None:
            return empty_policy, 0.0, -1

        # Every move is selected by PUCT; there are no tactical overrides.
        try:
            while True:
                self.result_queue.get_nowait()
        except Empty:
            pass

        obs = game.get_board_state(game.current_player, game.last_move)
        try:
            policy_logits, root_value, _ = self._remote_initial_inference(obs)
        except Empty:
            self.logger.warning(f"Worker {self.worker_id}: timed out on root inference.")
            return empty_policy, 0.0, -1

        root = PUCTNode()
        self._expand(root, game, policy_logits)
        root.visit_count = 1
        root.value_sum = float(np.clip(root_value, -1.0, 1.0))

        for _ in range(max(0, int(config.NUM_SIMULATIONS) - 1)):
            if not self._simulate(root, game):
                break

        return self._decision(root, game)

    def _decision(self, root: PUCTNode, game: GomokuGame):
        action_size = config.ACTION_SPACE_SIZE
        visits = np.zeros(action_size, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        total = visits.sum()
        if total > 0:
            policy = visits / total
        elif root.is_expanded:
            policy = np.zeros(action_size, dtype=np.float64)
            policy[root.actions] = root.priors
        else:
            return np.zeros(action_size, dtype=np.float32), 0.0, -1

        policy = policy.astype(np.float32)

        if self.sample_moves and game.move_count < self.play_temperature_moves and policy.sum() > 0:
            action = int(np.random.choice(action_size, p=policy / policy.sum()))
        else:
            action = int(np.argmax(policy))

        return policy, float(root.value()), action
