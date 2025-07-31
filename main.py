# =====================================================================
#
#       AlphaZero Gomoku - V55.10 (Strategic Self-Play)
#
# - [特性] 增加了对最新模型的“优先自对弈”策略。通过
#   `NEWEST_MODEL_SELF_PLAY_RATIO` 参数，可以强制让最新的冠军模型
#   执行更高比例的自对弈，以加速高质量新数据的产生。
# - [重构] `run_self_play_phase` 函数被重构，以支持智能地将不同的
#   模型池（最新 vs 元老）分配给不同的工作进程。
# - [最终形态] 这是当前设计中功能最完整、运行最稳定、界面最高效、
#   逻辑最精简的最终版本。
#
# =====================================================================

import torch
import numpy as np
import time
import os
import shutil
from flask import Flask, request, jsonify, render_template
from network import GomokuNet
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import argparse
import math
from tqdm import trange, tqdm
import multiprocessing as mp
from queue import Empty
import threading
import webbrowser
import json
import glob
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pickle
import optuna
from colorama import init, Fore, Style
from numba import njit

# =====================================================================
#                          配置与全局变量
# =====================================================================

class QuickTestConfig:
    """用于快速原型验证的精简配置"""
    NAME = "QuickTest"

    # 快速验证的话棋盘尺寸可以缩小
    BOARD_SIZE = 15
    N_IN_ROW = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型尺寸太小了不行，学习不到东西没办法进步
    NUM_RES_BLOCKS = 10
    NUM_FILTERS = 256
    DROPOUT_P = 0.3
    
    # MCTS_SIMULATIONS太小了不行，局面杂乱无章会难以学习，如果没办法实现自举可以设置成1200
    MCTS_SIMULATIONS = 800
    MCTS_NOISE_ALPHA = 0.3
    MCTS_NOISE_FRAC = 0.25
    MCTS_C_PUCT = 3.0    
    SELF_PLAY_GAMES_PER_ITER = 280
    SELF_PLAY_SAVE_INTERVAL = 25

    # 4090单卡推荐配置，根据需求自己调
    NUM_SELF_PLAY_WORKERS = 12
    MAX_GPU_WORKERS_SELF_PLAY = 9

    
    DEFAULT_BATCH_SIZE = 512
    DEFAULT_LEARNING_RATE = 2e-4
    DEFAULT_WEIGHT_DECAY = 1e-4
    DEFAULT_TRAIN_STEPS = 1400
    TRAIN_SAVE_INTERVAL = 100
    
    TRAIN_BUFFER_SIZE = 1000000
    PER_ALPHA = 0.6
    
    EVAL_GAMES = 10
    EVAL_WIN_RATE = 0.55
    
    NEWEST_MODEL_SELF_PLAY_RATIO = 0.9
    
    HALL_OF_FAME_SIZE = 5
    
    OPTUNA_N_TRIALS = 15
    
    HOST = '127.0.0.1'
    PORT = 5001
    CHECKPOINT_DIR = f"checkpoints_RES{NUM_RES_BLOCKS}_FIL{NUM_FILTERS}"

init(autoreset=True)

class CLog:
    @staticmethod
    def info_block(title, data, indent=4):
        print(f"{' ' * indent}{Style.DIM}{title}{Style.RESET_ALL}")
        max_key_len = max(len(k) for k in data.keys()) if data else 0
        for key, value in data.items():
            if isinstance(value, float): value_str = f"{value:.2e}"
            else: value_str = str(value)
            print(f"{' ' * (indent+2)}{Style.DIM}- {key.ljust(max_key_len)}: {value_str}{Style.RESET_ALL}")

config = QuickTestConfig()
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

PAUSE_TRAINING_EVENT = mp.Event()
gpu_semaphore = mp.Semaphore(config.MAX_GPU_WORKERS_SELF_PLAY)

current_model = GomokuNet(board_size=config.BOARD_SIZE, num_res_blocks=config.NUM_RES_BLOCKS, num_filters=config.NUM_FILTERS, dropout_p=config.DROPOUT_P)
best_model = GomokuNet(board_size=config.BOARD_SIZE, num_res_blocks=config.NUM_RES_BLOCKS, num_filters=config.NUM_FILTERS, dropout_p=config.DROPOUT_P)
optimizer, scheduler = None, None
scaler = torch.amp.GradScaler(enabled=(config.DEVICE.type == 'cuda'))

train_data = {}; train_priorities = {}; train_order = deque(maxlen=config.TRAIN_BUFFER_SIZE)
next_data_id, iteration, current_train_step = 0, 0, 0

# 用于Live模式的模型缓存，避免重复加载
live_model_cache = {}

# =====================================================================
#                      游戏与MCTS核心逻辑 (Numba 加速)
# =====================================================================

@njit(cache=True)
def _check_win_numba(board, last_move_row, last_move_col, n_in_row, board_size):
    if last_move_row == -1: return False
    player = board[last_move_row, last_move_col]
    if player == 0: return False
    
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        for i in range(1, n_in_row):
            r, c = last_move_row + i * dr, last_move_col + i * dc
            if 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player: count += 1
            else: break
        for i in range(1, n_in_row):
            r, c = last_move_row - i * dr, last_move_col - i * dc
            if 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player: count += 1
            else: break
        if count >= n_in_row: return True
    return False

class GomokuGame:
    def __init__(self, board_size=15, n_in_row=5): self.board_size, self.n_in_row = board_size, n_in_row; self.reset()
    def reset(self): self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8); self.current_player, self.last_move, self.move_count = 1, None, 0
    def get_board_state(self, player, last_move):
        board_state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32); board_state[0] = (self.board == player); board_state[1] = (self.board == -player)
        if last_move is not None: board_state[2, last_move[0], last_move[1]] = 1
        return board_state
    def get_valid_moves(self): return list(zip(*np.where(self.board == 0)))
    def do_move(self, move): self.board[move[0], move[1]] = self.current_player; self.last_move = move; self.current_player = -self.current_player; self.move_count += 1
    def check_win(self):
        if self.last_move is None: return False
        return _check_win_numba(self.board, self.last_move[0], self.last_move[1], self.n_in_row, self.board_size)
    def get_game_ended(self):
        if self.check_win(): return self.board[self.last_move[0], self.last_move[1]]
        if self.move_count >= self.board_size * self.board_size: return 0
        return None
    def clone(self): ng = GomokuGame(self.board_size, self.n_in_row); ng.board = np.copy(self.board); ng.current_player, ng.last_move, ng.move_count = self.current_player, self.last_move, self.move_count; return ng

class MCTSNode:
    def __init__(self, parent, prior_p):
        self.parent = parent; self.children = {}; self.N = 0; self.W = 0; self.Q = 0; self.P = prior_p
    def select(self, c_puct): return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
    def expand(self, policy, game):
        valid_moves_map = {m: True for m in game.get_valid_moves()}
        for move_idx, prob in enumerate(policy):
            if prob > 0:
                move = (move_idx // config.BOARD_SIZE, move_idx % config.BOARD_SIZE)
                if move in valid_moves_map and move_idx not in self.children: self.children[move_idx] = MCTSNode(self, prob)
    def backpropagate(self, value):
        self.N += 1; self.W += value; self.Q = self.W / self.N
        if self.parent: self.parent.backpropagate(-value)
    def get_value(self, c_puct):
        U = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + U

class MCTS:
    def __init__(self, model, c_puct, noise_frac=config.MCTS_NOISE_FRAC):
        self.model = model; self.c_puct = c_puct; self.noise_frac = noise_frac
    @torch.no_grad()
    def run_simulations(self, game, add_exploration_noise=False):
        root = MCTSNode(None, 1.0)
        state_tensor = torch.from_numpy(game.get_board_state(game.current_player, game.last_move)).unsqueeze(0).to(config.DEVICE)
        with torch.amp.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type == 'cuda')):
            policy, value, _ = self.model(state_tensor)
        policy_np = torch.exp(policy).squeeze(0).cpu().numpy()
        if add_exploration_noise:
            valid_moves = game.get_valid_moves(); valid_indices = [m[0]*config.BOARD_SIZE + m[1] for m in valid_moves]
            if valid_indices:
                noise = np.random.dirichlet([config.MCTS_NOISE_ALPHA] * len(valid_indices))
                for i, idx in enumerate(valid_indices): policy_np[idx] = (1 - self.noise_frac) * policy_np[idx] + self.noise_frac * noise[i]
        root.expand(policy_np, game); root.backpropagate(value.item())
        leaf_nodes_to_evaluate = []
        for _ in range(config.MCTS_SIMULATIONS - 1):
            node, sim_game = root, game.clone()
            while node.children:
                action_idx, node = node.select(self.c_puct)
                sim_game.do_move((action_idx // config.BOARD_SIZE, action_idx % config.BOARD_SIZE))
            winner = sim_game.get_game_ended()
            if winner is not None:
                node.backpropagate(float(winner) * sim_game.current_player * -1)
            else:
                leaf_nodes_to_evaluate.append((node, sim_game))
        if leaf_nodes_to_evaluate:
            states_batch = np.array([sg.get_board_state(sg.current_player, sg.last_move) for _, sg in leaf_nodes_to_evaluate])
            with torch.amp.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type == 'cuda')):
                 policies_batch, values_batch, _ = self.model(torch.from_numpy(states_batch).to(config.DEVICE))
            policies_np = torch.exp(policies_batch).cpu().numpy(); values_np = values_batch.squeeze(1).cpu().numpy()
            for i, (node, sim_game) in enumerate(leaf_nodes_to_evaluate):
                node.expand(policies_np[i], sim_game); value = values_np[i]; node.backpropagate(value)
        return root
    @torch.no_grad()
    def get_action_probs(self, game, temp=1.0, add_exploration_noise=False):
        self.model.eval()
        root = self.run_simulations(game, add_exploration_noise)
        visit_counts = np.array([root.children.get(i, MCTSNode(None,0)).N for i in range(config.BOARD_SIZE**2)])
        if np.sum(visit_counts) == 0:
            valid_moves = game.get_valid_moves(); probs = np.zeros(config.BOARD_SIZE**2)
            if valid_moves:
                prob = 1.0 / len(valid_moves)
                for r, c in valid_moves: probs[r * config.BOARD_SIZE + c] = prob
            return probs
        if temp == 0:
            action_idx = np.argmax(visit_counts); probs = np.zeros_like(visit_counts, dtype=float); probs[action_idx] = 1.0
        else:
            visit_counts_temp = visit_counts**(1/temp); probs = visit_counts_temp / np.sum(visit_counts_temp)
        return probs
    @torch.no_grad()
    def get_move_and_value(self, game, temp=0):
        self.model.eval()
        root = self.run_simulations(game, add_exploration_noise=False)
        visit_counts = np.array([root.children.get(i, MCTSNode(None,0)).N for i in range(config.BOARD_SIZE**2)])
        if np.sum(visit_counts) == 0:
            return -1, 0.0
        action_idx = np.argmax(visit_counts) if temp == 0 else np.random.choice(len(visit_counts), p=(visit_counts**(1/temp))/np.sum(visit_counts**(1/temp)))
        return action_idx, root.Q

# =====================================================================
#                核心工作与工具函数
# =====================================================================

def game_worker(process_id, games_per_worker, data_queue, mode, gpu_sem, model_paths, c_puct, debug_logging=False):
    def log(message):
        if debug_logging: print(f"[{time.strftime('%H:%M:%S')}][Worker {process_id}] {message}", flush=True)
    try:
        model_a = GomokuNet(board_size=config.BOARD_SIZE, num_res_blocks=config.NUM_RES_BLOCKS, num_filters=config.NUM_FILTERS, dropout_p=config.DROPOUT_P)
        model_b = None
        if mode == 'self_play':
            model_path = random.choice(model_paths)
            state = torch.load(model_path, map_location='cpu')
            model_state_key = 'state_dict' if 'state_dict' in state else 'best_model_state_dict'
            model_a.load_state_dict(state[model_state_key])
            mcts_a = MCTS(model_a, c_puct, noise_frac=config.MCTS_NOISE_FRAC)
        elif mode == 'eval':
            model_b = GomokuNet(board_size=config.BOARD_SIZE, num_res_blocks=config.NUM_RES_BLOCKS, num_filters=config.NUM_FILTERS, dropout_p=config.DROPOUT_P)
            model_a.load_state_dict(torch.load(model_paths[0], map_location='cpu')['state_dict']) # Challenger
            model_b.load_state_dict(torch.load(model_paths[1], map_location='cpu')['state_dict']) # Defender
            mcts_a, mcts_b = MCTS(model_a, c_puct), MCTS(model_b, c_puct)
        elif mode == 'random':
            model_a.load_state_dict(torch.load(model_paths[0], map_location='cpu')['state_dict'])
            mcts_a = MCTS(model_a, c_puct)
        
        gpu_sem.acquire()
        try:
            model_a.to(config.DEVICE).eval()
            if model_b: model_b.to(config.DEVICE).eval()
            for i in range(games_per_worker):
                game = GomokuGame(config.BOARD_SIZE, config.N_IN_ROW)
                if mode == 'self_play':
                    play_data = []
                    while True:
                        temp = 1.0 if len(play_data) < 60 else 0.1
                        action_probs = mcts_a.get_action_probs(game, temp=temp, add_exploration_noise=True)
                        if np.sum(action_probs) == 0: winner = 0; break
                        action_idx = np.random.choice(len(action_probs), p=action_probs)
                        pi = mcts_a.get_action_probs(game, temp=1.0, add_exploration_noise=False)
                        play_data.append([game.get_board_state(game.current_player, game.last_move), pi.reshape(config.BOARD_SIZE, config.BOARD_SIZE), 0])
                        game.do_move((action_idx // config.BOARD_SIZE, action_idx % config.BOARD_SIZE))
                        winner = game.get_game_ended()
                        if winner is not None: break
                    if winner is not None and play_data:
                        value = float(winner)
                        for step_data in reversed(play_data):
                            step_data[2] = value; value *= -1
                        game_id = f"{os.getpid()}-{time.time()}-{i}"
                        data_queue.put(("raw_game", play_data, game_id))
                elif mode == 'eval':
                    move_history = []
                    players = {1: mcts_a, -1: mcts_b} if (process_id + i) % 2 == 0 else {1: mcts_b, -1: mcts_a}
                    while True:
                        mcts_player = players[game.current_player]
                        move_idx = np.argmax(mcts_player.get_action_probs(game, temp=0, add_exploration_noise=False))
                        move = (move_idx // config.BOARD_SIZE, move_idx % config.BOARD_SIZE)
                        move_history.append((int(move[0]), int(move[1])))
                        game.do_move(move)
                        winner = game.get_game_ended()
                        if winner is not None:
                            result = 0 if winner == 0 else 1 if (winner == 1 and players[1] == mcts_a) or (winner == -1 and players[-1] == mcts_a) else -1
                            game_length = len(move_history)
                            winner_name = "Draw" if result == 0 else ("Challenger" if result == 1 else "Defender")
                            detailed_moves = move_history if i == 0 else []
                            replay_data = {"challenger_color": "Black" if players[1] == mcts_a else "White", "defender_color": "White" if players[1] == mcts_a else "Black", "moves": detailed_moves, "winner": winner_name, "length": game_length}
                            data_queue.put(('eval_game_data', result, replay_data))
                            break
                elif mode == 'random':
                    model_player_side = 1 if i % 2 == 0 else -1
                    while True:
                        if game.current_player == model_player_side:
                            move_idx = np.argmax(mcts_a.get_action_probs(game, temp=0, add_exploration_noise=False))
                            move = (move_idx // config.BOARD_SIZE, move_idx % config.BOARD_SIZE)
                        else:
                            valid_moves = game.get_valid_moves()
                            if not valid_moves: winner = 0; break
                            move = random.choice(valid_moves)
                        game.do_move(move)
                        winner = game.get_game_ended()
                        if winner is not None:
                            data_queue.put(('result', 1 if winner == model_player_side else 0))
                            break
        finally:
            model_a.cpu();
            if model_b: model_b.cpu()
            gpu_sem.release()
    except Exception as e:
        log(f"!!! WORKER ERROR: {e}")
        with open(f"worker_{process_id}_error.log", "w") as f: import traceback; traceback.print_exc(file=f)
    finally:
        data_queue.put((None, None))

def get_training_batch(batch_size):
    if len(train_data) < batch_size: return None
    ids = list(train_data.keys()); priorities = np.array([train_priorities[i] for i in ids], dtype=np.float64)
    probs = priorities / priorities.sum() if priorities.sum() > 0 else None
    sampled_ids = np.random.choice(ids, batch_size, p=probs, replace=len(ids) < batch_size)
    batch = [train_data[i] for i in sampled_ids]
    augmented_states, augmented_pis_2d, original_values = [], [], []
    for state, pi_2d, value in batch:
        k, flip = np.random.randint(4), np.random.choice([True, False])
        aug_state, aug_pi_2d = np.rot90(state, k, axes=(1, 2)), np.rot90(pi_2d, k, axes=(0, 1))
        if flip: aug_state, aug_pi_2d = np.flip(aug_state, axis=2), np.flip(aug_pi_2d, axis=1)
        augmented_states.append(np.ascontiguousarray(aug_state))
        augmented_pis_2d.append(np.ascontiguousarray(aug_pi_2d))
        original_values.append(value)
    states_tensor = torch.FloatTensor(np.stack(augmented_states)).to(config.DEVICE)
    target_policies_tensor = torch.FloatTensor(np.stack(augmented_pis_2d).reshape(batch_size, -1)).to(config.DEVICE)
    target_values_tensor = torch.FloatTensor(np.array(original_values)).unsqueeze(1).to(config.DEVICE)
    return states_tensor, target_policies_tensor, target_values_tensor, sampled_ids

def perform_train_step(model, optimizer, scaler, batch_data, writer=None, global_step=None):
    states_tensor, target_policies_tensor, target_values_tensor, sampled_ids = batch_data
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=config.DEVICE.type, dtype=torch.float16, enabled=(config.DEVICE.type == 'cuda')):
        out_policy, out_value, out_outcome = model(states_tensor)
        policy_loss = -torch.sum(target_policies_tensor * out_policy) / states_tensor.size(0)
        value_loss = torch.nn.functional.mse_loss(out_value, target_values_tensor)
        outcome_loss = torch.nn.functional.mse_loss(out_outcome, target_values_tensor)
        loss = policy_loss + value_loss + outcome_loss
    with torch.no_grad():
        td_error = torch.abs(target_values_tensor - out_value).squeeze(1).cpu().numpy()
        for i, idx in enumerate(sampled_ids):
            if idx in train_priorities: train_priorities[idx] = (td_error[i] + 1e-6) ** config.PER_ALPHA
    scaler.scale(loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer); scaler.update()
    if writer and global_step:
        writer.add_scalar('Loss/Total', loss.item(), global_step)
    return loss.item(), policy_loss.item(), value_loss.item(), outcome_loss.item()

def run_evaluation_duel(challenger_path, defender_path, desc="Eval"):
    num_games = config.EVAL_GAMES
    data_queue = mp.Queue()
    num_workers = min(config.NUM_SELF_PLAY_WORKERS, num_games)
    base_games, rem = divmod(num_games, num_workers)
    games_per_worker = [base_games + 1 if i < rem else base_games for i in range(num_workers)]
    processes = [mp.Process(target=game_worker, args=(i, games_per_worker[i], data_queue, 'eval', gpu_semaphore, [challenger_path, defender_path], config.MCTS_C_PUCT)) for i in range(num_workers) if games_per_worker[i] > 0]
    for p in processes: p.start()
    
    results, game_lengths, wins, losses, draws = [], [], 0, 0, 0
    full_replay_data = None
    pbar_format = "  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    with tqdm(total=num_games, desc=f"--> {desc}", leave=True, ncols=100, bar_format=pbar_format, ascii=True) as pbar:
        workers_done, results_collected = 0, 0
        while results_collected < num_games and workers_done < len(processes):
            try:
                data = data_queue.get(timeout=600)
                if data == (None, None): workers_done += 1; continue
                
                data_type, result, replay_data = data
                if data_type == 'eval_game_data':
                    results.append(result)
                    if result == 1: wins += 1
                    elif result == -1: losses += 1
                    else: draws += 1
                    
                    if 'length' in replay_data: game_lengths.append(replay_data['length'])
                    if full_replay_data is None and replay_data.get('moves'):
                        full_replay_data = replay_data
                    
                    pbar.update(1); results_collected += 1
                
                avg_len_str = f"AvgLen:{np.mean(game_lengths):.1f}" if game_lengths else ""
                
                pbar.set_postfix_str(f" ✌️{wins} 🥀{losses} 🟰{draws}, {avg_len_str}", refresh=True)

            except Empty: print(f"\n{Fore.RED}FATAL: Duel queue timed out!{Style.RESET_ALL}"); break
    
    avg_game_length = np.mean(game_lengths) if game_lengths else 0
    win_rate = wins / num_games if num_games > 0 else 0
    success = win_rate > config.EVAL_WIN_RATE
    
    final_postfix = f" ✌️{wins} 🥀{losses} 🟰{draws}, AvgLen:{avg_game_length:.1f}"
    pbar.set_postfix_str(final_postfix)
    pbar.refresh()
    
    for p in processes: p.join()
    return {"win_rate": win_rate, "success": success, "avg_game_length": avg_game_length, "replay_data": full_replay_data}

def get_unique_timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def save_replay_file(replay_data, iter_num, attempt_name):
    if not replay_data: return
    replay_dir = os.path.join(config.CHECKPOINT_DIR, "replays")
    os.makedirs(replay_dir, exist_ok=True)
    filename = f"replay_iter_{iter_num}_{attempt_name}.json"
    replay_filepath = os.path.join(replay_dir, filename)
    try:
        with open(replay_filepath, 'w') as f:
            json.dump(replay_data, f, indent=4)
        print(f"  {Style.DIM}- Replay saved: {filename}{Style.RESET_ALL}")
    except Exception as e:
        print(f"  {Fore.RED}- Error saving replay file: {e}{Style.RESET_ALL}")

def visualize_policy(model, iter_num, attempt_name):
    print(f"  {Style.DIM}- Generating policy visualization for Iteration {iter_num}...{Style.RESET_ALL}")
    vis_dir = os.path.join(config.CHECKPOINT_DIR, "visuals")
    os.makedirs(vis_dir, exist_ok=True)
    filename = f"policy_iter_{iter_num}_{attempt_name}.png"
    filepath = os.path.join(vis_dir, filename)
    model.to(config.DEVICE).eval()
    game = GomokuGame(config.BOARD_SIZE, config.N_IN_ROW)
    state = game.get_board_state(1, None)
    state_tensor = torch.from_numpy(state).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad(), torch.amp.autocast(device_type=config.DEVICE.type, enabled=(config.DEVICE.type == 'cuda')):
        policy_logits, _, _ = model(state_tensor)
        policy = torch.exp(policy_logits).squeeze(0).cpu().detach().numpy().reshape(config.BOARD_SIZE, config.BOARD_SIZE)
    plt.figure(figsize=(10, 10))
    sns.heatmap(policy, cmap="viridis", cbar=True, square=True)
    plt.title(f"Policy Heatmap at Iteration {iter_num} ({attempt_name})")
    plt.savefig(filepath)
    plt.close()
    model.cpu()
    print(f"  {Style.DIM}- Heatmap saved: {filename}{Style.RESET_ALL}")

def save_selfplay_progress(iteration, collected_data):
    progress_file = os.path.join(config.CHECKPOINT_DIR, f"selfplay_progress_iter_{iteration}.pkl")
    try:
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        with open(progress_file, 'wb') as f: pickle.dump(collected_data, f)
    except IOError as e: print(f"{Fore.YELLOW}Warning: Could not save self-play progress: {e}{Style.RESET_ALL}")

def load_selfplay_progress(iteration):
    progress_file = os.path.join(config.CHECKPOINT_DIR, f"selfplay_progress_iter_{iteration}.pkl")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'rb') as f: return pickle.load(f)
        except Exception as e: print(f"{Fore.YELLOW}Warning: Could not load self-play progress file. Error: {e}{Style.RESET_ALL}"); return []
    return []

def save_buffer(filepath=None):
    if filepath is None: filepath = os.path.join(config.CHECKPOINT_DIR, "buffer.pkl")
    temp_filepath = filepath + ".tmp"
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(temp_filepath, 'wb') as f: pickle.dump({'train_data': dict(train_data), 'train_priorities': dict(train_priorities), 'train_order': train_order, 'next_data_id': next_data_id}, f)
        shutil.move(temp_filepath, filepath)
    except Exception as e: print(f"{Fore.RED}❌ Error saving replay buffer: {e}{Style.RESET_ALL}")

def load_buffer(filepath=None):
    global train_data, train_priorities, train_order, next_data_id
    
    if filepath is None:
        filepath = os.path.join(config.CHECKPOINT_DIR, "buffer.pkl")
        
    if os.path.exists(filepath):
        try:
            print(f"  {Style.DIM}- Loading replay buffer from: {os.path.basename(filepath)}{Style.RESET_ALL}")
            with open(filepath, 'rb') as f: 
                buffer_state = pickle.load(f)

            first_key = next(iter(buffer_state.get('train_data', {'0':None})), '0')
            needs_reindexing = not str(first_key).isdigit()

            if needs_reindexing:
                print(f"  {Fore.YELLOW}Warning: Incompatible key format ('{first_key}') detected in buffer.{Style.RESET_ALL}")
                print(f"  {Fore.CYAN}Action: Starting a one-time re-indexing process to recover data...{Style.RESET_ALL}")
                loaded_train_data_values = buffer_state.get('train_data', {}).values()
                
                train_data.clear()
                train_priorities.clear()
                train_order.clear()
                
                new_id = 0
                
                print(f"  {Style.DIM}- Re-indexing {len(loaded_train_data_values)} data points...{Style.RESET_ALL}")
                
                latest_data = list(loaded_train_data_values)[-config.TRAIN_BUFFER_SIZE:]

                for step_data in latest_data:
                    train_data[new_id] = step_data
                    train_priorities[new_id] = 1.0
                    train_order.append(new_id)
                    new_id += 1
                
                next_data_id = new_id
                
                print(f"  {Fore.GREEN}[✅] Buffer successfully re-indexed. {len(train_data)} items recovered.{Style.RESET_ALL}")
                print(f"  {Fore.CYAN}Action: Saving the newly indexed buffer to prevent future recovery...{Style.RESET_ALL}")
                save_buffer()

            else:
                print(f"  {Style.DIM}- Buffer format looks OK. Sanitizing data types...{Style.RESET_ALL}")
                train_data = {int(k): v for k, v in buffer_state.get('train_data', {}).items()}
                train_priorities = {int(k): v for k, v in buffer_state.get('train_priorities', {}).items()}
                
                sanitized_order = deque(maxlen=config.TRAIN_BUFFER_SIZE)
                for item in buffer_state.get('train_order', []):
                    sanitized_order.append(int(item))
                train_order = sanitized_order
                
                next_data_id = buffer_state.get('next_data_id', 0)
                print(f"  {Style.DIM}- Buffer sanitized successfully.{Style.RESET_ALL}")

        except Exception as e: 
            print(f"{Fore.RED}❌ FATAL Error during buffer recovery: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            print(f"{Fore.YELLOW}Warning: Starting with a fresh, empty buffer due to unrecoverable error.{Style.RESET_ALL}")
            train_data, train_priorities, train_order, next_data_id = {}, {}, deque(maxlen=config.TRAIN_BUFFER_SIZE), 0

def save_checkpoint(is_best=False, step_override=None):
    resumable_state = {
        'iteration': iteration, 
        'current_train_step': step_override if step_override is not None else current_train_step, 
        'state_dict': current_model.state_dict(), 
        'best_model_state_dict': best_model.state_dict(), 
        'optimizer': optimizer.state_dict() if optimizer else None, 
        'scheduler': scheduler.state_dict() if scheduler else None, 
        'scaler': scaler.state_dict()
    }
    filepath = os.path.join(config.CHECKPOINT_DIR, "checkpoint.pth.tar")
    temp_filepath = filepath + ".tmp"
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    torch.save(resumable_state, temp_filepath)
    shutil.move(temp_filepath, filepath)
    if is_best:
        torch.save({'state_dict': best_model.state_dict()}, os.path.join(config.CHECKPOINT_DIR, "best.pth.tar"))

def load_checkpoint(resume=False):
    global iteration, current_train_step, optimizer, scheduler, scaler
    best_path, resume_path = os.path.join(config.CHECKPOINT_DIR, "best.pth.tar"), os.path.join(config.CHECKPOINT_DIR, "checkpoint.pth.tar")
    load_path = resume_path if resume and os.path.exists(resume_path) else best_path
    if not os.path.exists(load_path):
        optimizer = optim.Adam(current_model.parameters(), lr=config.DEFAULT_LEARNING_RATE, weight_decay=config.DEFAULT_WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        save_checkpoint(is_best=True)
        return
    state = torch.load(load_path, map_location='cpu')
    current_model.load_state_dict(state.get('state_dict', state.get('best_model_state_dict')))
    best_model.load_state_dict(state.get('best_model_state_dict', state.get('state_dict')))
    optimizer = optim.Adam(current_model.parameters(), lr=config.DEFAULT_LEARNING_RATE, weight_decay=config.DEFAULT_WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    if resume and 'optimizer' in state and state['optimizer']:
        iteration, current_train_step = state.get('iteration', 0), state.get('current_train_step', 0)
        try:
            optimizer.load_state_dict(state['optimizer'])
            if config.DEVICE.type == 'cuda':
                for param_state in optimizer.state.values():
                    for k, v in param_state.items():
                        if isinstance(v, torch.Tensor): param_state[k] = v.to(config.DEVICE)
            if 'scheduler' in state and state['scheduler']: scheduler.load_state_dict(state['scheduler'])
            if 'scaler' in state and state['scaler']: scaler.load_state_dict(state['scaler'])
        except Exception as e: print(f"{Fore.YELLOW}Warning: Could not load optimizer/scheduler state: {e}{Style.RESET_ALL}")

def update_best_model_and_hof(iter_num, attempt_name):
    """Handles all logic for promoting a winning model with a UNIFIED naming convention."""
    print(f"  {Style.BRIGHT}{Fore.GREEN}Promoting new best model from '{attempt_name}'...{Style.RESET_ALL}")
    
    best_model.load_state_dict(current_model.state_dict())
    
    save_checkpoint(is_best=True)
    print(f"  {Style.DIM}- 'best.pth.tar' updated successfully.{Style.RESET_ALL}")

    hof_dir = os.path.join(config.CHECKPOINT_DIR, "hall_of_fame")
    os.makedirs(hof_dir, exist_ok=True)

    label_iter = iter_num + 1
    
    hof_filename = f"hof_model_iter_{label_iter}_{attempt_name}.pth.tar"
    hof_filepath = os.path.join(hof_dir, hof_filename)
    torch.save({'state_dict': best_model.state_dict()}, hof_filepath)
    print(f"  {Style.DIM}- Added to Hall of Fame: {hof_filename}{Style.RESET_ALL}")
    
    all_hof_models = sorted(glob.glob(os.path.join(hof_dir, "*.pth.tar")), key=os.path.getmtime)
    if len(all_hof_models) > config.HALL_OF_FAME_SIZE:
        num_to_prune = len(all_hof_models) - config.HALL_OF_FAME_SIZE
        for old_model_path in all_hof_models[:num_to_prune]:
            try:
                os.remove(old_model_path)
                print(f"  {Style.DIM}- Pruned old Hall of Fame model: {os.path.basename(old_model_path)}{Style.RESET_ALL}")
            except OSError as e:
                print(f"  {Fore.RED}- Error pruning HoF model {os.path.basename(old_model_path)}: {e}{Style.RESET_ALL}")

# =====================================================================
#                      Web UI
# =====================================================================
def run_web_server():
    from werkzeug.serving import run_simple, WSGIRequestHandler
    class QuietWSGIRequestHandler(WSGIRequestHandler):
        def log(self, type, message, *args): pass
    logging.getLogger('werkzeug').setLevel(logging.CRITICAL)
    try:
        run_simple(config.HOST, config.PORT, app, use_reloader=False, use_debugger=False, request_handler=QuietWSGIRequestHandler)
    except Exception as e: print(f"\n{Fore.RED}FATAL ERROR in Web Server thread: {e}{Style.RESET_ALL}")

@app.route('/')
def index(): return render_template('index.html')

@app.route('/move', methods=['POST'])
def handle_move():
    PAUSE_TRAINING_EVENT.set(); time.sleep(0.1)
    ai_move, winner, black_win_rate = None, None, 50.0
    try:
        best_model.to(config.DEVICE).eval()
        data = request.json
        game = GomokuGame(config.BOARD_SIZE, config.N_IN_ROW)
        game.board = np.array(data['board'])
        game.move_count = np.sum(game.board != 0)
        if 'move' in data and data['move'] is not None:
            human_move = tuple(data['move'])
            game.board[human_move[0], human_move[1]] = data['player_to_move']
            game.last_move = human_move; game.move_count += 1
            if game.check_win():
                winner = game.board[human_move]
                return jsonify({'ai_move': None, 'game_over': True, 'winner': int(winner), 'black_win_rate': 100.0 if winner == 1 else 0.0})
            if game.get_game_ended() == 0: return jsonify({'ai_move': None, 'game_over': True, 'winner': 0, 'black_win_rate': 50.0})
            game.current_player = -data['player_to_move']
        ai_move_idx, value_for_ai = MCTS(best_model, config.MCTS_C_PUCT).get_move_and_value(game)
        value_for_ai = float(value_for_ai)
        black_win_rate = (value_for_ai + 1) / 2 * 100 if game.current_player == 1 else (-value_for_ai + 1) / 2 * 100
        if ai_move_idx != -1:
            ai_move = (ai_move_idx // config.BOARD_SIZE, ai_move_idx % config.BOARD_SIZE)
            game.do_move(ai_move)
            winner = game.get_game_ended()
    except Exception as e: print(f"Error in /move: {e}"); return jsonify({"error": str(e)}), 500
    finally:
        best_model.cpu(); PAUSE_TRAINING_EVENT.clear()
    return jsonify({'ai_move': [int(ai_move[0]), int(ai_move[1])] if ai_move else None, 'game_over': winner is not None, 'winner': int(winner) if winner is not None else None, 'black_win_rate': black_win_rate})

@app.route('/get_replay_list')
def get_replay_list():
    replay_dir = os.path.join(config.CHECKPOINT_DIR, "replays")
    if not os.path.exists(replay_dir):
        return jsonify([])

    files = glob.glob(os.path.join(replay_dir, "replay_iter_*.json"))
    replay_info_list = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.replace('.json', '').split('_')
        if len(parts) >= 4:
            try:
                iter_num = int(parts[2])
                attempt_name = ' '.join(parts[3:]).title()
                replay_info_list.append({
                    "path": basename,
                    "iter": iter_num,
                    "name": attempt_name
                })
            except (ValueError, IndexError):
                continue
    
    sorted_replays = sorted(replay_info_list, key=lambda x: x['iter'], reverse=True)
    return jsonify(sorted_replays)

@app.route('/load_replay/<path:filename>')
def load_replay(filename):
    if '..' in filename or filename.startswith('/'):
        return jsonify({"error": "Invalid filename"}), 400
    filepath = os.path.join(config.CHECKPOINT_DIR, "replays", filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f: return jsonify(json.load(f))
        except Exception as e: return jsonify({"error": f"Failed to load replay: {e}"}), 500
    return jsonify({"error": "Replay not found"}), 404

def get_cached_model(model_relative_path):
    """
    从缓存中获取模型，如果缓存未命中，则从磁盘加载并存入缓存。
    """
    if model_relative_path in live_model_cache:
        return live_model_cache[model_relative_path]
    else:
        # 在打印前后都加上换行符 \n，确保日志输出独立成行，避免与tqdm等其他日志交错。
        print(f"\n  {Style.DIM}- Caching live model: {model_relative_path}{Style.RESET_ALL}")
        
        model_full_path = os.path.join(config.CHECKPOINT_DIR, model_relative_path)
        
        model = GomokuNet(board_size=config.BOARD_SIZE, num_res_blocks=config.NUM_RES_BLOCKS, num_filters=config.NUM_FILTERS, dropout_p=config.DROPOUT_P)
        state_dict = torch.load(model_full_path, map_location='cpu')
        model.load_state_dict(state_dict.get('state_dict', state_dict.get('best_model_state_dict')))
        
        model.to(config.DEVICE).eval()
        
        live_model_cache[model_relative_path] = model
        return model

@app.route('/get_hof_list')
def get_hof_list():
    """获取名人堂模型列表，用于前端下拉菜单"""
    models_info = []
    best_model_path = "best.pth.tar"
    if os.path.exists(os.path.join(config.CHECKPOINT_DIR, best_model_path)):
        models_info.append({
            "name": "Current Best Model",
            "path": best_model_path
        })

    hof_dir = os.path.join(config.CHECKPOINT_DIR, "hall_of_fame")
    if os.path.exists(hof_dir):
        hof_files = sorted(
            glob.glob(os.path.join(hof_dir, "*.pth.tar")),
            key=os.path.getmtime,
            reverse=True
        )
        for f_path in hof_files:
            basename = os.path.basename(f_path)
            parts = basename.replace('.pth.tar', '').split('_')
            if len(parts) >= 4:
                try:
                    iter_num = int(parts[3])
                    attempt_name = ' '.join(parts[4:]).title()
                    display_name = f"Iter {iter_num} - {attempt_name}"
                    models_info.append({
                        "name": display_name,
                        "path": os.path.join("hall_of_fame", basename)
                    })
                except (ValueError, IndexError):
                    continue
    return jsonify(models_info)

@app.route('/live_move', methods=['POST'])
def handle_live_move():
    """处理Live模式下的模型对战走子请求 (已通过模型缓存优化)"""
    PAUSE_TRAINING_EVENT.set()
    time.sleep(0.01)
    
    ai_move, winner, black_win_rate = None, None, 50.0

    try:
        data = request.json
        game = GomokuGame(config.BOARD_SIZE, config.N_IN_ROW)
        game.board = np.array(data['board'])
        game.move_count = np.sum(game.board != 0)
        game.current_player = data['current_player']

        move_model_path = data['black_model_path'] if game.current_player == 1 else data['white_model_path']
        model_for_move = get_cached_model(move_model_path)
        
        mcts_for_move = MCTS(model_for_move, config.MCTS_C_PUCT)
        move_idx, _ = mcts_for_move.get_move_and_value(game)
        
        if move_idx == -1:
            winner = 0
        else:
            ai_move = (move_idx // config.BOARD_SIZE, move_idx % config.BOARD_SIZE)
            game.do_move(ai_move)
            winner = game.get_game_ended()

        if winner is not None:
            if winner == 1: black_win_rate = 100.0
            elif winner == -1: black_win_rate = 0.0
            else: black_win_rate = 50.0
        else:
            black_model_full_path = os.path.join(config.CHECKPOINT_DIR, data['black_model_path'])
            white_model_full_path = os.path.join(config.CHECKPOINT_DIR, data['white_model_path'])
            is_black_newer = os.path.getmtime(black_model_full_path) > os.path.getmtime(white_model_full_path)
            eval_model_path = data['black_model_path'] if is_black_newer else data['white_model_path']
            
            model_for_eval = get_cached_model(eval_model_path)
            
            mcts_for_eval = MCTS(model_for_eval, config.MCTS_C_PUCT)
            _, value_for_eval = mcts_for_eval.get_move_and_value(game)
            
            value_for_eval = float(value_for_eval)
            if game.current_player == 1:
                black_win_rate = (value_for_eval + 1) / 2 * 100
            else:
                black_win_rate = (-value_for_eval + 1) / 2 * 100
            
    except Exception as e:
        print(f"Error in /live_move: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        PAUSE_TRAINING_EVENT.clear()
        
    return jsonify({
        'ai_move': [int(ai_move[0]), int(ai_move[1])] if ai_move else None,
        'game_over': winner is not None,
        'winner': int(winner) if winner is not None else None,
        'black_win_rate': black_win_rate
    })

# =====================================================================
#                      核心执行流程
# =====================================================================

def run_self_play_phase(effective_iteration, args):
    print(f"\n{Fore.BLUE}[>] Phase 1/2: SELF-PLAY for Iteration {effective_iteration + 1}{Style.RESET_ALL}")
    
    collected_data_cache = load_selfplay_progress(effective_iteration)
    games_already_done = len(collected_data_cache)
    total_target_games = config.SELF_PLAY_GAMES_PER_ITER

    if games_already_done >= total_target_games:
        print(f"  {Style.DIM}--> Self-play already completed for this iteration.{Style.RESET_ALL}")
        return

    games_to_play_this_run = total_target_games - games_already_done
    
    hof_dir = os.path.join(config.CHECKPOINT_DIR, "hall_of_fame")
    all_hof_models = sorted(glob.glob(os.path.join(hof_dir, "*.pth.tar")), key=os.path.getmtime)
    
    newest_model_paths = []
    other_model_paths = []

    if all_hof_models:
        newest_model_paths = [all_hof_models[-1]]
        other_model_paths = all_hof_models[:-1]
    
    if not other_model_paths:
        other_model_paths.append(os.path.join(config.CHECKPOINT_DIR, "best.pth.tar"))
        if not newest_model_paths:
            newest_model_paths.append(os.path.join(config.CHECKPOINT_DIR, "best.pth.tar"))

    ratio = config.NEWEST_MODEL_SELF_PLAY_RATIO
    if len(all_hof_models) <= 1:
        ratio = 1.0

    num_games_for_newest = int(games_to_play_this_run * ratio)
    num_games_for_others = games_to_play_this_run - num_games_for_newest

    print(f"  {Style.DIM}--> Total games to play: {games_to_play_this_run}{Style.RESET_ALL}")
    print(f"  {Style.DIM}--> Newest Model ({len(newest_model_paths)}): {num_games_for_newest} games ({ratio:.0%}){Style.RESET_ALL}")
    print(f"  {Style.DIM}--> Veteran Models ({len(other_model_paths)}): {num_games_for_others} games ({1-ratio:.0%}){Style.RESET_ALL}")

    all_processes = []
    total_workers = config.NUM_SELF_PLAY_WORKERS
    
    data_queue = mp.Manager().Queue()

    if num_games_for_newest > 0:
        num_workers_for_newest = max(1, round(total_workers * (num_games_for_newest / games_to_play_this_run))) if games_to_play_this_run > 0 else 0
        if num_workers_for_newest > 0:
            base_games, rem = divmod(num_games_for_newest, num_workers_for_newest)
            games_per_worker = [base_games + 1 if i < rem else base_games for i in range(num_workers_for_newest)]
            
            for i in range(num_workers_for_newest):
                if games_per_worker[i] > 0:
                    p = mp.Process(target=game_worker, args=(i, games_per_worker[i], data_queue, 'self_play', gpu_semaphore, newest_model_paths, config.MCTS_C_PUCT, args.debug_selfplay))
                    all_processes.append(p)

    if num_games_for_others > 0:
        num_workers_for_others = total_workers - len(all_processes)
        if num_workers_for_others > 0:
            base_games, rem = divmod(num_games_for_others, num_workers_for_others)
            games_per_worker = [base_games + 1 if i < rem else base_games for i in range(num_workers_for_others)]
            
            for i in range(num_workers_for_others):
                if games_per_worker[i] > 0:
                    process_id = len(all_processes) + i
                    p = mp.Process(target=game_worker, args=(process_id, games_per_worker[i], data_queue, 'self_play', gpu_semaphore, other_model_paths, config.MCTS_C_PUCT, args.debug_selfplay))
                    all_processes.append(p)
    
    for p in all_processes:
        p.start()
        
    try:
        pbar_format = "  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        with tqdm(total=total_target_games, initial=games_already_done, desc=f"--> Self-Play", leave=True, ncols=100, bar_format=pbar_format, ascii=True) as pbar:
            workers_finished = 0
            last_saved_count = (games_already_done // config.SELF_PLAY_SAVE_INTERVAL) * config.SELF_PLAY_SAVE_INTERVAL
            
            game_lengths = []
            
            while workers_finished < len(all_processes):
                try:
                    data = data_queue.get(timeout=900)
                    if data == (None, None): 
                        workers_finished += 1
                        continue
                    
                    data_type, play_data, game_id = data
                    if data_type == "raw_game":
                        collected_data_cache.append(data)
                        game_lengths.append(len(play_data))
                        pbar.update(1)
                    
                    current_games = len(collected_data_cache)
                    if current_games % config.SELF_PLAY_SAVE_INTERVAL == 0 and current_games < total_target_games:
                        save_selfplay_progress(effective_iteration, collected_data_cache)
                        last_saved_count = current_games
                        
                    postfix_dict = {}
                    if game_lengths:
                        postfix_dict["AvgLen"] = f"{np.mean(game_lengths):.1f}"
                    postfix_dict["Saved at"] = last_saved_count
                    
                    postfix_str = ", ".join(f"{k}: {v}" for k,v in postfix_dict.items())
                    pbar.set_postfix_str(postfix_str, refresh=True)
                    
                except Empty: 
                    print(f"\n{Fore.RED}FATAL: Data queue timed out!{Style.RESET_ALL}")
                    break
    finally:
        print(f"  {Style.DIM}--> Finalizing self-play phase: saving {len(collected_data_cache)} collected games...{Style.RESET_ALL}")
        save_selfplay_progress(effective_iteration, collected_data_cache)
        
        for p in all_processes:
            if p.is_alive():
                p.join(timeout=10)
                p.terminate()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def add_completed_selfplay_to_buffer(iteration_to_add):
    global train_data, train_priorities, train_order, next_data_id
    
    progress_file = os.path.join(config.CHECKPOINT_DIR, f"selfplay_progress_iter_{iteration_to_add}.pkl")
    if not os.path.exists(progress_file):
        return

    print(f"  {Style.DIM}--> Merging self-play data from Iteration {iteration_to_add + 1} into buffer...{Style.RESET_ALL}")
    
    try:
        with open(progress_file, 'rb') as f:
            completed_games_cache = pickle.load(f)
    except Exception as e:
        print(f"{Fore.RED}❌ Error loading self-play progress file: {e}{Style.RESET_ALL}")
        return

    games_added = 0
    steps_added = 0
    for game_data in completed_games_cache:
        _, play_data, _ = game_data
        if not play_data:
            continue
        
        games_added += 1
        for step_data in play_data:
            if len(train_order) == config.TRAIN_BUFFER_SIZE:
                oldest_id = train_order.popleft()
                if oldest_id in train_data: del train_data[oldest_id]
                if oldest_id in train_priorities: del train_priorities[oldest_id]

            new_id = next_data_id
            train_data[new_id] = step_data
            train_priorities[new_id] = 1.0
            train_order.append(new_id)
            next_data_id += 1
            steps_added += 1

    print(f"  {Style.DIM}- Merged {games_added} games ({steps_added} steps). New Buffer Size: {len(train_data)} / {config.TRAIN_BUFFER_SIZE}{Style.RESET_ALL}")

    print(f"  {Style.DIM}- Saving updated buffer to disk...{Style.RESET_ALL}")
    save_buffer() 
    
    try:
        os.remove(progress_file)
        print(f"  {Style.DIM}- Cleaned up temporary file: {os.path.basename(progress_file)}{Style.RESET_ALL}")
    except OSError as e:
        print(f"{Fore.YELLOW}Warning: Could not remove processed self-play file: {e}{Style.RESET_ALL}")

def run_training_attempt(effective_iteration, writer, hyperparams, attempt_name="Training", is_verification=False):
    global current_train_step
    
    lr, wd, bs, steps = hyperparams['lr'], hyperparams['weight_decay'], hyperparams['batch_size'], hyperparams['train_steps']
    
    start_step = 0 if is_verification else current_train_step

    if not is_verification and start_step > 0 and start_step >= steps:
        print(f"  {Fore.YELLOW}--> Warning: New target steps ({steps}) is less than or equal to previously completed steps ({start_step}).{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}--> Action: Resetting training for this attempt to start from step 0.{Style.RESET_ALL}")
        start_step = 0
        current_train_step = 0
    
    if start_step < steps:
        current_model.load_state_dict(best_model.state_dict())
        current_model.to(config.DEVICE).train()
    
        local_optimizer = optim.Adam(current_model.parameters(), lr=lr, weight_decay=wd)
        local_scheduler = ReduceLROnPlateau(local_optimizer, mode='min', factor=0.5, patience=3)
        
        if len(train_data) < bs:
            print(f"  {Fore.RED}--> Error: Buffer size ({len(train_data)}) too small for batch size ({bs}). Skipping.{Style.RESET_ALL}")
            return {"success": False, "replay_data": None}
        
        losses = []
        pbar_format = "  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        
        with tqdm(total=steps, initial=start_step, desc=f"--> Train", leave=True, ncols=100, bar_format=pbar_format, ascii=True) as pbar:
            last_saved_step = (start_step // config.TRAIN_SAVE_INTERVAL) * config.TRAIN_SAVE_INTERVAL
            for step in range(start_step, steps):
                if PAUSE_TRAINING_EVENT.is_set(): time.sleep(1)
                
                batch_data = get_training_batch(bs)
                if batch_data is None: 
                    time.sleep(0.1)
                    continue
                
                if not is_verification:
                    current_train_step = step

                global_tensorboard_step = (effective_iteration * config.DEFAULT_TRAIN_STEPS) + step 
                loss_stats = perform_train_step(current_model, local_optimizer, scaler, batch_data, writer, global_tensorboard_step)
                losses.append(loss_stats)
                
                postfix_dict = {
                    "Loss": f"{np.mean([l[0] for l in losses]):.4f}" if losses else "N/A"
                }
                
                if not is_verification:
                    saved_str = f"{last_saved_step}"
                    if (step + 1) % config.TRAIN_SAVE_INTERVAL == 0 and step < steps - 1:
                        save_checkpoint(is_best=False, step_override=step + 1)
                        last_saved_step = step + 1
                        saved_str = f"{last_saved_step}"
                    postfix_dict["Saved"] = saved_str
                
                postfix_str = ", ".join([f"{k}: {v}" for k, v in postfix_dict.items()])
                pbar.set_postfix_str(postfix_str, refresh=True)

                pbar.update(1)

        if not is_verification:
            current_train_step = steps
    else:
        print(f"  {Fore.YELLOW}--> Info: Training for this attempt already completed ({start_step}/{steps} steps). Skipping to evaluation.{Style.RESET_ALL}")

    if 'losses' in locals() and losses: 
        avg_losses = np.mean(losses, axis=0)
        if 'local_scheduler' in locals(): local_scheduler.step(avg_losses[0])
        print(f"    Avg Losses -> Total: {avg_losses[0]:.4f} (Policy: {avg_losses[1]:.4f}, Value: {avg_losses[2]:.4f}, Outcome: {avg_losses[3]:.4f})")

    challenger_path, defender_path = os.path.join(config.CHECKPOINT_DIR, "temp_challenger.pth.tar"), os.path.join(config.CHECKPOINT_DIR, "temp_defender.pth.tar")
    torch.save({'state_dict': current_model.state_dict()}, challenger_path)
    torch.save({'state_dict': best_model.state_dict()}, defender_path)
    
    eval_desc = "Eval"
    eval_results = run_evaluation_duel(challenger_path, defender_path, desc=eval_desc)
    os.remove(challenger_path); os.remove(defender_path)
    return eval_results

def run_hyperparameter_search(effective_iteration, writer, initial_params, initial_results):
    storage_path = f"sqlite:///{os.path.join(config.CHECKPOINT_DIR, 'optuna.db')}"
    study_name = f"iter-{effective_iteration+1}-search"
    
    print(f"\n{Fore.CYAN}  --> Attempting with Hyperparameter Search (Optuna)...{Style.RESET_ALL}")
    print(f"  L--> Launching Optuna search (Study: '{study_name}')")
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(study_name=study_name, storage=storage_path, direction="maximize", pruner=optuna.pruners.MedianPruner(), load_if_exists=True)

    if len(study.trials) == 0 and initial_results:
        initial_score = (initial_results['win_rate'] * 1000) + (initial_results['avg_game_length'] * 0.1)
        
        param_distributions = {
            'lr': optuna.distributions.FloatDistribution(1e-5, 5e-3, log=True),
            'weight_decay': optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
            'batch_size': optuna.distributions.CategoricalDistribution([256, 512, 1024, 2048]),
            'train_steps': optuna.distributions.IntDistribution(400, 2000)
        }
        
        study.add_trial(
            optuna.trial.create_trial(
                params=initial_params, 
                distributions=param_distributions,
                value=initial_score
            )
        )
        
        print(f"    {Style.DIM}- Optuna seeded with default trial results (Score: {initial_score:.2f}).{Style.RESET_ALL}")

    n_trials_to_run = config.OPTUNA_N_TRIALS - len(study.trials)
    if n_trials_to_run <= 0:
        print(f"    {Fore.YELLOW}- Study already completed with {len(study.trials)} trials. No new trials to run.{Style.RESET_ALL}")
        if study.best_value > config.EVAL_WIN_RATE * 1000:
             print(f"    {Fore.GREEN}- Historical best trial meets success criteria. Concluding search.{Style.RESET_ALL}")
             return True, study.best_params, None
        return False, None, None

    print(f"    {Style.DIM}- Resuming search, {n_trials_to_run}/{config.OPTUNA_N_TRIALS} trials remaining...{Style.RESET_ALL}")
    
    for i in range(n_trials_to_run):
        trial = study.ask({
            'lr': optuna.distributions.FloatDistribution(1e-5, 5e-3, log=True),
            'weight_decay': optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
            'batch_size': optuna.distributions.CategoricalDistribution([256, 512, 1024, 2048]),
            'train_steps': optuna.distributions.IntDistribution(400, 2000)
        })
        
        if len(train_data) < trial.params['batch_size']:
            print(f"    {Fore.YELLOW}- Optuna trial #{trial.number} skipped: Buffer size ({len(train_data)}) too small for batch size ({trial.params['batch_size']}).{Style.RESET_ALL}")
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
            
        attempt_name = f"Optuna #{trial.number}"
        print(f"\n{Fore.CYAN}  --> Attempting with {attempt_name}...{Style.RESET_ALL}")
        CLog.info_block("Params:", trial.params, indent=4)
        
        results = run_training_attempt(effective_iteration, writer, trial.params, attempt_name, is_verification=True)
        
        score = (results['win_rate'] * 1000) + (results['avg_game_length'] * 0.1)
        study.tell(trial, score)
        
        if results["success"]:
            print(f"  {Fore.GREEN}[✅] Result: {attempt_name} SUCCEEDED. Score: {score:.0f}{Style.RESET_ALL}")
            return True, trial.params, results
        else:
            print(f"  {Fore.RED}[❌] Result: {attempt_name} FAILED. Score: {score:.0f}{Style.RESET_ALL}")

    print(f"\n{Fore.RED}--> Search finished after {config.OPTUNA_N_TRIALS} trials without finding a winning configuration.{Style.RESET_ALL}")
    return False, None, None


def handle_training_cycle(iter_to_train, writer):
    print(f"\n{Fore.BLUE}[>] Phase 2/2: TRAINING for Iteration {iter_to_train + 1}{Style.RESET_ALL}")
    
    load_checkpoint(resume=True)

    print(f"{Fore.CYAN}  --> Attempting with Default Parameters...{Style.RESET_ALL}")
    CLog.info_block("Params:", {'Learning Rate': config.DEFAULT_LEARNING_RATE, 'Weight Decay': config.DEFAULT_WEIGHT_DECAY, 'Batch Size': config.DEFAULT_BATCH_SIZE, 'Train Steps': config.DEFAULT_TRAIN_STEPS}, indent=4)
    default_params = {'lr': config.DEFAULT_LEARNING_RATE, 'weight_decay': config.DEFAULT_WEIGHT_DECAY, 'batch_size': config.DEFAULT_BATCH_SIZE, 'train_steps': config.DEFAULT_TRAIN_STEPS}
    default_results = run_training_attempt(iter_to_train, writer, default_params, "Default", is_verification=False)
    
    score = (default_results['win_rate'] * 1000) + (default_results['avg_game_length'] * 0.1)
    
    if default_results["success"]:
        print(f"  {Fore.GREEN}[✅] Result: Default training SUCCEEDED. Score: {score:.0f}{Style.RESET_ALL}")
        return True, default_results['replay_data'], "default"
    
    print(f"  {Fore.RED}[❌] Result: Default training FAILED. Score: {score:.0f}{Style.RESET_ALL}")

    search_success, best_params, final_results = run_hyperparameter_search(iter_to_train, writer, default_params, default_results)
    
    if search_success:
        attempt_name = f"optuna_{get_unique_timestamp()}"
        return True, final_results['replay_data'], attempt_name
        
    return False, None, None

def display_startup_menu():
    global iteration, current_train_step
    
    print(f"\n{Style.DIM}--------------------------------------------------------------------------------{Style.RESET_ALL}")
    print(f"{Fore.CYAN}                             -- ACTION REQUIRED --                              {Style.RESET_ALL}")
    print(f"{Style.DIM}--------------------------------------------------------------------------------{Style.RESET_ALL}")

    load_checkpoint(resume=True)
    effective_iteration = iteration
    
    current_iter_sp_data = load_selfplay_progress(effective_iteration)
    is_current_sp_complete = len(current_iter_sp_data) >= config.SELF_PLAY_GAMES_PER_ITER
    
    choices = []
    
    if effective_iteration == 0 and not current_iter_sp_data:
        print(f"{Fore.CYAN}[i] Current State: Ready to start fresh.{Style.RESET_ALL}")
        choices.append({'text': f"Start {Fore.BLUE}[SELF-PLAY]{Style.RESET_ALL} for Iteration 1", 'action': 'self_play', 'iter': 0})
    
    elif is_current_sp_complete:
        status = "IN PROGRESS" if current_train_step > 0 else "Ready to start"
        print(f"{Fore.CYAN}[i] Current State: {status} [TRAINING] for Iteration {effective_iteration + 1}.{Style.RESET_ALL}")
        choices.append({'text': f"Start/Resume {Fore.BLUE}[TRAINING]{Style.RESET_ALL} for Iteration {effective_iteration + 1}", 'action': 'train', 'iter': effective_iteration})

    else:
        status = "IN PROGRESS" if current_iter_sp_data else "Ready to start"
        game_status = f"({len(current_iter_sp_data)}/{config.SELF_PLAY_GAMES_PER_ITER} games)" if current_iter_sp_data else ""
        print(f"{Fore.CYAN}[i] Current State: {status} [SELF-PLAY] for Iteration {effective_iteration + 1} {game_status}.{Style.RESET_ALL}")
        choices.append({'text': f"Start/Resume {Fore.BLUE}[SELF-PLAY]{Style.RESET_ALL} for Iteration {effective_iteration + 1}", 'action': 'self_play', 'iter': effective_iteration})

    prev_iter = effective_iteration - 1
    if prev_iter >= 0 and len(load_selfplay_progress(prev_iter)) >= config.SELF_PLAY_GAMES_PER_ITER:
        choices.append({'text': f"Redo {Fore.BLUE}[TRAINING]{Style.RESET_ALL} for previous Iteration {effective_iteration}", 'action': 'redo_train', 'iter': prev_iter})

    print()
    for i, choice in enumerate(choices): print(f"  {Fore.YELLOW}[{i+1}]{Style.RESET_ALL} {choice['text']}")
    
    user_input = ''
    while not user_input.isdigit() or not (1 <= int(user_input) <= len(choices)):
        user_input = input(f"\nEnter your choice: {Fore.YELLOW}")
    print(Style.RESET_ALL, end="")
    
    return choices[int(user_input) - 1]

# =====================================================================
#                        主函数 (智能调度器)
# =====================================================================
def main():
    global iteration, current_train_step
    log_dir = os.path.join(config.CHECKPOINT_DIR, 'logs'); os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    parser = argparse.ArgumentParser(description="AlphaZero Gomoku - V55.9")
    parser.add_argument('--clean', action='store_true', help='Start a completely fresh training run, deleting checkpoints.')
    parser.add_argument('--no-gui', action='store_true', help='Run in headless mode without the web UI.')
    parser.add_argument('--debug-selfplay', action='store_true', help='Enable detailed logging for self-play workers.')
    args = parser.parse_args()

    if args.clean and os.path.exists(config.CHECKPOINT_DIR): shutil.rmtree(config.CHECKPOINT_DIR)
    
    load_checkpoint(resume=not args.clean)
    
    print(f"\n{Style.BRIGHT}Gomoku AI - AlphaZero V55.9 - PID: {os.getpid()}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}[✅] System Initialized.{Style.RESET_ALL}")
    print(f"  {Style.DIM}- Device        : {str(config.DEVICE).upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  {Style.DIM}- Workers       : {config.NUM_SELF_PLAY_WORKERS} CPU, {config.MAX_GPU_WORKERS_SELF_PLAY} on GPU")
    print(f"{Fore.GREEN}[✅] Checkpoint Loaded.{Style.RESET_ALL}")
    print(f"  {Style.DIM}- Iteration     : {iteration + 1}")
    
    hof_dir = os.path.join(config.CHECKPOINT_DIR, "hall_of_fame")
    hof_model_count = 0
    if os.path.exists(hof_dir):
        hof_model_count = len(glob.glob(os.path.join(hof_dir, "*.pth.tar")))
    print(f"  {Style.DIM}- Hall of Fame  : {hof_model_count} models")

    if not args.clean:
        load_buffer()

    print(f"  {Style.DIM}- Buffer Size   : {len(train_data)} / {config.TRAIN_BUFFER_SIZE}")

    if not args.no_gui:
        web_server_thread = threading.Thread(target=run_web_server, daemon=True); web_server_thread.start(); time.sleep(1)
        if web_server_thread.is_alive():
            print(f"{Fore.GREEN}[✅] Web Server Started at http://{config.HOST}:{config.PORT}{Style.RESET_ALL}")
            try: webbrowser.open_new_tab(f"http://{config.HOST}:{config.PORT}")
            except Exception: pass
        else: print(f"{Fore.RED}[❌] Web server thread failed to start.{Style.RESET_ALL}")
    
    try:
        action_to_run = display_startup_menu()

        while True:
            if PAUSE_TRAINING_EVENT.is_set(): time.sleep(1); continue


            if action_to_run['action'] == 'self_play':
                iter_to_process = action_to_run['iter']
                print(f"\n{Style.BRIGHT}================= AUTO-START: SELF-PLAY (Iter {iter_to_process+1}) ================{Style.RESET_ALL}")
                run_self_play_phase(iter_to_process, args)

                add_completed_selfplay_to_buffer(iter_to_process)
                
                action_to_run = {'action': 'train', 'iter': iter_to_process}
                continue


            elif action_to_run['action'] in ['train', 'redo_train']:
                is_redo = action_to_run['action'] == 'redo_train'
                iter_to_train = action_to_run['iter']
                
                if is_redo:
                    print(f"\n{Style.BRIGHT}================ MANUAL START: REDO TRAINING (Iter {iter_to_train+1}) ================{Style.RESET_ALL}")
                else:
                    print(f"\n{Style.BRIGHT}================== AUTO-START: TRAINING (Iter {iter_to_train+1}) ================={Style.RESET_ALL}")

                was_successful, winning_replay, attempt_name = handle_training_cycle(iter_to_train, writer)
                
                if was_successful:
                    print(f"\n{Fore.GREEN}[✅] SUCCESS: Training cycle for iteration {iter_to_train + 1} was successful.{Style.RESET_ALL}")
                    update_best_model_and_hof(iter_to_train, attempt_name)
                    
                    label_iter = iter_to_train + 1
                    save_replay_file(winning_replay, label_iter, attempt_name)
                    visualize_policy(best_model, label_iter, attempt_name)
                else:
                    print(f"\n{Fore.RED}[❌] FAILURE: Training cycle for iteration {iter_to_train + 1} failed to improve the model.{Style.RESET_ALL}")
                
                if not is_redo:
                    iteration += 1
                
                current_train_step = 0
                save_checkpoint(is_best=was_successful, step_override=0)

                action_to_run = {'action': 'self_play', 'iter': iteration}
                continue
            
    except (KeyboardInterrupt, Exception):
        print(f"\n{Fore.RED}🛑 An error occurred or training was interrupted by user...{Style.RESET_ALL}")
        import traceback; traceback.print_exc()
    finally:
        if 'writer' in locals() and writer is not None: writer.close()
        print(f"\n{Fore.YELLOW}Finalizing... Saving session state...{Style.RESET_ALL}")
        save_checkpoint(is_best=False); save_buffer()
        print(f"{Fore.GREEN}Training stopped. Session saved. Done.{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    from werkzeug.serving import run_simple, WSGIRequestHandler
    main()
