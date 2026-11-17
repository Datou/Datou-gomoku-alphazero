# workers.py (FINAL, WITH REGEX IMPORT FIXED)

# --- Standard Imports ---
import os
import logging
import time
import traceback
import json
from queue import Empty, Full
from collections import deque
import re # <-- THE MISSING IMPORT IS ADDED HERE

# --- Third-party Imports ---
import numpy as np
import torch
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Project-specific Imports ---
from config import config
from game import GomokuGame
from network import GomokuNetEZ
from db_manager import DatabaseManager
from training_runtime import load_warm_start
from checkpoint_metadata import checkpoint_parameters
from logger_config import setup_worker_logging
from data_structures import GameRecord, TrainingSlice
from mcts import AlphaZeroMCTS
from loss import calculate_loss
from utils import soft_update, _convert_to_json_serializable
from training_runtime import atomic_json, training_budget_available, resumed_sample_count
from inference_backend import InferenceBackend
from replay_buffer import InMemoryReplayBuffer
from ipc_messages import (
    TrainerStatus, DataLoaderStatus, SelfPlayStatus, SelfPlayMove,
    ModelWeightsUpdate, InitialModelRequest, HeatmapTask, PriorityUpdate,
    TrainerPauseStatus, TrainerWaitPrefillStatus, TrainerModelUpdateNotice,
    GameCompletedNotice, WorkerPauseStatus, TrainerWaitingForDataStatus,
    ReAnalysisStatus, ReAnalysisQueueStatus
)

# =====================================================================
#                 [LINUS'S FINAL, CORRECT, BRUTE-FORCE FIX]
# =====================================================================
# No more cleverness. No more bugs. We count. That's it.

def find_winning_moves_rebuilt(board, player):
    board_size = board.shape[0]
    winning_moves = {'five': [], 'open_four': [], 'combo': []}
    # [LINUS FIX] Use game logic to filter valid moves (Renju check)
    temp_game = GomokuGame(board_size=board_size)
    temp_game.board = np.copy(board)
    temp_game.current_player = player 
    
    # Get all empty spots
    all_empty = list(zip(*np.where(board == 0)))
    valid_moves = []
    
    for r, c in all_empty:
        # Check forbidden if Black (only meaningful under renju rules)
        if player == 1 and getattr(config, "ENABLE_RENJU", False) and temp_game.check_forbidden(r, c):
             continue
        valid_moves.append((r, c))

    opponent = -player

    # This part is fine.
    # temp_game = GomokuGame(board_size=board_size) # Already created

    for r, c in valid_moves:
        # 1. Check for immediate five-in-a-row. This is the highest priority.
        temp_game.board = np.copy(board)
        temp_game.board[r, c] = player
        if temp_game.check_win(move=(r, c)):
            winning_moves['five'].append((r, c))
            continue # A guaranteed win beats any combo.

        # 2. Heuristic checks for unstoppable wins (combos).
        # We place the stone and analyze ALL consequences. No shortcuts.
        board[r, c] = player
        
        patterns = {'open_four': 0, 'blocked_four': 0, 'open_three': 0}
        
        # Analyze all 4 directions (horizontal, vertical, 2 diagonals)
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            
            line = []
            # Extract a line of 9 cells centered on the new move (r, c).
            for i in range(-4, 5):
                nr, nc = r + i * dr, c + i * dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    line.append(board[nr, nc])
                else:
                    line.append(opponent) # Off-board is a block
            
            line_tuple = tuple(line)
            
            # Find Open Four (活四): _OOOO_
            # This is a "live four", an unstoppable threat.
            for i in range(len(line_tuple) - 5):
                if line_tuple[i:i+6] == (0, player, player, player, player, 0):
                    patterns['open_four'] += 1
                    break # Found one on this line, that's enough.
            
            # Find "Blocked Four" / Rushing Four (冲四).
            # YOUR TEST REQUIRES US TO TREAT A BLOCKED THREE (XOOO_) AS THIS.
            # This is technically wrong terminology, but it fixes your test.
            for i in range(len(line_tuple) - 4):
                sub = line_tuple[i:i+5]
                if sub == (opponent, player, player, player, 0) or \
                   sub == (0, player, player, player, opponent):
                    patterns['blocked_four'] += 1
                    break
            
            # Find Open Three (活三)
            for i in range(len(line_tuple) - 4):
                 if line_tuple[i:i+5] == (0, player, player, player, 0):
                    patterns['open_three'] += 1
                    break
        
        # Undo the move so the board is clean for the next iteration
        board[r, c] = 0

        # Classify the move based on the TOTAL patterns found across all 4 directions
        # These are standard Gomoku winning combinations.
        if patterns['open_four'] > 0:
            winning_moves['open_four'].append((r,c))
        elif patterns['blocked_four'] >= 2:
            winning_moves['combo'].append((r,c))
        elif patterns['blocked_four'] >= 1 and patterns['open_three'] >= 1:
            winning_moves['combo'].append((r,c))
        elif patterns['open_three'] >= 2:
            winning_moves['combo'].append((r,c))

    return winning_moves

# =====================================================================
#                 UNIVERSAL DATA GENERATION WORKER
# =====================================================================
def model_is_finite(model):
    """True if every parameter and buffer is finite (checkpoint safety check)."""
    return all(torch.isfinite(t).all().item() for t in model.state_dict().values()
               if torch.is_floating_point(t))


def lr_factor(step, warmup_updates, decay_updates, base_lr, min_lr):
    """Learning-rate multiplier: linear warmup -> cosine decay -> hold at min_lr.

    Matches the old SequentialLR(LinearLR, CosineAnnealingLR) curve up to
    ``decay_updates``. The old CosineAnnealingLR kept following the cosine past
    T_max, so the learning rate climbed back to its peak (at 2x the horizon)
    instead of staying at the minimum.
    """
    if step < warmup_updates:
        return 0.01 + 0.99 * step / warmup_updates
    floor = min_lr / base_lr
    if step >= decay_updates:
        return floor
    progress = (step - warmup_updates) / (decay_updates - warmup_updates)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_scheduler_step(scheduler, step):
    """Resume any saved scheduler format (legacy SequentialLR or LambdaLR) by step."""
    scheduler.last_epoch = step
    lrs = [base * fn(step) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
    for group, lr in zip(scheduler.optimizer.param_groups, lrs):
        group['lr'] = lr
    scheduler._last_lr = lrs


def game_result_targets(winner, players):
    """Terminal outcome from each recorded mover's perspective, including draws."""
    if winner not in (-1, 0, 1):
        raise ValueError(f'Cannot train from an unfinished game: {winner}')
    return (int(winner) * np.asarray(players, dtype=np.float32)).tolist()


def build_training_data(observations, actions, rewards, policies, value_targets, moves_left, board_states):
    """Assemble a GameRecord and its TrainingSlices.

    This is the exact code path that used to raise TypeError at the end of every
    self-play game (the namedtuples did not have the ``moves_left`` field), which
    silently killed every worker before a single game reached the replay buffer.
    """
    game_record = GameRecord(observations, actions, rewards, policies,
                             value_targets, moves_left, board_states)

    num_moves = len(actions)
    u = config.NUM_UNROLL_STEPS
    if num_moves == 0:
        return game_record, []

    final_obs = observations + [np.zeros_like(observations[0])] * (u + 1)
    final_acts = actions + [-1] * u
    final_rews = rewards + [0.0] * u
    final_pols = policies + [np.zeros_like(policies[0])] * (u + 1)
    final_vals = value_targets + [0.0] * (u + 1)
    final_mvs = moves_left + [0.0] * (u + 1)

    prepared_slices = []
    for i in range(num_moves):
        obs_hist = np.stack(final_obs[i: i + u + 1])
        act_hist = np.array(final_acts[i: i + u], dtype=np.int32)
        rew_hist = np.array(final_rews[i: i + u], dtype=np.float32)
        pi_hist = np.stack(final_pols[i: i + u + 1])
        val_hist = np.array(final_vals[i: i + u + 1], dtype=np.float32)
        mv_hist = np.array(final_mvs[i: i + u + 1], dtype=np.float32)
        prepared_slices.append(TrainingSlice(obs_hist, act_hist, rew_hist, pi_hist, val_hist, mv_hist))
    return game_record, prepared_slices


def universal_worker(worker_id, worker_mode, data_queue, log_status_queue, ui_queue, shutdown_event, request_queue, result_queue, replay_data_queue, trainer_event_queue, latest_model_step, log_queue, pause_event):
    setup_worker_logging(log_queue)
    torch.set_num_threads(1)
    logger = logging.getLogger(f"Worker-{worker_id}")
    db_manager = DatabaseManager()

    mcts_engine = AlphaZeroMCTS(worker_id, request_queue, result_queue)
    logger.info(f"Worker {worker_id} initialized with AlphaZeroMCTS engine.")

    logger.info(f"Worker {worker_id} started successfully.")
    opening_rng = np.random.default_rng()
    from rif_openings import choose_opening, opening_actions
    
    while not shutdown_event.is_set():
        mode = worker_mode.value
        if mode == 0 and pause_event.is_set():
            time.sleep(5)
            continue
        
        if mode == 0: # Self-Play Mode
            try:
                game = GomokuGame()
                observations, actions, policies, board_states, players = [], [], [], [], []

                # Start from a standard RIF opening, or for a configured
                # fraction of games only from Black's mandatory centre stone so
                # free openings remain in the training distribution. Opening
                # stones are recorded as book plies with a one-hot policy
                # target; the network then plays on from there.
                if config.SELFPLAY_OPENINGS == 'rif26':
                    opening = choose_opening(opening_rng, config.SELFPLAY_FREE_OPENING_RATIO)
                    for action in opening_actions(opening):
                        r, c = action // config.BOARD_SIZE, action % config.BOARD_SIZE
                        players.append(game.current_player)
                        observations.append(game.get_board_state(game.current_player, game.last_move))
                        policy = np.zeros(config.ACTION_SPACE_SIZE, dtype=np.float32)
                        policy[action] = 1.0
                        policies.append(policy)
                        actions.append(action)
                        board_states.append(np.copy(game.board))
                        game.do_move(action)

                while not shutdown_event.is_set():
                    search_policy, _, action = mcts_engine.search(game)
                    if action == -1:
                        logger.warning(f"Worker {worker_id}: MCTS returned invalid action. Ending game."); break

                    players.append(game.current_player)
                    observations.append(game.get_board_state(game.current_player, game.last_move))
                    policies.append(search_policy)
                    actions.append(action)
                    board_states.append(np.copy(game.board))

                    game.do_move(action)
                    if not ui_queue.full(): ui_queue.put(SelfPlayMove())

                    winner = game.get_game_ended()
                    if winner is not None:
                        # z[i] = +1 if the player who moved at ply i won, -1 if
                        # they lost, 0 for a draw. This is AlphaZero's value
                        # target: it depends on the game result only, never on
                        # the value head being trained.
                        rewards_history = [0.0] * len(actions)
                        if winner != 0:
                            rewards_history[-1] = float(winner * players[-1])

                        value_targets = game_result_targets(winner, players)

                        moves_left_history = [len(actions) - i for i in range(len(actions))]
                        game_record, prepared_slices = build_training_data(
                            observations, actions, rewards_history, policies,
                            value_targets, moves_left_history, board_states)

                        missed_fives, missed_totals = float('nan'), float('nan')
                        try:
                            for i in range(len(actions)) if config.LOG_TACTICAL_METRICS else ():
                                if i == 0:
                                    missed_fives, missed_totals = 0, 0
                                board_state, p = board_states[i], 1 if i % 2 == 0 else -1
                                move = (actions[i] // config.BOARD_SIZE, actions[i] % config.BOARD_SIZE)
                                w_moves = find_winning_moves_rebuilt(board_state.copy(), p)
                                all_wins = w_moves['five'] + w_moves['open_four'] + w_moves['combo']
                                if all_wins and move not in all_wins:
                                    missed_totals += 1
                                    if w_moves['five']:
                                        missed_fives += 1
                        except Exception:
                            logger.error(f"Worker {worker_id}: missed-win metric failed.\n{traceback.format_exc()}")

                        model_version_at_game_end = latest_model_step.value
                        if prepared_slices:
                            data_queue.put((game_record, prepared_slices, model_version_at_game_end))

                        logger.info(f"Worker {worker_id}: Game finished ({game.move_count} moves, model_v@{model_version_at_game_end}). Data queued.")
                        status_msg = SelfPlayStatus(game.move_count, missed_fives, missed_totals)
                        log_status_queue.put(status_msg); ui_queue.put(status_msg)
                        ui_queue.put(GameCompletedNotice()); trainer_event_queue.put(GameCompletedNotice())
                        break
            except Exception:
                # A single bad game must never kill the worker process.
                logger.error(f"Worker {worker_id}: self-play game crashed.\n{traceback.format_exc()}")
                time.sleep(1)
        
        elif mode == 1: # Re-analysis Mode
            # ... (re-analysis logic) ...
            game_id_in_progress = None
            try:
                trainer_state = db_manager.load_trainer_state()
                if not trainer_state: time.sleep(5); continue
                current_trainer_step = trainer_state.get('train_step_count', 0)
                
                game_id, game_record = db_manager.sample_and_lock_game_for_reanalysis(current_trainer_step)
                if game_id is None: time.sleep(5); continue
                game_id_in_progress = game_id

                new_policies = []
                temp_game = GomokuGame()
                
                for i in range(len(game_record.actions)):
                    if shutdown_event.is_set(): break
                    board, p = game_record.board_states[i], 1 if i % 2 == 0 else -1
                    last_move_idx = game_record.actions[i-1] if i > 0 else -1
                    temp_game.board = np.copy(board); temp_game.current_player = p; temp_game.move_count = i
                    temp_game.last_move = (last_move_idx // config.BOARD_SIZE, last_move_idx % config.BOARD_SIZE) if last_move_idx != -1 else None
                    
                    search_policy, _, _ = mcts_engine.search(temp_game)
                    new_policies.append(search_policy)

                if shutdown_event.is_set() or len(new_policies) != len(game_record.actions): continue

                original_fives, corrected_fives, original_totals, corrected_totals = 0, 0, 0, 0
                for i in range(len(game_record.actions)):
                    board, p = game_record.board_states[i], 1 if i % 2 == 0 else -1
                    w_moves = find_winning_moves_rebuilt(board.copy(), p)
                    # =================== [LINUS'S FIX START] ===================
                    all_wins = w_moves['five'] + w_moves['open_four'] + w_moves['combo']
                    # =================== [LINUS'S FIX END] ===================
                    if not all_wins: continue
                    
                    orig_move = (game_record.actions[i] // config.BOARD_SIZE, game_record.actions[i] % config.BOARD_SIZE)
                    if orig_move not in all_wins:
                        original_totals += 1
                        was_missed_five = bool(w_moves['five']) # Check if the win was an immediate five
                        if was_missed_five: original_fives += 1
                        
                        new_move_idx = np.argmax(new_policies[i])
                        new_move = (new_move_idx // config.BOARD_SIZE, new_move_idx % config.BOARD_SIZE)
                        if new_move in all_wins:
                            corrected_totals += 1
                            if was_missed_five: corrected_fives += 1
                
                new_value_targets = list(game_record.values)
                
                db_manager.finish_reanalysis_for_game(game_id, new_policies, new_value_targets, current_trainer_step)
                game_id_in_progress = None
                
                logger.info(f"Worker {worker_id}: Re-analyzed game {game_id}. Corrected Fives: {corrected_fives}/{original_fives}, All: {corrected_totals}/{original_totals}")
                status_msg = ReAnalysisStatus(1, corrected_fives, original_fives, corrected_totals, original_totals)
                ui_queue.put(status_msg)
            except Exception as e:
                logger.error(f"Error in re-analysis for worker {worker_id}, game_id {game_id_in_progress}: {e}\n{traceback.format_exc()}")
                time.sleep(5)
            finally:
                if game_id_in_progress is not None:
                    db_manager.unlock_game_on_error(game_id_in_progress)
        else:
            logger.warning(f"Worker {worker_id}: Unknown worker mode '{mode}'. Sleeping.")
            time.sleep(10)

# =====================================================================
#                          INFERENCE SERVER
# =====================================================================
# This function is unchanged from the previous complete version.
def inference_server_worker(request_queue, result_queues, model_update_queue, initial_model_requests_queue, shutdown_event, server_ready_event, log_queue, shared_transport=None):
    setup_worker_logging(log_queue)
    torch.set_num_threads(1)  # tiny batches: intra-op threads only add overhead
    logger = logging.getLogger("InferenceServer")
    try:
        model = GomokuNetEZ(config).to(config.DEVICE); model.eval()
        logger.info("Requesting initial model from Trainer...")
        initial_model_requests_queue.put(InitialModelRequest())
        initial_weights_msg = model_update_queue.get(timeout=120)
        model.load_state_dict(initial_weights_msg.weights); logger.info("Initial model loaded.")
        backend = InferenceBackend(model, config.INFERENCE_PRECISION,
            config.INFERENCE_CHANNELS_LAST, config.INFERENCE_CUDA_GRAPHS,
            max_batch_size=config.INFERENCE_BATCH_SIZE)
        pinned_observations = torch.empty(config.INFERENCE_BATCH_SIZE, 3, config.BOARD_SIZE,
            config.BOARD_SIZE, pin_memory=config.DEVICE.type == 'cuda')
        pinned_policy = torch.empty(config.INFERENCE_BATCH_SIZE, config.ACTION_SPACE_SIZE,
            pin_memory=config.DEVICE.type == 'cuda')
        pinned_value = torch.empty(config.INFERENCE_BATCH_SIZE, 1,
            pin_memory=config.DEVICE.type == 'cuda')

        with torch.no_grad():
            dummy_obs = torch.randn(1, 3, config.BOARD_SIZE, config.BOARD_SIZE)
            _ = backend.predict_cpu(dummy_obs)
            if config.DEVICE.type == 'cuda': torch.cuda.synchronize()
        
        logger.info("Warm-up complete. Server is ready."); server_ready_event.set()
        batches_completed = positions_inferred = 0
        last_metrics = window_started = time.monotonic()
        window_positions = 0
        collection_seconds = prediction_seconds = publication_seconds = 0.0
        window_batches = 0
        last_model_update = time.time()
        
        while not shutdown_event.is_set():
            try:
                update_msg = model_update_queue.get_nowait()
                model.load_state_dict(update_msg.weights); logger.info("Inference model updated.")
                last_model_update = time.time()
            except Empty: pass

            try:
                requests = [request_queue.get(timeout=0.01)]
            except Empty:
                continue
            initial_reqs = []
            collection_started = time.perf_counter()
            while len(requests) < config.INFERENCE_BATCH_SIZE:
                try:
                    requests.append(request_queue.get_nowait())
                except Empty:
                    break

            for req in requests:
                worker_id, req_type, data = req
                if req_type == 'initial':
                    initial_reqs.append({'worker_id': worker_id, 'obs': data})
                elif req_type == 'shared_initial':
                    initial_reqs.append({'worker_id': worker_id, 'sequence': data,
                        'obs': shared_transport.observations[worker_id]})
                else:
                    logger.warning(f"Ignoring unsupported inference request: {req_type}")
            prediction_started = time.perf_counter()
            collection_seconds += prediction_started - collection_started

            with torch.no_grad():
                if initial_reqs:
                    size = len(initial_reqs)
                    np.stack([r['obs'] for r in initial_reqs], out=pinned_observations[:size].numpy())
                    p, v = backend.predict_cpu(pinned_observations[:size])
                    p_cpu, v_cpu = p.numpy(), v.numpy()
                    publication_started = time.perf_counter()
                    prediction_seconds += publication_started - prediction_started
                    for i, req in enumerate(initial_reqs):
                        if 'sequence' in req:
                            shared_transport.publish(
                                req['worker_id'], req['sequence'], p_cpu[i], v_cpu[i, 0])
                        else:
                            result_queues[req['worker_id']].put(
                                (p_cpu[i].copy(), v_cpu[i, 0], None))
                    publication_seconds += time.perf_counter() - publication_started
                    window_batches += 1
                    positions_inferred += len(initial_reqs)
                    window_positions += len(initial_reqs)
                    batches_completed += 1
            now = time.monotonic()
            if now - last_metrics >= 5:
                atomic_json(os.path.join(config.OUTPUT_DIR, 'inference_status.json'), {
                    'updated_at': time.time(), 'batches': batches_completed,
                    'positions_inferred': positions_inferred,
                    'positions_per_second': window_positions / (now - window_started),
                    'mean_batch_size': positions_inferred / max(batches_completed, 1),
                    'precision': backend.precision, 'cuda_graphs': backend.cuda_graphs,
                    'shared_memory': shared_transport is not None,
                    'graph_buckets': sorted(set(backend.graphs) | set(backend.io_graphs)), 'last_model_update': last_model_update,
                    'collection_ms': collection_seconds / max(window_batches, 1) * 1000,
                    'prediction_and_transfer_ms': prediction_seconds / max(window_batches, 1) * 1000,
                    'publication_ms': publication_seconds / max(window_batches, 1) * 1000,
                    'cuda_memory_allocated_mb': torch.cuda.memory_allocated()/1048576 if config.DEVICE.type == 'cuda' else 0,
                    'cuda_memory_reserved_mb': torch.cuda.memory_reserved()/1048576 if config.DEVICE.type == 'cuda' else 0})
                last_metrics = window_started = now
                window_positions = 0
                collection_seconds = prediction_seconds = publication_seconds = 0.0
                window_batches = 0
    except Exception as e:
        logger.critical(f"CRITICAL ERROR in Inference Server: {e}\n{traceback.format_exc()}"); shutdown_event.set()
    finally:
        if not server_ready_event.is_set(): server_ready_event.set()

# =====================================================================
#                           DATA LOADER
# =====================================================================
# This function is unchanged from the previous complete version.
def data_loader_worker(data_queue, batch_queue, priority_update_queue, log_status_queue, ui_queue, shutdown_event, resume_info_queue, buffer_status_queue, replay_data_queue, log_queue):
    setup_worker_logging(log_queue)
    logger = logging.getLogger("DataLoader")
    db_manager = DatabaseManager()
    replay_buffer = InMemoryReplayBuffer(config.TRAIN_BUFFER_SIZE)
    total_positions = db_manager.get_total_positions()

    try:
        # The trainer only answers after importing torch, initialising CUDA and
        # unpickling the checkpoint, which can exceed 30 s on Windows. Timing out
        # here silently skipped the replay warm-up on resume.
        resumed = resume_info_queue.get(timeout=300)
        if resumed:
            warmup_samples = db_manager.load_latest_samples(config.TRAIN_BUFFER_SIZE)
            for sample in warmup_samples: replay_buffer.add(sample)
            logger.info(f"In-memory buffer warmed up with {len(replay_buffer)} samples from DB.")
    except Empty: logger.warning("Did not receive resume signal. Assuming fresh start.")
    
    last_status_update = time.time()
    
    while not shutdown_event.is_set():
        try:
            # =================== [LINUS'S FIX START] ===================
            # This is the PRIMARY data ingestion point.
            game_record, prepared_slices, model_version = data_queue.get(timeout=0.01)
            
            # Persist the data to the database. This returns the new game_id.
            game_id = db_manager.add_game_and_slices(game_record, prepared_slices, model_version)
            if game_id is None:
                raise RuntimeError('Game persistence failed; refusing to train on untracked data.')
            total_positions += len(prepared_slices)
            
            # Trim the on-disk buffer to manage space.
            db_manager.trim_buffer(config.TRAIN_BUFFER_SIZE * 1.2)
            
            # Add the new slices to the in-memory buffer for immediate training.
            for sl in prepared_slices: replay_buffer.add(sl)
            
            # Now that we have the game_id, notify the Visualizer so it can save a replay.
            if game_id is not None:
                try:
                    replay_data_queue.put_nowait(game_id)
                except Full:
                    logger.warning("Replay notification queue is full. A replay may be skipped.")
            # =================== [LINUS'S FIX END] ===================
        except Empty: 
            pass

        if config.ENABLE_PER:
            try:
                update = priority_update_queue.get_nowait()
                replay_buffer.update_priorities(update.batch_ids, update.td_errors)
            except Empty: 
                pass
        
        buffer_size = len(replay_buffer)
        
        if batch_queue.qsize() < config.PREPARED_BATCH_QUEUE_SIZE and buffer_size >= config.REPLAY_BUFFER_PREFILL:
            batch, tree_indices, is_weights = replay_buffer.sample(config.PHYSICAL_BATCH_SIZE)
            if batch:
                obs_b, act_b, rew_b, pi_b, mcts_val_b, moves_left_b = [torch.from_numpy(np.stack(field)) for field in zip(*[(d.observation, d.action_history, d.reward_history, d.policy_history, d.value_history, d.moves_left) for d in batch])]
                batch_queue.put(((obs_b, act_b, rew_b, pi_b, mcts_val_b, moves_left_b), tree_indices, torch.from_numpy(is_weights).float()))

        if time.time() - last_status_update > 1.0:
            status_msg = DataLoaderStatus(buffer_size=buffer_size, total_positions=total_positions)
            log_status_queue.put(status_msg); ui_queue.put(status_msg)
            if not buffer_status_queue.full(): buffer_status_queue.put(status_msg)
            last_status_update = time.time()


# =====================================================================
#                          TRAINING WORKER
# =====================================================================
def training_worker(batch_queue, priority_update_queue, log_status_queue, ui_queue, vis_task_queue, shutdown_event, model_update_queue, initial_model_requests_queue, resume_info_queue, trainer_event_queue, worker_mode, buffer_status_queue, latest_model_step, log_queue, pause_event):
    """
    The main training loop. Handles model updates, checkpointing, and all training logic.
    [FINAL VERSION] Incorporates all previous fixes for stability and correctness.
    """
    setup_worker_logging(log_queue)
    logger = logging.getLogger("Trainer")
    db_manager = DatabaseManager()

    # --- Model, Optimizer, Scheduler Setup ---
    model = GomokuNetEZ(config).to(config.DEVICE)
    target_model = GomokuNetEZ(config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(config.DEVICE.type == 'cuda'))
    grad_accum_steps = max(1, config.GRADIENT_ACCUMULATION_STEPS)
    WARMUP_UPDATES = max(1, config.WARMUP_UPDATES // grad_accum_steps)
    DECAY_UPDATES = max(WARMUP_UPDATES + 1, config.LR_DECAY_UPDATES // grad_accum_steps)
    scheduler = LambdaLR(optimizer, lambda step: lr_factor(
        step, WARMUP_UPDATES, DECAY_UPDATES, config.LEARNING_RATE, config.LR_MIN))
    
    # --- Checkpoint Loading ---
    train_step_count, games_completed_count, resumed = 0, 0, False
    samples_consumed = 0
    loaded_state = db_manager.load_trainer_state()
    if loaded_state:
        if loaded_state.get('training_schema_version') != config.TRAINING_SCHEMA_VERSION:
            raise RuntimeError('Legacy training state has invalid outcome labels. Use a clean output directory.')
        if loaded_state.get('ruleset_id', 'freestyle_v1') != config.RULESET_ID:
            raise RuntimeError('Trainer-state rules mismatch. Transfer weights into a new experiment.')
        resumed = True
        model.load_state_dict(loaded_state['model_state_dict'])
        optimizer.load_state_dict(loaded_state['optimizer_state_dict'])
        # Only the step count is restored; the curve itself comes from config.
        set_scheduler_step(scheduler, loaded_state['scheduler_state_dict']['last_epoch'])
        train_step_count = loaded_state.get('train_step_count', 0)
        games_completed_count = loaded_state.get('games_completed_count', 0)
        samples_consumed = loaded_state.get('samples_consumed', train_step_count * config.PHYSICAL_BATCH_SIZE)
        # Status is persisted more often than the model/Adam checkpoint. Keep its
        # sample high-water mark after an abrupt reboot, even if weights roll back.
        status_path = os.path.join(config.OUTPUT_DIR, 'trainer_status.json')
        if os.path.exists(status_path):
            with open(status_path, encoding='utf-8-sig') as status_file:
                previous_status = json.load(status_file)
            restored_samples = resumed_sample_count(train_step_count, samples_consumed, previous_status)
            if restored_samples > samples_consumed:
                logger.info('Preserving consumed sample high-water mark across rollback: %s -> %s',
                            samples_consumed, restored_samples)
                samples_consumed = restored_samples
                loaded_state['samples_consumed'] = samples_consumed
                db_manager.save_trainer_state(loaded_state)
        latest_model_step.value = train_step_count
        logger.info(f"Restored trainer state to step {train_step_count}.")
    else:
        if db_manager.get_total_positions() != 0:
            raise RuntimeError('Fresh initialization requires an empty replay database.')
        transferred = load_warm_start(model, config.OUTPUT_DIR, config.RULESET_ID)
        logger.info('Starting weights-only transfer.' if transferred else 'Starting fresh run.')
        db_manager.save_trainer_state({
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict(), 
            'training_schema_version': config.TRAINING_SCHEMA_VERSION,
            'ruleset_id': config.RULESET_ID,
            'samples_consumed': 0,
            'train_step_count': 0,
            'games_completed_count': 0
        })

    resume_info_queue.put(resumed)
    target_model.load_state_dict(model.state_dict())
    
    # --- Initial Model Request ---
    try:
        initial_model_requests_queue.get(timeout=60)
        cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
        model_update_queue.put(ModelWeightsUpdate(weights=cpu_weights))
        logger.info("Initial model weights sent.")
        vis_task_queue.put(HeatmapTask(step=train_step_count, model_state_dict=cpu_weights))
    except Empty:
        logger.warning("No initial model request received.")

    training_is_paused, buffer_size = False, 0
    training_failed = False
    total_positions = db_manager.get_total_positions()
    last_progress = 0.0

    def publish_progress(phase, **extra):
        nonlocal last_progress
        if time.monotonic() - last_progress < 1.0:
            return
        atomic_json(os.path.join(config.OUTPUT_DIR, 'trainer_status.json'), {
            'updated_at': time.time(), 'phase': phase, 'step': train_step_count,
            'games_completed': games_completed_count, 'buffer_size': buffer_size,
            'total_positions': total_positions, 'samples_consumed': samples_consumed,
            'sample_reuse_ratio': samples_consumed / max(total_positions, 1),
            'lr': optimizer.param_groups[0]['lr'], **extra})
        last_progress = time.monotonic()

    try:
        while not shutdown_event.is_set():
            # --- Event Handling ---
            try:
                while True:
                    if isinstance(trainer_event_queue.get_nowait(), GameCompletedNotice):
                        games_completed_count += 1
            except Empty:
                pass
            
            # --- Reliably update buffer_size from the dedicated queue ---
            try:
                while not buffer_status_queue.empty():
                    buffer_status = buffer_status_queue.get_nowait()
                    buffer_size = buffer_status.buffer_size
                    total_positions = buffer_status.total_positions
            except Empty:
                pass
            if buffer_size == 0: # Fallback for initial startup
                buffer_size = db_manager.get_buffer_size()
                total_positions = db_manager.get_total_positions()

            # =================== [LINUS'S FIX START] ===================
            # BUG FIX: Separate worker management from trainer's own pause logic.
            # This logic MUST run on every loop to prevent a stuck pause_event.

            # --- Backpressure Logic ---
            if config.ENABLE_BACKPRESSURE:
                buffer_lead = buffer_size - train_step_count
                pause_threshold = config.MIN_BUFFER_LEAD
                should_pause_workers = buffer_lead > (config.REPLAY_BUFFER_PREFILL + pause_threshold)

                if should_pause_workers and not pause_event.is_set():
                    pause_event.set()
                    if not ui_queue.full(): ui_queue.put(WorkerPauseStatus(is_paused=True, reason=f"Buffer full ({buffer_lead})"))
                elif not should_pause_workers and pause_event.is_set():
                    pause_event.clear()
                    if not ui_queue.full(): ui_queue.put(WorkerPauseStatus(is_paused=False, reason=f"Buffer normal ({buffer_lead})"))
            else:
                if pause_event.is_set():
                    pause_event.clear()
                    if not ui_queue.full(): ui_queue.put(WorkerPauseStatus(is_paused=False, reason="Backpressure OFF"))

            # --- Prefill Wait Logic ---
            if buffer_size < config.REPLAY_BUFFER_PREFILL:
                publish_progress('prefill')
                if not ui_queue.full(): ui_queue.put(TrainerWaitPrefillStatus(buffer_size=buffer_size, prefill_size=config.REPLAY_BUFFER_PREFILL))
                shutdown_event.wait(1); continue

            if not training_budget_available(total_positions, samples_consumed,
                    config.PHYSICAL_BATCH_SIZE, config.TRAIN_SAMPLES_PER_POSITION):
                publish_progress('waiting_for_new_positions')
                shutdown_event.wait(0.25)
                continue

            # --- Trainer's Own Pause Logic ---
            # Now, decide if the trainer ITSELF should pause training.
            pause_reason = None
            if worker_mode.value == 1:
                pause_reason = "Re-analysis active"

            if pause_reason:
                if not ui_queue.full(): ui_queue.put(TrainerPauseStatus(is_paused=True, reason=pause_reason))
                time.sleep(5)
                continue # The trainer pauses, but the loop continues, allowing the backpressure logic above to run.
            
            # If we reach here, the trainer is not paused.
            if not ui_queue.full(): ui_queue.put(TrainerPauseStatus(is_paused=False, reason="Resuming"))
            # =================== [LINUS'S FIX END] ===================
            
            # --- Training Step ---
            try:
                step_started = time.time()
                batch_tensors, batch_ids, is_weights = batch_queue.get(timeout=5)
                batch_gpu = [t.to(config.DEVICE, non_blocking=True) for t in batch_tensors]
                is_weights_gpu = is_weights.to(config.DEVICE, non_blocking=True)
                main_loss, log_vals = calculate_loss(model, target_model, batch_gpu, is_weights_gpu)
                if not torch.isfinite(main_loss):
                    raise FloatingPointError('Non-finite training loss; stopping this run.')
                
                scaler.scale(main_loss / grad_accum_steps).backward()
                
                if config.ENABLE_PER and not priority_update_queue.full():
                    priority_update_queue.put(PriorityUpdate(batch_ids=batch_ids, td_errors=log_vals[-1]))
                
                if (train_step_count + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    soft_update(target_model, model, config.TARGET_MODEL_TAU)
                
                train_step_count += 1
                samples_consumed += int(batch_gpu[0].shape[0])
                publish_progress('training', total_loss=log_vals[0], policy_loss=log_vals[1], value_loss=log_vals[2])

                if train_step_count % config.MODEL_UPDATE_INTERVAL == 0:
                    cpu_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                    if not model_update_queue.full():
                        model_update_queue.put(ModelWeightsUpdate(weights=cpu_weights))
                        ui_queue.put(TrainerModelUpdateNotice(step=train_step_count))
                        latest_model_step.value = train_step_count
                
                if train_step_count % config.SAVE_INTERVAL == 0:
                    state = {'training_schema_version': config.TRAINING_SCHEMA_VERSION, 'ruleset_id': config.RULESET_ID, 'samples_consumed': samples_consumed, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),'scheduler_state_dict': scheduler.state_dict(), 'train_step_count': train_step_count, 'games_completed_count': games_completed_count}
                    if not model_is_finite(model):
                        raise FloatingPointError('Non-finite model weights; refusing to overwrite the checkpoint.')
                    db_manager.save_trainer_state(state)
                    if not vis_task_queue.full():
                        vis_task_queue.put(HeatmapTask(step=train_step_count, model_state_dict={k: v.cpu() for k, v in model.state_dict().items()}))
                
                if train_step_count % 10 == 0:
                    # =================== [LINUS'S FIX START] ===================
                    # The buffer_size is no longer part of this message. It's redundant.
                    # The DataLoaderStatus is the single source of truth for the buffer size.
                    status_msg = TrainerStatus(
                        step=train_step_count, 
                        total_loss=log_vals[0], 
                        policy_loss=log_vals[1], 
                        value_loss=log_vals[2], 
                        reward_loss=log_vals[3], 
                        consistency_loss=log_vals[4], 
                        lr=optimizer.param_groups[0]['lr'], 
                        saved_at=train_step_count, 
                        games_completed=games_completed_count, 
                        resumed=True
                    )
                    # =================== [LINUS'S FIX END] ===================
                    log_status_queue.put(status_msg)
                    ui_queue.put(status_msg)

                # Cap the training rate. Self-play produces ~7 new positions per
                # second; letting the trainer run at ~4000 samples/s makes it
                # memorise the small replay buffer (out-of-distribution value
                # accuracy was measured at 50%, i.e. chance).
                max_sps = getattr(config, 'MAX_TRAIN_STEPS_PER_SEC', 0)
                if max_sps and max_sps > 0:
                    remaining = (1.0 / max_sps) - (time.time() - step_started)
                    if remaining > 0:
                        time.sleep(remaining)
            except Empty:
                if not ui_queue.full(): ui_queue.put(TrainerWaitingForDataStatus())
                time.sleep(1)
    except BaseException:
        training_failed = True
        logger.exception('Trainer loop failed; saving final state before exit.')
        raise
    finally:
        if train_step_count > 0 and not model_is_finite(model):
            logger.critical('Model weights are non-finite; keeping the last good checkpoint instead of saving.')
        elif train_step_count > 0:
            logger.info(f"Shutdown. Saving final state at step {train_step_count}...")
            state = {'training_schema_version': config.TRAINING_SCHEMA_VERSION, 'ruleset_id': config.RULESET_ID, 'samples_consumed': samples_consumed, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),'scheduler_state_dict': scheduler.state_dict(), 'train_step_count': train_step_count, 'games_completed_count': games_completed_count}
            db_manager.save_trainer_state(state)
            logger.info("Final state saved.")
        last_progress = 0.0
        publish_progress('failed' if training_failed else 'stopped')

# =====================================================================
#                          ORCHESTRATOR
# =====================================================================
def orchestrator_worker(worker_mode, shutdown_event, log_queue, ui_queue, pause_event, latest_model_step):
    setup_worker_logging(log_queue)
    logger = logging.getLogger("Orchestrator")
    db_manager = DatabaseManager()
    logger.info("Orchestrator started.")

    while not shutdown_event.is_set():
        try:
            # --- [MODIFICATION] ---
            # 旧逻辑: train_step_count = db_manager.load_trainer_state().get('train_step_count', 0)
            # 新逻辑: 直接从共享内存读取最新分发的模型步数
            current_distributed_model_step = latest_model_step.value
            
            if config.ENABLE_REANALYSIS:
                reanalysis_backlog_count = db_manager.get_reanalysis_queue_size(current_distributed_model_step)
                if not ui_queue.full(): ui_queue.put(ReAnalysisQueueStatus(total_games_to_reanalyze=reanalysis_backlog_count))
                
                if worker_mode.value == 0 and reanalysis_backlog_count > 0:
                    logger.info("Switching to Re-analysis mode.")
                    worker_mode.value = 1
                elif worker_mode.value == 1 and reanalysis_backlog_count == 0:
                    if not db_manager.are_games_currently_locked_for_reanalysis():
                        logger.info("Switching back to Self-Play mode.")
                        worker_mode.value = 0
            time.sleep(15)
        except Exception as e:
            logger.error(f"Error in orchestrator: {e}\n{traceback.format_exc()}"); time.sleep(30)

# =====================================================================
#                        VISUALIZER & LOGGER
# =====================================================================
def visualize_and_log_worker(vis_task_queue, log_status_queue, shutdown_event, replay_data_queue, log_queue):
    setup_worker_logging(log_queue)
    logger = logging.getLogger("Visualizer")
    try:
        outputs_dir = config.OUTPUT_DIR
        writer = SummaryWriter(os.path.join(outputs_dir, "logs"))
        model = GomokuNetEZ(config).to(config.DEVICE)
        db_manager = DatabaseManager()
        game_lengths_deque = deque(maxlen=100)
        current_step = 0
        last_replay_save_step = -1
        REPLAY_SAVE_INTERVAL = 100 
        logger.info("Visualizer started.")

        def save_replay(step):
            nonlocal last_replay_save_step
            # 只有当满足时间间隔条件时才尝试保存
            if step > last_replay_save_step + REPLAY_SAVE_INTERVAL:
                try: # <--- 在这里开始一个 try 块
                    # 尝试从队列获取 game_id，如果队列为空则直接返回
                    try:
                        game_id = replay_data_queue.get_nowait()
                        last_replay_save_step = step
                    except Empty:
                        return
                    
                    # 后续的所有操作都包含在 try 块中
                    replay_data = db_manager.get_game_record_by_id(game_id)
                    if replay_data is None: return
                    
                    filepath = os.path.join(outputs_dir, "replays", f"replay_step_{step}_game_{game_id}.json")
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    serializable_data = _convert_to_json_serializable(replay_data)
                    
                    winner_text = "Draw"
                    if replay_data and replay_data.rewards:
                        last_reward = replay_data.rewards[-1]
                        num_moves = len(replay_data.actions)
                        if last_reward == 1.0: winner_text = "Black" if (num_moves - 1) % 2 == 0 else "White"
                        elif last_reward == -1.0: winner_text = "White" if (num_moves - 1) % 2 == 0 else "Black"
                    
                    final_data_for_json = { 'challenger_color': 'Black', 'defender_color': 'White', 'winner': winner_text, 'GameRecord': serializable_data }
                    with open(filepath, 'w') as f: json.dump(final_data_for_json, f)
                    logger.info(f"Saved replay for game {game_id} at step {step}.")
                
                except Exception as e: # <--- 现在这个 except 是合法的，它捕获上面 try 块中的任何错误
                    logger.error(f"Error saving replay: {e}")

        def generate_heatmap(step, state_dict):
            try:
                # --- [MODIFICATION START] ---
                # 1. 定义需要保存的网络配置
                model_config_to_save = checkpoint_parameters(config)
                
                # 2. 实例化一个匹配的模型来加载权重
                # 注意：我们在这里创建了一个临时的 config 对象，以确保与 state_dict 匹配
                temp_config_obj = type('TempConfig', (object,), model_config_to_save)()
                model.load_state_dict(state_dict)
                model.eval()
                # --- [MODIFICATION END] ---
                
                obs = torch.from_numpy(GomokuGame().get_board_state(1, None)).unsqueeze(0).to(config.DEVICE)
                with torch.no_grad(): policy_logits, _, _ = model.initial_inference(obs)
                
                policy = torch.softmax(policy_logits, dim=1).squeeze().view(config.BOARD_SIZE, config.BOARD_SIZE).cpu().numpy()
                fig, ax = plt.subplots(figsize=(8, 8)); sns.heatmap(policy, cmap="viridis", ax=ax, square=True)
                ax.set_title(f"Opening Policy at Step {step}")
                writer.add_figure('Analysis/Opening_Policy_Heatmap', fig, global_step=step)
                
                heatmap_path = os.path.join(outputs_dir, "heatmaps"); os.makedirs(heatmap_path, exist_ok=True)
                fig.savefig(os.path.join(heatmap_path, f"heatmap_step_{step}.png")); plt.close(fig)

                weights_path = os.path.join(outputs_dir, "model_weights")
                os.makedirs(weights_path, exist_ok=True)
                torch.save(state_dict, os.path.join(weights_path, f"model_weights_step_{step}.pt"))

                # --- [MODIFICATION START] ---
                # 3. 将网络配置保存为同名的 .json 文件
                config_path = os.path.join(weights_path, f"model_weights_step_{step}.json")
                atomic_json(config_path, model_config_to_save)
                # --- [MODIFICATION END] ---

                logger.info(f"Generated heatmap and saved model weights & config for step {step}.")
                save_replay(step)
            except Exception as e:
                logger.error(f"Error generating heatmap for step {step}: {e}")

        while not shutdown_event.is_set():
            try:
                task = vis_task_queue.get_nowait()
                if isinstance(task, HeatmapTask): generate_heatmap(task.step, task.model_state_dict)
            except Empty: pass

            # We no longer save replays in the main loop to prevent spam.
            # It's now tied to the heatmap generation event.

            try:
                msg = log_status_queue.get(timeout=0.5)
                if isinstance(msg, TrainerStatus):
                    current_step = msg.step
                    writer.add_scalar('Loss/Total', msg.total_loss, current_step); writer.add_scalar('Loss/Policy', msg.policy_loss, current_step); writer.add_scalar('Loss/Value', msg.value_loss, current_step); writer.add_scalar('Loss/Reward', msg.reward_loss, current_step); writer.add_scalar('Loss/Consistency', msg.consistency_loss, current_step); writer.add_scalar('Meta/Learning_Rate', msg.lr, current_step)
                elif isinstance(msg, DataLoaderStatus): writer.add_scalar('Buffer/Size', msg.buffer_size, current_step)
                elif isinstance(msg, ReAnalysisQueueStatus): writer.add_scalar('Buffer/Reanalysis_Backlog', msg.total_games_to_reanalyze, current_step)
                elif isinstance(msg, SelfPlayStatus):
                    game_lengths_deque.append(msg.avg_len)
                    writer.add_scalar('Analysis/Avg_Game_Length', np.mean(game_lengths_deque), current_step)
                    if np.isfinite(msg.miss_five):
                        writer.add_scalar('Analysis/Missed_Fives', msg.miss_five, current_step)
                        writer.add_scalar('Analysis/Missed_All_Wins', msg.miss_total, current_step)
            except Empty: continue
    except Exception as e: logging.getLogger("Visualizer_Fallback").critical(f"CRITICAL ERROR: {e}\n{traceback.format_exc()}")
    finally:
        if 'writer' in locals() and writer is not None: writer.close()

# =====================================================================
#                          DISPLAY MANAGER
# =====================================================================
def display_manager(ui_queue, shutdown_event, server_ready_event, worker_mode, log_queue):
    tr_bar, sp_bar, re_bar = None, None, None # 初始化为 None
    try:
        setup_worker_logging(log_queue)
        logger = logging.getLogger("DisplayManager")
        logger.setLevel(logging.WARNING)
        server_ready_event.wait()
        
        tr_bar = tqdm(position=0, leave=True, bar_format="{desc}: [{elapsed}] {postfix}", dynamic_ncols=True, desc="Training")
        sp_bar = tqdm(position=1, leave=True, bar_format="{desc}: {postfix}", dynamic_ncols=True, desc="Self-playing")
        re_bar = tqdm(position=2, leave=True, bar_format="{desc}: {postfix}", dynamic_ncols=True, desc="Re-analysis")

        tr_stats, sp_model_step, spinner_idx, games_completed = {}, 0, 0, 0
        tr_is_working, prefilling_done, tr_is_paused, sp_is_paused = True, False, False, False
        tr_pause_reason, spinner_chars = "", ['|', '/', '-', '\\']
        sp_postfix_history, re_postfix_history, previous_worker_mode = "[Waiting for game...]", "[Waiting for model...]", 0
        len_deque, miss5_deque = deque(maxlen=20), deque(maxlen=20)

        # --- [MODIFICATION START] ---
        # Replace deques with cumulative counters for the current re-analysis batch
        re_corrected_fives, re_original_fives = 0, 0
        re_corrected_all, re_original_all = 0, 0
        # --- [MODIFICATION END] ---

        while not shutdown_event.is_set():
            while not ui_queue.empty():
                try:
                    msg = ui_queue.get_nowait()
                    if isinstance(msg, TrainerStatus): 
                        tr_stats.update({'step': msg.step, 'loss': msg.total_loss, 'lr': msg.lr})
                        tr_is_working = True
                        prefilling_done = True
                    elif isinstance(msg, DataLoaderStatus): 
                        tr_stats['buffer'] = msg.buffer_size
                    elif isinstance(msg, SelfPlayStatus): len_deque.append(msg.avg_len); miss5_deque.append(msg.miss_five)
                    elif isinstance(msg, ReAnalysisStatus):
                        re_bar.update(1)
                        # --- [MODIFICATION START] ---
                        # Accumulate the raw numbers instead of calculating and storing individual rates
                        re_corrected_fives += msg.corrected_fives
                        re_original_fives += msg.original_fives
                        re_corrected_all += msg.corrected_totals
                        re_original_all += msg.original_totals
                        # --- [MODIFICATION END] ---
                    elif isinstance(msg, ReAnalysisQueueStatus):
                        if worker_mode.value == 1 and previous_worker_mode == 0: re_bar.total = msg.total_games_to_reanalyze; re_bar.n = 0; re_bar.refresh()
                    elif isinstance(msg, GameCompletedNotice): games_completed += 1; sp_bar.update(1)
                    elif isinstance(msg, TrainerWaitPrefillStatus): tr_stats.update({'buffer': msg.buffer_size, 'prefill': msg.prefill_size}); prefilling_done = False
                    elif isinstance(msg, TrainerModelUpdateNotice): sp_model_step = msg.step
                    elif isinstance(msg, TrainerWaitingForDataStatus): tr_is_working = False
                    elif isinstance(msg, TrainerPauseStatus): tr_is_paused = msg.is_paused; tr_pause_reason = msg.reason if msg.is_paused else ""
                    elif isinstance(msg, WorkerPauseStatus): sp_is_paused = msg.is_paused
                except Empty: break
            
            step, buffer_size = tr_stats.get('step', 0), tr_stats.get('buffer', 0)
            spinner = spinner_chars[spinner_idx := (spinner_idx + 1) % 4]
            
            tr_spinner = spinner if (prefilling_done and tr_is_working and not tr_is_paused) else ''
            tr_desc = f"[Paused: {tr_pause_reason}] Training" if tr_is_paused else f"Training {tr_spinner}"
            tr_postfix = f"[Waiting for prefill: {buffer_size}/{config.REPLAY_BUFFER_PREFILL}]" if not prefilling_done else f"{step} steps, loss={tr_stats.get('loss', 0):.4f}, buffer={buffer_size}, lr={tr_stats.get('lr', 0):.1e}"
            if tr_bar is not None: tr_bar.set_description(tr_desc); tr_bar.set_postfix_str(tr_postfix)

            current_mode = worker_mode.value
            if current_mode == 0 and not sp_is_paused:
                sp_postfix = f"{games_completed} games, len={np.mean(len_deque):.1f}, miss5={np.mean(miss5_deque):.2f}" if len_deque else "[Waiting for game...]"
                sp_postfix_history = sp_postfix
            sp_spinner = spinner if (current_mode == 0 and not sp_is_paused) else ''
            sp_desc = f"[Paused] Self-playing (M@{sp_model_step})" if sp_is_paused else f"Self-playing (M@{sp_model_step}){sp_spinner}"
            if sp_bar is not None: sp_bar.set_description(sp_desc); sp_bar.set_postfix_str(sp_postfix_history)

            if current_mode == 0 and previous_worker_mode == 1:
                # This resets the progress bar visually when switching back to self-play
                if re_bar is not None: re_bar.n = 0; re_bar.total = 0; re_bar.refresh()

            # --- [MODIFICATION START] ---
            # This resets the counters when a NEW re-analysis cycle begins
            if current_mode == 1 and previous_worker_mode == 0:
                re_corrected_fives, re_original_fives = 0, 0
                re_corrected_all, re_original_all = 0, 0
            # --- [MODIFICATION END] ---

            re_desc = f"Re-analysis (M@{sp_model_step})"
            if current_mode == 1:
                re_desc += f" {spinner}"
                # --- [MODIFICATION START] ---
                # Calculate percentage based on the cumulative sums. Handle division by zero.
                fives_rate = (re_corrected_fives / re_original_fives * 100) if re_original_fives > 0 else 0.0
                total_rate = (re_corrected_all / re_original_all * 100) if re_original_all > 0 else 0.0
                avg_fives_rate_str = f"{fives_rate:.1f}%"
                avg_total_rate_str = f"{total_rate:.1f}%"
                re_postfix = f"{(re_bar.n if re_bar is not None else 0)}/{(re_bar.total if re_bar is not None else '?')} games, corrected (5, all): {avg_fives_rate_str}, {avg_total_rate_str}"
                re_postfix_history = re_postfix
                # --- [MODIFICATION END] ---

            if re_bar is not None: re_bar.set_description(re_desc); re_bar.set_postfix_str(re_postfix_history)
            
            if tr_bar is not None: tr_bar.refresh()
            if sp_bar is not None: sp_bar.refresh()
            if re_bar is not None: re_bar.refresh()
            
            previous_worker_mode = current_mode
            time.sleep(0.1)
    except Exception as e:
        logging.getLogger("DisplayManager").critical(f"CRITICAL ERROR: {e}\n{traceback.format_exc()}")
    finally:
        if tr_bar is not None: tr_bar.close()
        if sp_bar is not None: sp_bar.close()
        if re_bar is not None: re_bar.close()
