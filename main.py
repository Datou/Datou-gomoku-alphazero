# main.py (FINAL, CLEANED VERSION)

# --- Standard Imports ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# The inference server runs thousands of tiny (batch 1-15) forward passes per
# second. PyTorch's default intra-op pool (one thread per core) spends more time
# synchronising than computing here and was measured burning ~6 cores, starving
# the 15 self-play workers. One thread per process is faster for this workload.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import torch
torch.set_num_threads(1)
import signal
import logging
import torch.multiprocessing as mp
from queue import Empty
import time  # <--- [LINUS'S FIX] ADD THIS. IT IS NOT OPTIONAL.
import hashlib
import json
from pathlib import Path
from training_runtime import RunLock, atomic_json, worker_entry, validate_run_rules
from shared_inference import SharedInferenceTransport

# --- Project-specific Imports ---
from config import config
from network import GomokuNetEZ
from db_manager import DatabaseManager
from logger_config import setup_worker_logging, logger_process

# Import refactored workers and loss function
from workers import universal_worker, inference_server_worker, data_loader_worker, training_worker, orchestrator_worker, visualize_and_log_worker, display_manager
from loss import calculate_loss

# Import refactored IPC messages and helper classes/functions
from ipc_messages import InitialModelRequest, ModelWeightsUpdate, HeatmapTask
from utils import _convert_to_json_serializable # Note: Only used in a function that is now in workers.py, but kept for context if needed elsewhere.

def log_and_display_config(logger):
    """Logs the key configuration parameters at startup."""
    header = "="*30
    config_details = f"\n{header} Key Configuration {header}\n"
    config_details += f"[System & Environment]\n  - Device: {config.DEVICE}\n  - Universal Workers: {config.NUM_WORKERS}\n\n"
    config_details += f"[MCTS Configuration]\n  - Simulations per move: {config.NUM_SIMULATIONS}\n  - c_puct: {config.C_PUCT}\n  - Root noise: eps={config.ROOT_DIRICHLET_EPSILON}, alpha={config.ROOT_DIRICHLET_ALPHA}\n  - Renju rules: {config.ENABLE_RENJU}\n\n"
    config_details += f"[Buffer Configuration]\n  - Prefill Requirement: {config.REPLAY_BUFFER_PREFILL}\n  - Total Buffer Size: {config.TRAIN_BUFFER_SIZE}\n\n"
    config_details += f"[Network & Training]\n  - ResNet Blocks: {config.NUM_RES_BLOCKS}\n  - Learning Rate: {config.LEARNING_RATE}\n  - Physical Batch Size: {config.PHYSICAL_BATCH_SIZE}\n  - Unroll Steps: {config.NUM_UNROLL_STEPS}\n"
    config_details += header + "========================" + header
    logger.info(config_details)
    print(config_details)

# =====================================================================
#                      MAIN EXECUTION BLOCK
# =====================================================================
def main():
    validate_run_rules(config.OUTPUT_DIR, config.RULESET_ID)
    if (Path(config.OUTPUT_DIR) / 'STOP').exists():
        raise RuntimeError('STOP exists. An explicit user resume is required before training.')
    run_started_at = time.time()
    max_train_seconds = float(os.environ.get('GOMOKU_TRAIN_SECONDS', '0') or 0)
    if max_train_seconds < 0:
        raise ValueError('GOMOKU_TRAIN_SECONDS must be non-negative.')
    if config.ENABLE_REANALYSIS:
        raise ValueError('This training entry requires reanalysis disabled.')
    if config.ENABLE_FORCED_MOVE_PRUNING:
        raise ValueError('Tactical move overrides are disabled for this experiment.')
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # 1. --- Centralized Logging Setup ---
    log_queue = mp.Queue()
    log_file = os.path.join(config.OUTPUT_DIR, 'training.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging_proc = mp.Process(target=worker_entry, args=(logger_process, (log_queue, log_file)), name="Logger")
    logging_proc.start()
    
    setup_worker_logging(log_queue)
    logger = logging.getLogger("Main")
    source_files = ['config.py', 'workers.py', 'loss.py', 'mcts.py', 'network.py', 'game.py', 'renju_rules.py', 'main.py', 'training_runtime.py', 'db_manager.py', 'inference_backend.py', 'shared_inference.py']
    source_root = Path(__file__).resolve().parent
    atomic_json(Path(config.OUTPUT_DIR) / 'run_config.json', {
        'config': vars(config), 'started_at': time.time(), 'pid': os.getpid(),
        'source_sha256': {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in source_files}})
    log_and_display_config(logger)
    
    # 2. --- Inter-Process Communication Objects ---
    # (This part is correct, no changes needed)
    log_status_queue = mp.Queue()
    ui_queue = mp.Queue()
    trainer_event_queue = mp.Queue()
    resume_info_queue = mp.Queue(maxsize=1)
    data_queue = mp.Queue(maxsize=50000)
    vis_task_queue = mp.Queue()
    inference_request_queue = mp.Queue()
    shared_transport = (SharedInferenceTransport(
        mp, config.NUM_WORKERS, config.BOARD_SIZE)
        if config.SHARED_INFERENCE else None)
    model_update_queue = mp.Queue(maxsize=1)
    replay_data_queue = mp.Queue(maxsize=20)
    batch_queue = mp.Queue(maxsize=config.PREPARED_BATCH_QUEUE_SIZE)
    priority_update_queue = mp.Queue()
    initial_model_requests_queue = mp.Queue(maxsize=1)
    inference_result_queues = {i: mp.Queue() for i in range(config.NUM_WORKERS)} if shared_transport is None else {}
    buffer_status_queue = mp.Queue()
    shutdown_event = mp.Event()
    server_ready_event = mp.Event()
    pause_event = mp.Event() 
    # These aligned 32-bit control words each have exactly ONE writer:
    # Orchestrator for mode, Trainer for model version. Readers do not need a
    # shared recursive Windows mutex (which failed with WinError 5 on startup).
    # Model tensors continue travelling through the synchronized update queue.
    WORKER_MODE = mp.RawValue('i', 0)
    LATEST_MODEL_STEP = mp.RawValue('i', 0)
    
    # 3. --- Initial State Check ---
    db_manager = DatabaseManager()
    is_first_ever_run = not db_manager.load_trainer_state()
    if is_first_ever_run:
        logger.info("First ever run detected. Database appears empty.")
    else:
        logger.info("Resuming from existing state found in database.")
        
    # 4. --- Process Definitions ---
    processes = []
    process_definitions = {
        "Orchestrator": (orchestrator_worker, (WORKER_MODE, shutdown_event, log_queue, ui_queue, pause_event, LATEST_MODEL_STEP)),
        "InferenceServer": (inference_server_worker, (inference_request_queue, inference_result_queues, model_update_queue, initial_model_requests_queue, shutdown_event, server_ready_event, log_queue, shared_transport)),
        # ======== [LINUS'S FIX] ADD replay_data_queue HERE ========
        "DataLoader": (data_loader_worker, (data_queue, batch_queue, priority_update_queue, log_status_queue, ui_queue, shutdown_event, resume_info_queue, buffer_status_queue, replay_data_queue, log_queue)),
        "Visualizer": (visualize_and_log_worker, (vis_task_queue, log_status_queue, shutdown_event, replay_data_queue, log_queue)),
        "DisplayManager": (display_manager, (ui_queue, shutdown_event, server_ready_event, WORKER_MODE, log_queue)),
        "Trainer": (training_worker, (batch_queue, priority_update_queue, log_status_queue, ui_queue, vis_task_queue, shutdown_event, model_update_queue, initial_model_requests_queue, resume_info_queue, trainer_event_queue, WORKER_MODE, buffer_status_queue, LATEST_MODEL_STEP, log_queue, pause_event)),
    }
    for i in range(config.NUM_WORKERS):
        client = shared_transport.client(i, inference_request_queue) if shared_transport is not None else None
        process_definitions[f"Worker-{i}"] = (universal_worker, (i, WORKER_MODE, data_queue, log_status_queue, ui_queue, shutdown_event, client or inference_request_queue, client or inference_result_queues[i], replay_data_queue, trainer_event_queue, LATEST_MODEL_STEP, log_queue, pause_event))

    if not os.isatty(1):
        process_definitions.pop('DisplayManager')

    for name, (target, args) in process_definitions.items():
        processes.append(mp.Process(target=worker_entry, args=(target, args), name=name))

    # 5. --- Process Execution and Shutdown ---
    logger.info(f"Starting {len(processes)} processes...")
    for p in processes:
        p.start()
        
    def sigint_handler(signum, frame):
        if not shutdown_event.is_set():
            logger.info("Ctrl+C detected. Initiating graceful shutdown...")
            shutdown_event.set()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    
    # =================== [LINUS'S FIX START] ===================
    # This is the correct way to manage the main process.
    # It waits for the shutdown signal, not for a single child process to die.
    try:
        while not shutdown_event.is_set():
            failed = [p for p in processes + [logging_proc] if not p.is_alive()]
            # A target can fail while its OS process is still unwinding queues.
            for p in processes + [logging_proc]:
                lifecycle = Path(config.OUTPUT_DIR) / 'worker_status' / f'{p.pid}.json'
                if lifecycle.exists():
                    state = json.loads(lifecycle.read_text(encoding='utf-8'))
                    if state['updated_at'] >= run_started_at and state['phase'] != 'running' and p not in failed:
                        failed.append(p)
            if failed:
                logger.critical('Child exited unexpectedly: %s', [(p.name, p.exitcode) for p in failed])
                shutdown_event.set()
            if (Path(config.OUTPUT_DIR) / 'STOP').exists():
                logger.info('STOP file detected. Requesting graceful shutdown.')
                shutdown_event.set()
            if max_train_seconds > 0 and time.time() - run_started_at >= max_train_seconds:
                logger.info('GOMOKU_TRAIN_SECONDS elapsed. Requesting graceful shutdown.')
                shutdown_event.set()
            atomic_json(Path(config.OUTPUT_DIR) / 'run_status.json', {
                'pid': os.getpid(), 'updated_at': time.time(),
                'phase': 'stopping' if shutdown_event.is_set() else 'running',
                'children': [{'name': p.name, 'pid': p.pid, 'alive': p.is_alive(), 'exitcode': p.exitcode} for p in processes + [logging_proc]]})
            shutdown_event.wait(1)
    finally:
        logger.info("Shutdown signal received in main process. Finalizing shutdown of all workers...")
        
        # Give processes a moment to finish cleanly
        deadline = time.monotonic() + 10
        for p in processes:
            if p.is_alive():
                p.join(timeout=max(0, deadline - time.monotonic()))
                
        # Now, terminate any stubborn processes
        for p in processes:
            if p.is_alive():
                logger.warning(f"Process {p.name} (PID: {p.pid}) did not exit gracefully. Terminating.")
                p.terminate()
                p.join(timeout=3)

        # Cleanly shut down the logger
        log_queue.put(None)
        logging_proc.join(timeout=5)
        if logging_proc.is_alive():
            logging_proc.terminate()
            logging_proc.join(timeout=3)
        atomic_json(Path(config.OUTPUT_DIR) / 'run_status.json', {
            'pid': os.getpid(), 'updated_at': time.time(), 'phase': 'stopped',
            'children': [{'name': p.name, 'pid': p.pid, 'alive': p.is_alive(), 'exitcode': p.exitcode} for p in processes + [logging_proc]]})
        
        logger.info("All workers shut down.")
    # =================== [LINUS'S FIX END] ===================


if __name__ == '__main__':
    with RunLock(Path(__file__).resolve().parent / '.training.lock'):
        main()
