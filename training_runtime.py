"""Run isolation, process lifetime and data-budget helpers."""
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
import hashlib
import traceback


def validate_run_rules(output_dir, ruleset_id):
    """Reject a rules change before overwriting the old run's configuration."""
    directory = Path(output_dir)
    previous_path = directory / 'run_config.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding='utf-8-sig'))['config']
        previous_id = previous.get('RULESET_ID') or (
            'renju_legacy_unverified' if previous.get('ENABLE_RENJU') else 'freestyle_v1')
        if previous_id != ruleset_id:
            raise RuntimeError('Rules changed: use a separate output directory and weights-only transfer.')
    manifest_path = directory / 'warmstart.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['target_ruleset_id'] != ruleset_id:
            raise RuntimeError('Warm-start rules do not match this experiment.')


def load_warm_start(model, output_dir, ruleset_id):
    """Import weights only; optimizer, scheduler, replay and counters stay fresh."""
    import torch
    directory = Path(output_dir).resolve()
    path = directory / 'warmstart.json'
    if not path.exists():
        return False
    validate_run_rules(directory, ruleset_id)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    checkpoint = (directory / manifest['seed_checkpoint']).resolve()
    if not checkpoint.is_relative_to(directory):
        raise RuntimeError('Warm-start seed must be inside its experiment.')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest['seed_sha256']:
        raise RuntimeError('Warm-start seed checksum mismatch.')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise RuntimeError('Warm-start weights contain non-finite values.')
    model.load_state_dict(state, strict=True)
    return True


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    # Windows readers/scanners can briefly hold a handle without FILE_SHARE_DELETE.
    # A transient sharing violation must not terminate an otherwise healthy worker.
    try:
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.01 * 2**attempt, 0.32))
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


class RunLock:
    """OS-owned lock: automatically released if the main process dies."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.file = self.path.open('a+b')
        self.file.seek(0)
        if self.path.stat().st_size == 0:
            self.file.write(b' ')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError('Another training process holds the project lock.') from exc
        return self

    def __exit__(self, *args):
        self.file.close()


def worker_entry(target, args):
    """Stop a child even when Windows terminates the main process abruptly."""
    parent = mp.parent_process()

    def watch_parent():
        while True:
            time.sleep(1)
            if parent is not None and not parent.is_alive():
                os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    output = os.environ.get('GOMOKU_OUTPUT_DIR')
    status_path = Path(output) / 'worker_status' / f'{os.getpid()}.json' if output else None

    def record(phase, **extra):
        if status_path:
            atomic_json(status_path, {'pid': os.getpid(), 'name': mp.current_process().name,
                                     'phase': phase, 'updated_at': time.time(), **extra})

    try:
        record('running')
        target(*args)
        record('completed')
    except BaseException as exc:
        # multiprocessing prints exceptions only AFTER joining queue feeders.
        # Preserve the traceback first, even if a feeder is blocked on a full pipe.
        detail = traceback.format_exc()
        traceback.print_exc()
        try:
            record('failed', error=repr(exc), traceback=detail)
        except OSError:
            traceback.print_exc()
        raise
    finally:
        # The target's own finally has already saved its durable state. Consumers
        # may have exited, so do not keep the OS process alive flushing telemetry
        # or in-flight games to queues that nobody can drain anymore.
        from multiprocessing.queues import Queue
        for arg in args:
            if isinstance(arg, Queue):
                arg.cancel_join_thread()


def training_budget_available(total_positions, samples_consumed, batch_size, ratio):
    return samples_consumed + batch_size <= total_positions * ratio


def resumed_sample_count(saved_step, saved_samples, last_status):
    """Weight rollback after a crash must not refund already-used samples."""
    if not last_status or 'step' not in last_status or 'samples_consumed' not in last_status:
        return saved_samples
    observed_step, observed_samples = last_status['step'], last_status['samples_consumed']
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0
           for v in (observed_step, observed_samples)):
        raise ValueError('Invalid persisted training sample high-water mark.')
    if observed_step < saved_step:
        return saved_samples
    return max(saved_samples, observed_samples)


def assert_evaluation_allowed(output_dir, review_only=False):
    """A controlled review may evaluate saved models while training stays stopped."""
    directory = Path(output_dir)
    stop = directory / 'STOP'
    if not stop.exists():
        if review_only:
            raise RuntimeError('Review evaluation requires the stage-review STOP to remain present.')
        return
    if review_only:
        review_path = directory / 'review_pause.json'
        pause_path = directory / 'user_pause.json'
        review = json.loads(review_path.read_text(encoding='utf-8-sig')) if review_path.exists() else {}
        pause = json.loads(pause_path.read_text(encoding='utf-8-sig')) if pause_path.exists() else {}
        if (review.get('active') and review.get('status') == 'stage_review_paused'
                and review.get('allow_evaluation') is True and review.get('stop_token')
                and pause.get('status') not in ('user_paused', 'prepared_user_paused', 'pause_requested')
                and stop.read_text(encoding='utf-8-sig').strip() == 'STAGE_REVIEW_PAUSE:' + review['stop_token']):
            return
    raise RuntimeError('Experiment STOP exists; evaluation paused. A user pause always takes priority.')
