"""One outstanding inference per actor; payloads stay in shared memory.

Only root observations and policy/value results travel through the transport.
Sequence numbers (integers) are the only data sent over the multiprocessing
queue, which avoids pickling a 3x15x15 float array for every simulation.
"""
from queue import Empty
import time

import numpy as np


class SharedInferenceTransport:
    def __init__(self, context, workers, board_size):
        self.workers = workers
        self.board_size = board_size
        self.action_size = board_size * board_size
        self.observation_storage = context.RawArray('f', workers * 3 * self.action_size)
        self.output_storage = context.RawArray('f', workers * (self.action_size + 1))
        self.completed = context.RawArray('q', workers)
        self.ready = [context.Semaphore(0) for _ in range(workers)]

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ('_observation_view', '_output_view'):
            state.pop(key, None)
        return state

    @property
    def observations(self):
        if not hasattr(self, '_observation_view'):
            self._observation_view = np.frombuffer(self.observation_storage, dtype=np.float32).reshape(
                self.workers, 3, self.board_size, self.board_size)
        return self._observation_view

    @property
    def outputs(self):
        if not hasattr(self, '_output_view'):
            self._output_view = np.frombuffer(self.output_storage, dtype=np.float32).reshape(
                self.workers, self.action_size + 1)
        return self._output_view

    def client(self, worker_id, request_queue):
        return SharedInferenceClient(self, worker_id, request_queue)

    def publish(self, worker_id, sequence, policy, value):
        self.outputs[worker_id, :-1] = policy
        self.outputs[worker_id, -1] = value
        self.completed[worker_id] = sequence
        self.ready[worker_id].release()


class SharedInferenceClient:
    def __init__(self, transport, worker_id, request_queue):
        self.transport = transport
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.sequence = 0

    def put(self, request):
        worker_id, kind, data = request
        if worker_id != self.worker_id:
            raise ValueError('Shared inference client received another actor request.')
        if kind != 'initial':
            raise ValueError(f'Unsupported shared inference request: {kind}')
        while self.transport.ready[worker_id].acquire(block=False):
            pass
        self.transport.observations[worker_id] = data
        self.sequence += 1
        self.request_queue.put((worker_id, 'shared_initial', self.sequence))

    def get(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else None
        signal = self.transport.ready[self.worker_id]
        while True:
            remaining = max(0, deadline - time.monotonic()) if deadline is not None else None
            if not signal.acquire(timeout=remaining):
                raise Empty
            if self.transport.completed[self.worker_id] != self.sequence:
                if deadline is not None and time.monotonic() >= deadline:
                    raise Empty
                continue
            output = self.transport.outputs[self.worker_id]
            policy = output[:-1].copy()
            value = float(output[-1])
            return policy, value, None

    def get_nowait(self):
        return self.get(timeout=0)
