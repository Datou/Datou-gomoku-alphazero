"""Batched inference with optional mixed precision and static CUDA graphs."""
import torch


class InferenceBackend:
    def __init__(self, model, precision='fp32', channels_last=False,
                 cuda_graphs=False, max_batch_size=32):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.precision = precision
        self.channels_last = channels_last and self.device.type == 'cuda'
        self.cuda_graphs = cuda_graphs and self.device.type == 'cuda'
        self.max_batch_size = max_batch_size
        self.graphs = {}
        self.io_graphs = {}
        if self.channels_last:
            model.to(memory_format=torch.channels_last)

    def _forward(self, observations):
        dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
        with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=dtype,
                enabled=self.device.type == 'cuda' and self.precision != 'fp32',
                cache_enabled=False):
            policy, value, _ = self.model.initial_inference(observations)
            return policy.float(), value.float()

    def _capture(self, batch_size):
        shape = (batch_size, 3, self.model.board_size, self.model.board_size)
        memory_format = torch.channels_last if self.channels_last else torch.contiguous_format
        observations = torch.zeros(shape, device=self.device).contiguous(memory_format=memory_format)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward(observations)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self._forward(observations)
        self.graphs[batch_size] = (graph, observations, outputs)

    def predict(self, observations):
        size = observations.shape[0]
        if size < 1 or size > self.max_batch_size:
            raise ValueError(f'Invalid inference batch size: {size}')
        if not self.cuda_graphs:
            if self.channels_last:
                observations = observations.contiguous(memory_format=torch.channels_last)
            return self._forward(observations.to(self.device, non_blocking=True))
        bucket = min(1 << (size - 1).bit_length(), self.max_batch_size)
        if bucket not in self.graphs:
            self._capture(bucket)
        graph, static_input, outputs = self.graphs[bucket]
        static_input[:size].copy_(observations)
        graph.replay()
        return tuple(tensor[:size] for tensor in outputs)

    def predict_cpu(self, observations):
        """Capture pinned H2D/D2H copies too, avoiding extra driver submissions."""
        size = observations.shape[0]
        if size < 1 or size > self.max_batch_size:
            raise ValueError(f'Invalid inference batch size: {size}')
        if not self.cuda_graphs:
            policy, value = self.predict(observations)
            return policy.cpu(), value.cpu()
        bucket = min(1 << (size - 1).bit_length(), self.max_batch_size)
        if bucket not in self.io_graphs:
            shape = (bucket, 3, self.model.board_size, self.model.board_size)
            host_input = torch.zeros(shape, pin_memory=True)
            device_input = host_input.to(self.device)
            if self.channels_last:
                device_input = device_input.contiguous(memory_format=torch.channels_last)
            host_policy = torch.empty(bucket, self.model.action_space_size, pin_memory=True)
            host_value = torch.empty(bucket, 1, pin_memory=True)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward(device_input)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                device_input.copy_(host_input, non_blocking=True)
                outputs = self._forward(device_input)
                host_policy.copy_(outputs[0], non_blocking=True)
                host_value.copy_(outputs[1], non_blocking=True)
            self.io_graphs[bucket] = (graph, host_input, host_policy, host_value, device_input, outputs)
        graph, host_input, host_policy, host_value, _, _ = self.io_graphs[bucket]
        host_input[:size].copy_(observations)
        graph.replay()
        torch.cuda.current_stream(self.device).synchronize()
        return host_policy[:size], host_value[:size]
