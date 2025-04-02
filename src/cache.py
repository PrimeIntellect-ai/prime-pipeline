from functools import reduce
from typing import List, Tuple

import torch
from torch import Tensor

from .logger import get_logger


class StaticKVCache:
    def __init__(
        self,
        num_layers: int,
        num_micro_batches: int,
        micro_batch_size: int,
        seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        self.logger = get_logger()
        self.num_layers = num_layers
        self.num_micro_batches = num_micro_batches
        self.micro_batch_size = micro_batch_size
        self.dtype = dtype
        self.device = device

        # Create a single cache tensor per layer that contains all micro-batches
        self.key_cache = []
        self.value_cache = []
        for _ in range(num_layers):
            self.key_cache.append([])
            self.value_cache.append([])
            for _ in range(num_micro_batches):
                cache_shape = (micro_batch_size, n_heads, seq_length, head_dim)
                self.key_cache[-1].append(torch.empty(cache_shape, dtype=dtype, device=device))
                self.value_cache[-1].append(torch.empty(cache_shape, dtype=dtype, device=device))

        # Compute cache size
        cache_size = 2 * num_layers * num_micro_batches * reduce(lambda x, y: x * y, cache_shape) * dtype.itemsize
        self.logger.info(f"StaticKVCache: {cache_size / 1e9:.2f}GB")

    def update(self, layer_idx: int, micro_batch_idx: int, input_pos: Tensor, key_states: Tensor, value_states: Tensor):
        """Update the cache with new key and value tensors."""
        # Get the cache slice for this micro-batch
        k_out = self.key_cache[layer_idx][micro_batch_idx]
        v_out = self.value_cache[layer_idx][micro_batch_idx]

        # Update the cache
        k_out.index_copy_(2, input_pos, key_states)
        v_out.index_copy_(2, input_pos, value_states)

        return k_out, v_out


class StaticOffloadedKVCache:
    def __init__(
        self,
        num_layers: int,
        num_micro_batches: int,
        micro_batch_size: int,
        seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
        offload_device: torch.device = torch.device("cpu"),
    ) -> None:
        self.logger = get_logger()

        # Create cache shape
        self.num_layers = num_layers
        self.num_micro_batches = num_micro_batches
        cache_shape = (
            micro_batch_size,
            n_heads,
            seq_length,
            head_dim,
        )
        self.device = device
        self.offload_device = offload_device

        # Create offloaded CPU tensors
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        for layer_idx in range(num_layers):
            # First layer is always on-device.
            device = self.device if layer_idx == 0 else self.offload_device
            # Pin memory for CPU tensors
            is_cpu_device = device == torch.device("cpu")
            self.key_cache.append([])
            self.value_cache.append([])
            for _ in range(num_micro_batches):
                new_key_cache = torch.zeros(cache_shape, dtype=dtype, device=device, pin_memory=is_cpu_device).contiguous()
                new_value_cache = torch.zeros(cache_shape, dtype=dtype, device=device, pin_memory=is_cpu_device).contiguous()
                self.key_cache[-1].append(new_key_cache)
                self.value_cache[-1].append(new_value_cache)

        # Create device tensors
        self.device_key_cache = []
        self.device_value_cache = []
        for _ in range(2):
            self.device_key_cache.append([])
            self.device_value_cache.append([])
            for _ in range(num_micro_batches):
                self.device_key_cache[-1].append(torch.zeros(cache_shape, dtype=dtype, device=self.device))
                self.device_value_cache[-1].append(torch.zeros(cache_shape, dtype=dtype, device=self.device))

        # Calculate total cache size
        self.logger.info(f"cache_shape: {cache_shape}")
        self.logger.info(f"dtype: {dtype}")
        self.logger.info(f"num_layers: {num_layers}")
        self.logger.info(f"num_micro_batches: {num_micro_batches}")
        cpu_cache_size = 2 * num_layers * num_micro_batches * reduce(lambda x, y: x * y, cache_shape) * dtype.itemsize
        gpu_cache_size = 2 * 2 * num_micro_batches * reduce(lambda x, y: x * y, cache_shape) * dtype.itemsize
        self.logger.info(f"StaticOffloadedKVCache: {cpu_cache_size / 1e9:.2f}GB (CPU), {gpu_cache_size / 1e9:.2f}GB (GPU)")

        # Create prefetch stream
        self.prefetch_stream = torch.cuda.Stream()

    def prefetch_layer(self, layer_idx: int, micro_batch_idx: int):
        """Asynchronously prefetch the next layer's cache to GPU."""
        # Don't prefetch what does not exist
        if layer_idx >= self.num_layers or micro_batch_idx >= self.num_micro_batches:
            return

        with torch.cuda.stream(self.prefetch_stream):
            # Copy next layer's cache to GPU
            self.device_key_cache[layer_idx & 1][micro_batch_idx].copy_(self.key_cache[layer_idx][micro_batch_idx], non_blocking=True)
            self.device_value_cache[layer_idx & 1][micro_batch_idx].copy_(self.value_cache[layer_idx][micro_batch_idx], non_blocking=True)

    def update(
        self,
        layer_idx: int,
        micro_batch_idx: int,
        input_pos: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value tensors.

        This implementation:
        1. Waits for any pending prefetch operations
        2. Updates the current layer's cache
        3. Starts prefetching the next layer
        4. Returns the updated cache for the current layer
        """
        if layer_idx == 0:
            # Always there.
            k_out = self.key_cache[layer_idx][micro_batch_idx]
            v_out = self.value_cache[layer_idx][micro_batch_idx]
        else:
            # Wait for prefetch stream.
            # t0 = perf_counter()
            torch.cuda.default_stream(self.device).wait_stream(self.prefetch_stream)
            # self.logger.debug(f"Prefetch stream wait time: {(perf_counter() - t0) * 1000:.2f}ms")

            k_out = self.device_key_cache[layer_idx & 1][micro_batch_idx]
            v_out = self.device_value_cache[layer_idx & 1][micro_batch_idx]

        # Prefetch next layer
        # t0 = perf_counter()
        self.prefetch_layer(layer_idx + 1, micro_batch_idx)
        # torch.cuda.synchronize()
        # self.logger.debug(f"Prefetch call time: {(perf_counter() - t0) * 1000:.2f}ms")

        # Update device cache
        # t0 = perf_counter()
        k_out.index_copy_(2, input_pos, key_states)
        v_out.index_copy_(2, input_pos, value_states)
        # torch.cuda.synchronize()
        # self.logger.debug(f"Index copy time: {(perf_counter() - t0) * 1000:.2f}ms")

        if layer_idx != 0:
            # t0 = perf_counter()
            input_pos = input_pos.to(self.offload_device)
            key_states = key_states.to(self.offload_device)
            value_states = value_states.to(self.offload_device)
            # torch.cuda.synchronize()
            # self.logger.debug(f"Offload copy time: {(perf_counter() - t0) * 1000:.2f}ms")

            # t0 = perf_counter()
            self.key_cache[layer_idx][micro_batch_idx].index_copy_(2, input_pos, key_states)
            self.value_cache[layer_idx][micro_batch_idx].index_copy_(2, input_pos, value_states)
            # torch.cuda.synchronize()
            # self.logger.debug(f"Offload index copy time: {(perf_counter() - t0) * 1000:.2f}ms")

        return k_out, v_out
