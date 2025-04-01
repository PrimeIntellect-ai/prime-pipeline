from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .logger import get_logger


class KVCache(ABC, nn.Module):
    """Abstract base class for key-value cache implementations.

    This class defines the interface that all cache implementations must follow.
    The cache is used to store and retrieve key-value pairs for transformer models.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        cache_size = self._setup(*args, **kwargs)
        self.logger.info(f"KV Cache has size {cache_size / 1024**3:.2f} GB")

    @abstractmethod
    def _setup(
        self,
        num_layers: int,
        num_micro_batches: int,
        max_micro_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Initialize the cache.

        Args:
            num_layers: Number of transformer layers
            num_micro_batches: Number of micro-batches for pipeline parallelism
            n_heads: Number of attention heads
            head_dim: Dimension of each attention head
            dtype: Data type for the cache (default: bfloat16)

        Returns:
            Size of the cache in bytes
        """
        pass

    @abstractmethod
    def update(
        self,
        layer_idx: int,
        micro_batch_idx: int,
        input_pos: torch.Tensor,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the cache with new key and value tensors.

        Args:
            layer_idx: Index of the transformer layer
            micro_batch_idx: Index of the micro-batch
            input_pos: Position indices for the input sequence [seq_length]
            k_val: Key tensor [micro_batch_size, n_heads, seq_length, head_dim]
            v_val: Value tensor [micro_batch_size, n_heads, seq_length, head_dim]

        Returns:
            Tuple of (k_cache, v_cache) for the specified layer and micro-batch
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear the cache."""
        pass


class StaticKVCache(KVCache):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _setup(self, num_layers, num_micro_batches, max_micro_batch_size, max_seq_length, n_heads, head_dim, dtype):
        cache_shape = (
            num_layers,
            num_micro_batches,
            max_micro_batch_size,
            n_heads,
            max_seq_length,
            head_dim,
        )
        self.logger = get_logger()
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

        cache_size = reduce(lambda x, y: x * y, cache_shape) * dtype.itemsize

        return cache_size

    def update(self, layer_idx: int, micro_batch_idx: int, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        k_out = self.k_cache
        v_out = self.v_cache

        k_out[layer_idx, micro_batch_idx, :, :, input_pos, :] = k_val.unsqueeze(0)
        v_out[layer_idx, micro_batch_idx, :, :, input_pos, :] = v_val.unsqueeze(0)

        return k_out[layer_idx, micro_batch_idx], v_out[layer_idx, micro_batch_idx]

    def clear(self):
        """Clear the cache by zeroing out all values."""
        self.k_cache.zero_()
        self.v_cache.zero_()
