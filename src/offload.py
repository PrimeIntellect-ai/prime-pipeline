import time
from abc import ABC, abstractmethod

import torch

from .logger import get_logger


class Offload(ABC):
    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Moves tensor from GPU to CPU"""
        pass

    @abstractmethod
    def to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Moves tensor from CPU to GPU"""
        pass


class BlockingOffload(Offload):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        cpu_tensor = tensor.cpu()
        get_logger().debug(f"Move to CPU took {(time.perf_counter() - t0) * 1000:.02f}ms")
        return cpu_tensor

    def to_gpu(self, cpu_tensor: torch.Tensor) -> torch.Tensor:
        t0 = time.perf_counter()
        tensor = cpu_tensor.to(self.device)
        get_logger().debug(f"Move to GPU took {(time.perf_counter() - t0) * 1000:.02f}ms")
        return tensor


def get_offload(device: torch.device) -> Offload:
    return BlockingOffload(device)
