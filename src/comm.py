import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from threading import Thread
from typing import Optional, Union

import torch
import torch.distributed as dist
from iroh_py import Node, RecvWork, SendWork

from .logger import get_logger
from .offload import Offload
from .serializer import Serializer
from .world import get_world


class WorkBase(ABC):
    def __init__(
        self,
        tensor: Optional[torch.Tensor],
        work: Union[SendWork, RecvWork, dist.Work],
        device: torch.device,
        serializer: Optional[Serializer] = None,
        offload: Optional[Offload] = None,
    ):
        self.tensor = tensor
        self.work = work
        self.device = device
        self.serializer = serializer
        self.offload = offload

    def deserialize(self, tensor: Union[bytes, torch.Tensor]) -> torch.Tensor:
        """Deserializes a tensor if a serializer is provided"""
        if self.serializer:
            return self.serializer.deserialize(tensor)
        return tensor

    def move_to_device(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Moves a tensor to the correct device"""
        if isinstance(tensor, torch.Tensor):
            t0 = time.perf_counter()
            tensor = tensor.to(self.device)
            get_logger().debug(f"Move to device took {(time.perf_counter() - t0) * 1000:.02f}ms")
            return tensor
        return tensor

    @abstractmethod
    def wait(self) -> Optional[torch.Tensor]:
        """Waits for the work to complete and returns the result"""


class P2PCommBase(ABC):
    def __init__(self, device: torch.device, serializer: Optional[Serializer] = None, offload: Optional[Offload] = None):
        self.world = get_world()
        self.serializer = serializer
        self.offload = offload
        self.device = device

    @abstractmethod
    def _setup(self):
        pass

    @abstractmethod
    def isend(self, data: torch.Tensor, tag: int) -> WorkBase:
        pass

    @abstractmethod
    def irecv(self, tag: int, shape: tuple[int, ...]) -> WorkBase:
        pass

    @abstractmethod
    def destroy(self):
        pass

    def send(self, tensor: torch.Tensor, tag: int) -> None:
        """Sends tensor (hidden states or next token) to the correct next rank"""
        work = self.isend(tensor, tag)
        if work:
            work.result()

    def recv(self, tag: int, prefill: bool = False) -> Optional[torch.Tensor]:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        work = self.irecv(tag, prefill=prefill)
        if work:
            return work.result()


_COMM: Optional[P2PCommBase] = None


def setup_comm(comm_backend: str, **kwargs) -> P2PCommBase:
    global _COMM
    if _COMM is not None:
        return _COMM
    if comm_backend == "torch":
        _COMM = TorchP2PComm(**kwargs)
    elif comm_backend == "iroh":
        _COMM = IrohP2PComm(**kwargs)
    else:
        raise ValueError(f"Invalid comm type: {comm_backend}")
    return _COMM


def destroy_comm():
    global _COMM
    if _COMM is not None:
        _COMM.destroy()
        _COMM = None


def get_comm() -> P2PCommBase:
    global _COMM
    assert _COMM is not None, "Comm not setup"
    return _COMM


class TorchWork(WorkBase):
    def __init__(self, tensor: Optional[torch.Tensor], work: dist.Work, device):
        super().__init__(tensor, work, device)
        self.tensor = tensor
        self.work = work
        self.device = device

    def wait(self) -> Optional[torch.Tensor]:
        self.work.wait()
        return self.move_to_device(self.deserialize(self.tensor))


class TorchP2PComm(P2PCommBase):
    def __init__(
        self,
        fwd_shape: tuple[int, ...],
        bwd_shape: tuple[int, ...],
        fwd_dtype: torch.dtype,
        bwd_dtype: torch.dtype,
        num_prompt_tokens: int,
        device: torch.device,
        serializer: Optional[Serializer],
        offload: Optional[Offload],
    ):
        super().__init__(device, serializer, offload)
        self.logger = get_logger()
        self.fwd_shape, self.bwd_shape = fwd_shape, bwd_shape
        self.fwd_prefill_shape = (fwd_shape[0], num_prompt_tokens, fwd_shape[2])
        self.bwd_prefill_shape = bwd_shape
        self.fwd_dtype, self.bwd_dtype = fwd_dtype, bwd_dtype
        self.num_prompt_tokens = num_prompt_tokens
        if self.world.size <= 1:
            return
        self._setup()
        self.transfer_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _setup(self):
        dist.init_process_group(init_method="tcp://localhost:29500", rank=self.world.rank, world_size=self.world.size, backend="gloo")

    def isend(self, tensor: torch.Tensor, tag: int) -> Future:
        if self.world.size <= 1:
            return None
        dst = self.world.first_stage_rank if self.world.is_last_stage else self.world.rank + 1

        future = Future()

        def _worker():
            try:
                # Move tensor to CPU synchronously
                cpu_tensor = self.offload.to_cpu(tensor)

                # Send tensor synchronously
                req = dist.isend(cpu_tensor, dst=dst, tag=tag)

                # Create a completion watcher thread
                def _completion_watcher():
                    req.wait()
                    future.set_result(True)

                watcher = Thread(target=_completion_watcher, daemon=True)
                watcher.start()
            except Exception as e:
                future.set_exception(e)

        worker = Thread(target=_worker, daemon=True)
        worker.start()

        return future

    def irecv(self, tag: int, prefill: bool = False) -> Future:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        if self.world.size <= 1:
            return None
        shape = (self.bwd_shape, self.bwd_prefill_shape) if self.world.is_first_stage else (self.fwd_shape, self.fwd_prefill_shape)
        dtype = self.bwd_dtype if self.world.is_first_stage else self.fwd_dtype
        src = self.world.last_stage_rank if self.world.is_first_stage else self.world.rank - 1

        future = Future()

        def _worker():
            try:
                # Create buffer (TODO: Gloo complains if initialized as torch.empty (performance hit?)
                cpu_tensor = torch.zeros(shape[int(prefill)], dtype=dtype)

                # Receive tensor synchronously
                req = dist.irecv(cpu_tensor, src=src, tag=tag)

                # Create a completion watcher thread
                def _completion_watcher():
                    req.wait()
                    gpu_tensor = self.offload.to_gpu(cpu_tensor)
                    future.set_result(gpu_tensor)

                watcher = Thread(target=_completion_watcher, daemon=True)
                watcher.start()
            except Exception as e:
                future.set_exception(e)

        worker = Thread(target=_worker, daemon=True)
        worker.start()

        return future

    def destroy(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class IrohWork(WorkBase):
    def __init__(self, work: Union[SendWork, RecvWork], device, serializer: Serializer):
        super().__init__(None, work, device, serializer)

    def wait(self) -> Optional[torch.Tensor]:
        ret = self.work.wait()
        if ret is None:
            return None
        return self.move_to_device(self.deserialize(ret))


class IrohP2PComm(P2PCommBase):
    def __init__(self, device: torch.device, serializer: Serializer, num_micro_batches: int, latency: int = 0):
        super().__init__(device, serializer)
        self.logger = get_logger()
        self.num_micro_batches, self.latency = num_micro_batches, latency
        if self.world.size <= 1:
            return
        self._setup()

    def _setup(self):
        # Create node
        seed = os.environ.get("IROH_SEED", None)
        self.node = Node.with_seed(self.num_micro_batches, seed=int(seed) if seed is not None else None)
        time.sleep(1)
        self.logger.info(f"Listening on {self.node.node_id()}")

        # Connect to remote node
        peer_id = os.environ.get("IROH_PEER_ID")
        if peer_id is None:
            self.logger.info("Didn't find IROH_PEER_ID environment variable, please enter the peer's public key: ")
            peer_id = input().strip()
        self.logger.info(f"Connecting to {peer_id}")
        self.node.connect(peer_id)

        # Wait for connection to be established
        while not self.node.is_ready():
            time.sleep(1 / 100)
        self.logger.info("Connected!")

    def irecv(self, tag: int, **kwargs) -> Optional[IrohWork]:
        if self.world.size <= 1:
            return None
        # self.logger.debug(f"irecv({tag=})")
        return IrohWork(self.node.irecv(tag), self.device, self.serializer)

    def isend(self, tensor: torch.Tensor, tag: int, **kwargs) -> Optional[IrohWork]:
        if self.world.size <= 1:
            return None
        # self.logger.debug(f"isend({tensor=}, {tag=}")
        return IrohWork(self.node.isend(self.serializer.serialize(tensor), tag, self.latency), self.device, self.serializer)

    def destroy(self):
        pass  # TODO: Fix
