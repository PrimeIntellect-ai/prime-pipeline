import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future
from threading import Thread
from typing import Optional

import torch
import torch.distributed as dist

from .logger import get_logger
from .offload import Offload
from .serializer import Serializer
from .utils import fake_future
from .world import get_world

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
    def isend(self, data: torch.Tensor, tag: int):
        pass

    @abstractmethod
    def irecv(self, tag: int, shape: tuple[int, ...]):
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


class TorchP2PComm(P2PCommBase):
    def __init__(self, device: torch.device, serializer: Optional[Serializer], offload: Optional[Offload], **kwargs):
        super().__init__(device, serializer, offload)
        self.logger = get_logger()
        if self.world.size <= 1:
            return
        self._setup()

    def _setup(self):
        dist.init_process_group(init_method="tcp://localhost:29500", rank=self.world.rank, world_size=self.world.size, backend="gloo")

    def isend(self, tensor: torch.Tensor, tag: int) -> Future[bool]:
        if self.world.size <= 1:
            return fake_future(True)

        # Get destination rank
        dst = self.world.first_stage_rank if self.world.is_last_stage else self.world.rank + 1
        self.logger.debug(f"Sending tensor {tensor=} to {dst=}")

        send_future = Future()

        def _worker():
            try:
                # Move tensor to CPU synchronously
                cpu_tensor = self.offload.to_cpu(tensor)

                # Send tensor synchronously
                req = dist.isend(cpu_tensor, dst=dst, tag=tag)

                # Create a completion watcher thread
                def _completion_watcher():
                    req.wait()
                    send_future.set_result(True)

                watcher = Thread(target=_completion_watcher, daemon=True)
                watcher.start()
            except Exception as e:
                send_future.set_exception(e)

        worker = Thread(target=_worker, daemon=True)
        worker.start()

        return send_future

    def irecv(
        self,
        tag: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> Future[torch.Tensor]:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        if self.world.size <= 1:
            return fake_future(torch.zeros(()))

        # Get source rank
        src = self.world.last_stage_rank if self.world.is_first_stage else self.world.rank - 1
        self.logger.debug(f"Receiving tensor from {src=}")

        recv_future = Future()

        def _worker():
            try:
                # Create buffer (TODO: Gloo complains if initialized as torch.empty (performance hit?)
                cpu_tensor = torch.zeros(shape, dtype=dtype)

                # Receive tensor synchronously
                req = dist.irecv(cpu_tensor, src=src, tag=tag)

                # Create a completion watcher thread
                def _completion_watcher():
                    self.logger.debug(f"Blocked for recv {tag=}")
                    req.wait()
                    self.logger.debug(f"Recv {tag=} done")
                    gpu_tensor = self.offload.to_gpu(cpu_tensor)
                    recv_future.set_result(gpu_tensor)

                watcher = Thread(target=_completion_watcher, daemon=True)
                watcher.start()
            except Exception as e:
                recv_future.set_exception(e)

        worker = Thread(target=_worker, daemon=True)
        worker.start()

        return recv_future

    def destroy(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class IrohP2PComm(P2PCommBase):
    def __init__(self, device: torch.device, serializer: Serializer, offload: Offload, num_micro_batches: int, latency: int = 0, **kwargs):
        super().__init__(device, serializer, offload, **kwargs)
        self.logger = get_logger()
        self.num_micro_batches, self.latency = num_micro_batches, latency
        self.node = None
        if self.world.size <= 1:
            return
        self._setup()

    def _setup(self):
        from prime_iroh import Node

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
        self.node.connect(peer_id, num_retries=1, backoff_ms=1000)

        # Wait for connection to be established
        while not self.node.is_ready():
            time.sleep(1 / 100)
        self.logger.info("Connected!")

    def isend(self, tensor: torch.Tensor, tag: int, **kwargs):
        if self.world.size <= 1:
            return fake_future(True)

        cpu_tensor = self.offload.to_cpu(tensor)
        serialized_tensor = self.serializer.serialize(cpu_tensor)
        send_req = self.node.isend(serialized_tensor, tag=tag, latency=self.latency)

        class SendTask:
            def __init__(self, send_req):
                self.send_req = send_req

            def result(self):
                self.send_req.wait()
                return True

        return SendTask(send_req)

    def irecv(self, tag: int, **kwargs):
        if self.world.size <= 1:
            return fake_future(torch.zeros(()))

        recv_work = self.node.irecv(tag)

        class RecvTask:
            def __init__(self, serializer, offload, recv_work):
                self.serializer = serializer
                self.offload = offload
                self.recv_work = recv_work

            def result(self):
                serialized_tensor = self.recv_work.wait()
                cpu_tensor = self.serializer.deserialize(serialized_tensor)
                return self.offload.to_gpu(cpu_tensor)

        return RecvTask(self.serializer, self.offload, recv_work)

    def destroy(self):
        if self.node is not None:
            self.node.close()
