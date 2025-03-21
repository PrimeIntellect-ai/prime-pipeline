import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Type, Union

import torch
import torch.distributed as dist

from iroh_py import Node, SendWork, RecvWork
from logger import get_logger
from serializer import Serializer
from world import get_world

class WorkBase(ABC):
    @abstractmethod
    def wait(self) -> Optional[torch.Tensor]:
        pass

class P2PCommBase(ABC):
    def __init__(self, serializer: Optional[Serializer] = None):
        self.world = get_world()
        self.serializer = serializer

    @abstractmethod
    def send(self, data: torch.Tensor) -> None:
        pass

    @abstractmethod
    def recv(self) -> torch.Tensor:
        pass

    @abstractmethod
    def irecv(self, tag: int, shape: tuple[int, ...]) -> WorkBase:
        pass

    @abstractmethod
    def destroy(self):
        pass

_COMM: Optional[P2PCommBase] = None


def setup_comm(target: Type[P2PCommBase], **kwargs):
    global _COMM
    assert _COMM is None, "Comm already setup"
    _COMM = target(**kwargs)


def get_comm() -> P2PCommBase:
    global _COMM
    assert _COMM is not None, "Comm not setup"
    return _COMM

class TorchWork(WorkBase):
    def __init__(self, tensor: Optional[torch.Tensor], work: dist.Work):
        self.tensor = tensor
        self.work = work

    def wait(self) -> Optional[torch.Tensor]:
        if isinstance(self.work, dist.Work):
            self.work.wait()
        return self.tensor


class TorchP2PComm(P2PCommBase):
    def __init__(
        self,
        fwd_shape: tuple[int, ...],
        bwd_shape: tuple[int, ...],
        fwd_dtype: torch.dtype,
        bwd_dtype: torch.dtype,
        device: torch.device,
        num_prompt_tokens: int,
        serializer: Optional[Serializer] = None,
        init_method: str = "env://",
        backend: str = "nccl",
    ):
        super().__init__(serializer)
        self.logger = get_logger()
        self.fwd_shape, self.bwd_shape = fwd_shape, bwd_shape
        self.fwd_prefill_shape = (fwd_shape[0], num_prompt_tokens, fwd_shape[2])
        self.bwd_prefill_shape = bwd_shape
        self.fwd_dtype, self.bwd_dtype = fwd_dtype, bwd_dtype
        self.device = device
        self.num_prompt_tokens = num_prompt_tokens
        if not os.environ.get("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = "localhost"
        if not os.environ.get("MASTER_PORT"):
            os.environ["MASTER_PORT"] = "29500"
        self.process_group = dist.init_process_group(
            init_method=init_method, backend=backend
        )

    def _send_fwd(self, hidden_states: torch.Tensor, tag: int) -> TorchWork:
        """Sends hidden states to the next stage"""
        self.logger.debug(f"send_fwd({hidden_states=}, {tag=})")
        return TorchWork(None, dist.isend(hidden_states, dst=self.world.rank + 1, tag=tag))

    def _send_bwd(self, next_token: torch.Tensor, tag: int) -> TorchWork:
        """Sends next token to the first stage"""
        self.logger.debug(f"send_bwd({next_token=}, {tag=})")
        return TorchWork(None, dist.isend(next_token, dst=self.world.first_stage_rank, tag=tag))

    def _recv_fwd(self, tag: int, shape: tuple[int, ...]) -> TorchWork:
        """Receives hidden states from the previous stage"""
        hidden_states = torch.empty(shape, dtype=self.fwd_dtype, device=self.device)
        self.logger.debug(f"recv_fwd({hidden_states=}, {tag=})")
        return TorchWork(hidden_states, dist.irecv(hidden_states, src=self.world.rank - 1, tag=tag))

    def _recv_bwd(self, tag: int, shape: tuple[int, ...]) -> TorchWork:
        """Receives next token from the last stage"""
        next_token = torch.empty(shape, dtype=self.bwd_dtype, device=self.device)
        self.logger.debug(f"recv_bwd({next_token=}, {tag=})")
        return TorchWork(next_token, dist.irecv(next_token, src=self.world.last_stage_rank, tag=tag))

    def send(self, tensor: torch.Tensor, tag: int) -> None:
        """Sends tensor (hidden states or next token) to the correct next rank"""
        work = self.isend(tensor, tag)
        if work:
            work.wait()

    def recv(self, tag: int, prefill: bool = False) -> Optional[torch.Tensor]:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        work = self.irecv(tag, prefill)
        if work:
            return work.wait()

    def isend(self, tensor: torch.Tensor, tag: int) -> TorchWork:
        """Sends tensor (hidden states or next token) to the correct next rank"""
        if self.world.size <= 1:
            return None
        send_fwd_or_bwd = self._send_bwd if self.world.is_last_stage else self._send_fwd
        return send_fwd_or_bwd(tensor, tag)

    def irecv(self, tag: int, prefill: bool = False) -> dist.Work:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        if self.world.size <= 1:
            return None
        shape = (
            (self.bwd_shape, self.bwd_prefill_shape)
            if self.world.is_first_stage
            else (self.fwd_shape, self.fwd_prefill_shape)
        )
        recv_fwd_or_bwd = self._recv_bwd if self.world.is_first_stage else self._recv_fwd
        return recv_fwd_or_bwd(tag, shape[int(prefill)])

    def destroy(self):
        if dist.is_initialized():
            dist.destroy_process_group()

class IrohWork(WorkBase):
     def __init__(self, work: Union[SendWork, RecvWork], serializer: Optional[Serializer] = None):
         self.work = work
         self.serializer = serializer

     def wait(self) -> Optional[torch.Tensor]:
         return self.serializer.deserialize(self.work.wait()) if self.serializer else self.work.wait()

class IrohP2PComm(P2PCommBase):
    """IrohNode running receiver and sender in separate processes with non-blocking API"""
    def __init__(self, serializer: Serializer, num_micro_batches: int):
        super().__init__(serializer)
        self.logger = get_logger()
        self.node, self.num_micro_batches = None, num_micro_batches
        if self.world.size <= 1:
            return
        self._setup()

    def _setup(self):
        # Create node
        self.node = Node(self.num_micro_batches)
        self.logger.info(f"Listening on {self.node.node_id()}")
        
        # Connect to remote node
        self.logger.info("Please enter the server's public key: ")
        self.node.connect(input().strip())

        # Wait for connection to be established
        while not self.node.is_ready():
            time.sleep(1 / 100)
        self.logger.info("Connected!")
        
    def irecv(self, tag: int, **kwargs) -> Optional[IrohWork]:
        if self.world.size <= 1:
            return None
        return IrohWork(self.node.irecv(tag), self.serializer)

    def isend(self, tensor: torch.Tensor, tag: int, **kwargs) -> Optional[IrohWork]:
        if self.world.size <= 1:
            return None
        return IrohWork(self.node.isend(self.serializer.serialize(tensor), tag, **kwargs), self.serializer)

    def send(self, tensor: torch.Tensor, tag: int, **kwargs) -> None:
        work = self.isend(tensor, tag)
        if work:
            work.wait()

    def recv(self, tag: int, **kwargs) -> Optional[torch.Tensor]:
        work = self.irecv(tag)
        if work:
            return work.wait()

    def destroy(self):
        pass # TODO: Implement