import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Type, Union
from concurrent.futures import ThreadPoolExecutor, Future

import torch
import torch.distributed as dist

from iroh_py import create_connector
from logger import get_logger
from serializer import Serializer
from world import get_world

class Request(ABC):
    @abstractmethod
    def wait(self) -> Optional[torch.Tensor]:
        pass

class P2PComm(ABC):
    def __init__(self):
        self.world = get_world()

    @abstractmethod
    def send(self, data: torch.Tensor) -> None:
        pass

    @abstractmethod
    def recv(self) -> torch.Tensor:
        pass

    @abstractmethod
    def isend(self, tensor: torch.Tensor) -> Request:
        pass

    @abstractmethod
    def irecv(self, tag: int, shape: tuple[int, ...]) -> Request:
        pass

    @abstractmethod
    def destroy(self):
        pass

_COMM: Optional[P2PComm] = None


def setup_comm(target: Type[P2PComm], **kwargs):
    global _COMM
    assert _COMM is None, "Comm already setup"
    _COMM = target(**kwargs)


def get_comm() -> P2PComm:
    global _COMM
    assert _COMM is not None, "Comm not setup"
    return _COMM

class TorchRequest(Request):
    def __init__(self, tensor: torch.Tensor, work: Union[None, int, dist.Work]):
        self.tensor = tensor
        self.work = work

    def wait(self) -> Optional[torch.Tensor]:
        if isinstance(self.work, dist.Work):
            self.work.wait()
        return self.tensor


class TorchP2PComm(P2PComm):
    def __init__(
        self,
        fwd_shape: tuple[int, ...],
        bwd_shape: tuple[int, ...],
        fwd_dtype: torch.dtype,
        bwd_dtype: torch.dtype,
        device: torch.device,
        num_prompt_tokens: int,
        init_method: str = "env://",
        backend: str = "nccl",
    ):
        super().__init__()
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

    def _send_fwd(self, send_func: Callable, hidden_states: torch.Tensor, tag: int) -> TorchRequest:
        """Sends hidden states to the next stage"""
        if self.world.size == 1:
            return TorchRequest(None, None)
        next_rank = self.world.rank + 1
        assert next_rank < self.world.size
        self.logger.debug(f"send_fwd {hidden_states=} to {next_rank=} with {tag=}")
        send_req = send_func(hidden_states, dst=next_rank, tag=tag)
        return TorchRequest(None, send_req)

    def _send_bwd(self, send_func: Callable, next_token: torch.Tensor, tag: int) -> TorchRequest:
        """Sends next token to the first stage"""
        if self.world.size == 1:
            return TorchRequest(None, None)
        self.logger.debug(
            f"send_bwd {next_token=} to {self.world.first_stage_rank=} with {tag=}"
        )
        send_req = send_func(next_token, dst=self.world.first_stage_rank, tag=tag)
        return TorchRequest(None, send_req)

    def _recv_fwd(self, recv_func: Callable, tag: int, shape: tuple[int, ...]) -> TorchRequest:
        """Receives hidden states from the previous stage"""
        if self.world.size == 1:
            return TorchRequest(None, None)
        prev_rank = self.world.rank - 1
        assert prev_rank >= 0
        hidden_states = torch.empty(shape, dtype=self.fwd_dtype, device=self.device)
        self.logger.debug(
            f"recv_fwd into {hidden_states=} from {prev_rank=} with {tag=}"
        )
        recv_req = recv_func(hidden_states, src=prev_rank, tag=tag)
        return TorchRequest(hidden_states, recv_req)

    def _recv_bwd(self, recv_func: Callable, tag: int, shape: tuple[int, ...]) -> TorchRequest:
        """Receives next token from the last stage"""
        if self.world.size == 1:
            return TorchRequest(None, None)
        next_token = torch.empty(shape, dtype=self.bwd_dtype, device=self.device)
        self.logger.debug(
            f"recv_bwd into {next_token=} from {self.world.last_stage_rank=} with {tag=}"
        )
        recv_req = recv_func(next_token, src=self.world.last_stage_rank, tag=tag)
        return TorchRequest(next_token, recv_req)

    def send(self, tensor: torch.Tensor, tag: int) -> None:
        """Sends tensor (hidden states or next token) to the correct next rank"""
        if self.world.size == 1:
            return
        send_fwd_or_bwd = self._send_bwd if self.world.is_last_stage else self._send_fwd
        send_fwd_or_bwd(dist.send, tensor, tag).wait()

    def recv(self, tag: int, prefill: bool = False) -> torch.Tensor:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        if self.world.size == 1:
            return
        shape = (
            (self.bwd_shape, self.bwd_prefill_shape)
            if self.world.is_first_stage
            else (self.fwd_shape, self.fwd_prefill_shape)
        )
        recv_fwd_or_bwd = self._recv_bwd if self.world.is_first_stage else self._recv_fwd
        return recv_fwd_or_bwd(dist.recv, tag, shape[int(prefill)]).wait()

    def isend(self, tensor: torch.Tensor, tag: int) -> Request:
        """Sends tensor (hidden states or next token) to the correct next rank"""
        if self.world.size == 1:
            return Request(None, None)
        send_fwd_or_bwd = self._send_bwd if self.world.is_last_stage else self._send_fwd
        return send_fwd_or_bwd(dist.isend, tensor, tag)

    def irecv(self, tag: int, prefill: bool = False) -> Request:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        if self.world.size == 1:
            return Request(None, None)
        shape = (
            (self.bwd_shape, self.bwd_prefill_shape)
            if self.world.is_first_stage
            else (self.fwd_shape, self.fwd_prefill_shape)
        )
        recv_fwd_or_bwd = self._recv_bwd if self.world.is_first_stage else self._recv_fwd
        return recv_fwd_or_bwd(dist.irecv, tag, shape[int(prefill)])

    def destroy(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class IrohRequest(Request):
    def __init__(self, future: Future):
        self.future = future

    def wait(self) -> torch.Tensor:
        return self.future.result()


class IrohP2PComm(P2PComm):
    def __init__(self, serializer: Serializer, device: torch.device):
        super().__init__()
        self.serializer = serializer
        self.device = device
        self.logger = get_logger()
        self.node = None
        self.executor = ThreadPoolExecutor(max_workers=2) # one for sending, one for receiving

        if self.world.size <= 1:
            return
        self._setup_iroh()

    def _setup_iroh(self):
        # Create iroh node
        self.node = create_connector()
        self.logger.info(f"Connect to node with: {self.node.get_node_id()}")

        # Connect to the remote node
        server_public_key = input("Please enter the server's public key: ").strip()
        self.node.connect(server_public_key)

        # Wait for connection to be established
        while not self.node.is_ready():
            time.sleep(1 / 100)

    def send(self, data: torch.Tensor, **kwargs) -> None:
        if self.world.size == 1:
            return
        serialized_data = self.serializer.serialize(data)
        self.logger.debug(f"Sending data to node with key: {self.node.get_node_id()}")
        self.node.send(serialized_data)

    def recv(self, **kwargs) -> torch.Tensor:
        if self.world.size == 1:
            return
        serialized_data = self.node.recv()
        self.logger.debug(f"Received data from node with key: {self.node.get_node_id()}")
        return self.serializer.deserialize(serialized_data).to(self.device)

    def isend(self, tensor: torch.Tensor, **kwargs) -> Optional[Request]:
        if self.world.size == 1:
            return
        future = self.executor.submit(self.send, tensor, **kwargs)
        return IrohRequest(future)

    def irecv(self, tag: int, shape: tuple[int, ...]) -> Optional[Request]:
        if self.world.size == 1:
            return
        future = self.executor.submit(self.recv, tag, shape)
        return IrohRequest(future)

    def destroy(self):
        if self.node is not None:
            self.executor.shutdown()
            self.node.shutdown()