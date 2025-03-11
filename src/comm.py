import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Type

import torch
import torch.distributed as dist

from iroh_py import create_connector
from logger import get_logger
from serializer import Serializer
from world import get_world


class P2PComm(ABC):
    def __init__(self):
        self.world = get_world()

    @abstractmethod
    def send(self, data: bytes):
        pass

    @abstractmethod
    def recv(self) -> bytes:
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

    def _send_fwd(self, hidden_states: torch.Tensor, tag: int) -> None:
        """Sends hidden states to the next stage"""
        next_rank = self.world.rank + 1
        assert next_rank < self.world.size
        self.logger.debug(f"send_fwd {hidden_states=} to {next_rank=} with {tag=}")
        dist.send(hidden_states, dst=next_rank, tag=tag)

    def _send_bwd(self, next_token: torch.Tensor, tag: int) -> None:
        """Sends next token to the first stage"""
        self.logger.debug(
            f"send_bwd {next_token=} to {self.world.first_stage_rank=} with {tag=}"
        )
        dist.send(next_token, dst=self.world.first_stage_rank, tag=tag)

    def _recv_fwd(self, tag: int, shape: tuple[int, ...]) -> torch.Tensor:
        """Receives hidden states from the previous stage"""
        prev_rank = self.world.rank - 1
        assert prev_rank >= 0
        hidden_states = torch.empty(shape, dtype=self.fwd_dtype, device=self.device)
        self.logger.debug(
            f"recv_fwd into {hidden_states=} from {prev_rank=} with {tag=}"
        )
        dist.recv(hidden_states, src=prev_rank, tag=tag)
        self.logger.debug(f"recv_fwd done {hidden_states=}")
        return hidden_states

    def _recv_bwd(self, tag: int, shape: tuple[int, ...]) -> torch.Tensor:
        """Receives next token from the last stage"""
        next_token = torch.empty(shape, dtype=self.bwd_dtype, device=self.device)
        self.logger.debug(
            f"recv_bwd into {next_token=} from {self.world.last_stage_rank=} with {tag=}"
        )
        dist.recv(next_token, src=self.world.last_stage_rank, tag=tag)
        self.logger.debug(f"recv_bwd done {next_token=}")
        return next_token

    def send(self, tensor: torch.Tensor, tag: int) -> None:
        """Sends tensor (hidden states or next token) to the correct next rank"""
        if self.world.size == 1:
            return None
        _send = self._send_bwd if self.world.is_last_stage else self._send_fwd
        _send(tensor, tag)

    def recv(self, tag: int, prefill: bool = False) -> torch.Tensor:
        """Receives tensor (hidden states or next token) from the correct previous rank"""
        if self.world.size == 1:
            return None
        shape = (
            (self.bwd_shape, self.bwd_prefill_shape)
            if self.world.is_first_stage
            else (self.fwd_shape, self.fwd_prefill_shape)
        )
        _recv = self._recv_bwd if self.world.is_first_stage else self._recv_fwd
        return _recv(tag, shape[int(prefill)])

    def destroy(self):
        if dist.is_initialized():
            dist.destroy_process_group()


class IrohP2PComm(P2PComm):
    def __init__(self, serializer: Serializer, device: torch.device):
        super().__init__()
        self.serializer = serializer
        self.device = device
        self.logger = get_logger()
        self.node = None

        if self.world.size <= 1:
            return

        # Create iroh node
        self.node = create_connector()
        self.logger.info(f"Connect to node with: {self.node.get_node_id()}")

        # Connect to the remote node
        server_public_key = input("Please enter the server's public key: ").strip()
        self.logger.info(f"Connecting to node with key: {server_public_key}")
        self.node.connect(server_public_key)

        # Wait for connection to be established
        while not self.node.is_ready():
            time.sleep(1 / 100)

    def send(self, data: torch.Tensor, **kwargs) -> None:
        if self.world.size == 1:
            return None
        serialized_data = self.serializer.serialize(data)
        self.logger.info(f"Sending data to node with key: {self.node.get_node_id()}")
        self.node.send(serialized_data)

    def recv(self, **kwargs) -> torch.Tensor:
        if self.world.size == 1:
            return None
        serialized_data = self.node.recv()
        self.logger.info(f"Received data from node with key: {self.node.get_node_id()}")
        return self.serializer.deserialize(serialized_data).to(self.device)

    def destroy(self):
        if self.node is not None:
            self.node.shutdown()
