from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from world import World


class P2PComm(ABC):
    def __init__(self, world: World):
        self.world = world

    @abstractmethod
    def send(self, data: bytes):
        pass

    @abstractmethod
    def recv(self) -> bytes:
        pass


class TorchP2PComm(P2PComm):
    def __init__(
        self,
        world: World,
        fwd_shape: tuple[int, ...],
        bwd_shape: tuple[int, ...],
        device: torch.device,
        num_prompt_tokens: int,
        fwd_dtype: torch.dtype = torch.bfloat16,
        bwd_dtype: torch.dtype = torch.long,
        init_method: str = "env://",
        backend: str = "nccl",
    ):
        super().__init__(world)
        self.fwd_shape, self.bwd_shape = fwd_shape, bwd_shape
        self.fwd_prefill_shape = (fwd_shape[0], num_prompt_tokens, fwd_shape[2])
        self.bwd_prefill_shape = bwd_shape
        self.fwd_dtype, self.bwd_dtype = fwd_dtype, bwd_dtype
        self.device = device
        self.num_prompt_tokens = num_prompt_tokens
        self.process_group = dist.init_process_group(
            init_method=init_method, backend=backend
        )

    def _send_fwd(self, hidden_states: torch.Tensor, tag: int) -> None:
        """Sends hidden states to the next stage"""
        next_rank = self.world.rank + 1
        assert next_rank < self.world.size
        # print(f"{self.world.rank=} send_fwd to {next_rank=}")
        dist.send(hidden_states, dst=next_rank, tag=tag)

    def _send_bwd(self, next_token: torch.Tensor, tag: int) -> None:
        """Sends next token to the first stage"""
        # print(f"{self.world.rank=} send_bwd to {self.world.first_stage_rank=}")
        dist.send(next_token, dst=self.world.first_stage_rank, tag=tag)

    def _recv_fwd(self, tag: int, shape: tuple[int, ...]) -> torch.Tensor:
        """Receives hidden states from the previous stage"""
        prev_rank = self.world.rank - 1
        assert prev_rank >= 0
        hidden_states = torch.empty(shape, dtype=self.fwd_dtype, device=self.device)
        # print(f"{self.world.rank=} recv_fwd into {hidden_states=} from {prev_rank=}")
        dist.recv(hidden_states, src=prev_rank, tag=tag)
        return hidden_states

    def _recv_bwd(self, tag: int, shape: tuple[int, ...]) -> torch.Tensor:
        """Receives next token from the last stage"""
        next_token = torch.empty(shape, dtype=self.bwd_dtype, device=self.device)
        # print(
        #     f"{self.world.rank=} recv_bwd into {next_token=} from {self.world.last_stage_rank=}"
        # )
        dist.recv(next_token, src=self.world.last_stage_rank, tag=tag)
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


class IrohP2PComm(P2PComm):
    pass
