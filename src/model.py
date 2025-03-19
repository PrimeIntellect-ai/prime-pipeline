# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import time
from dataclasses import dataclass
from typing import Optional
from time import perf_counter

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    flex_attention,
)

from logger import get_logger

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    rope_scaling: Optional[dict] = None

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [
            config
            for config in transformer_configs
            if config.lower() in str(name).lower()
        ]

        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), (
                name
            )  # make sure only one 'best' match

        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(
        block_size=16384, vocab_size=32000, n_layer=32, dim=4096, rope_base=1000000
    ),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(
        n_layer=48,
        n_head=64,
        dim=8192,
        vocab_size=32000,
        n_local_heads=8,
        intermediate_size=22016,
        rope_base=1000000,
    ),  # CodeLlama-34B-Python-hf
    "70B": dict(
        n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672
    ),
    "Mistral-7B": dict(
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=32000,
    ),
    "stories15M": dict(n_layer=6, n_head=6, dim=288),
    "stories110M": dict(n_layer=12, n_head=12, dim=768),
    "llama-3-8b": dict(
        block_size=8192,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3-70b": dict(
        block_size=8192,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
    ),
    "llama-3.1-8b": dict(
        block_size=131072,
        n_layer=32,
        n_head=32,
        n_local_heads=8,
        dim=4096,
        intermediate_size=14336,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "llama-3.1-70b": dict(
        block_size=131072,
        n_layer=80,
        n_head=64,
        n_local_heads=8,
        dim=8192,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
    "llama-3.1-405b": dict(
        block_size=131072,
        n_layer=126,
        n_head=128,
        n_local_heads=8,
        dim=16384,
        intermediate_size=53248,
        vocab_size=128256,
        rope_base=500000,
        rope_scaling=dict(
            factor=8.0,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            original_max_position_embeddings=8192,
        ),
    ),
}


class KVCache(nn.Module):
    def __init__(
        self,
        num_micro_batches,
        max_micro_batch_size,
        max_seq_length,
        n_heads,
        head_dim,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        cache_shape = (
            num_micro_batches,
            max_micro_batch_size,
            n_heads,
            max_seq_length,
            head_dim,
        )
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, micro_batch_idx, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[micro_batch_idx, :, :, input_pos, :] = k_val.unsqueeze(0)
        v_out[micro_batch_idx, :, :, input_pos, :] = v_val.unsqueeze(0)

        return k_out[micro_batch_idx], v_out[micro_batch_idx]


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod
        self.logger = get_logger()

    def setup_caches(self, num_micro_batches, max_micro_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= num_micro_batches * max_micro_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = num_micro_batches * max_micro_batch_size
        dtype = self.layers[0].feed_forward.w1.weight.dtype
        # For quantized layers, dtype is encoded in scales
        if hasattr(self.output, "scales"):
            dtype = self.output.scales.dtype
        elif hasattr(self.output, "scales_and_zeros"):
            dtype = self.output.scales_and_zeros.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                num_micro_batches,
                max_micro_batch_size,
                max_seq_length,
                self.config.n_local_heads,
                head_dim,
                dtype,
            )

        self.freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            dtype,
            self.config.rope_scaling,
        )

    def forward(
        self,
        micro_batch_idx: int,
        mask: BlockMask,
        input_pos: Tensor,
        input_ids: Optional[Tensor],
        hidden_states: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        # t0 = perf_counter()
        mask.mask_mod = self.get_mask_mod(mask.mask_mod, input_pos[0])
        freqs_cis = self.freqs_cis[input_pos]

        assert hidden_states is not None or input_ids is not None, (
            "Must provide either hidden states or input ids"
        )
        x = hidden_states if hidden_states is not None else input_ids

        x = self.tok_embeddings(x)

        for layer in self.layers:
            x = layer(micro_batch_idx, x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        # self.logger.debug(f"Forward pass took {(perf_counter() - t0) * 1000:.2f}ms")
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        micro_batch_idx: int,
        x: Tensor,
        input_pos: Tensor,
        freqs_cis: Tensor,
        mask: BlockMask,
    ) -> Tensor:
        h = x + self.attention(
            micro_batch_idx, self.attention_norm(x), freqs_cis, mask, input_pos
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        micro_batch_idx: int,
        x: Tensor,
        freqs_cis: Tensor,
        mask: BlockMask,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(micro_batch_idx, input_pos, k, v)

        y = flex_attention(
            q, k, v, block_mask=mask, enable_gqa=(self.n_head != self.n_local_heads)
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class TransformerShard(Transformer):
    def __init__(self, rank: int, world_size: int, model: Transformer) -> None:
        # Save parameters
        self.stage = rank
        self.num_stages = world_size
        self.__dict__.update(model.__dict__)

        # Setup sharded model
        layer_indices = TransformerShard.distribute_layers(
            stage=rank, num_stages=world_size, num_layers=model.config.n_layer
        )
        self.tok_embeddings = (
            model.tok_embeddings if self.is_first_stage else nn.Identity()
        )
        self.layers = nn.ModuleList([model.layers[i] for i in layer_indices])
        self.norm = model.norm if self.is_last_stage else nn.Identity()
        self.output = model.output if self.is_last_stage else nn.Identity()

        del model

    @property
    def is_first_stage(self) -> bool:
        return self.stage == 0

    @property
    def is_last_stage(self) -> bool:
        return self.stage == self.num_stages - 1

    @staticmethod
    def distribute_layers(stage: int, num_stages: int, num_layers: int) -> list[int]:
        layers_per_gpu = [
            num_layers // num_stages + (1 if i < num_layers % num_stages else 0)
            for i in range(num_stages)
        ]
        start_layer = sum(layers_per_gpu[:stage])
        return list(range(start_layer, start_layer + layers_per_gpu[stage]))


if __name__ == "__main__":
    print("Running model.py")

    device = "cpu"
    dtype = torch.float16

    world_size = 2

    def get_shard(rank: int) -> TransformerShard:
        print(f"Getting shard for rank {rank}")
        return TransformerShard(
            rank=rank,
            world_size=world_size,
            model=load_model(
                checkpoint_path=Path(
                    "checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
                ),
                device=device,
                precision=dtype,
            ),
        )

    # Load full model and model shards
    from pathlib import Path

    from utils import load_model

    model = load_model(
        checkpoint_path=Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"),
        device=device,
        precision=dtype,
    )
    model1 = get_shard(rank=0)
    model2 = get_shard(rank=1)

    # Setup caches
    batch_size = 16
    num_micro_batches = 2
    micro_batch_size = batch_size // num_micro_batches

    seq_length = 32  # Or any reasonable sequence length for testing
    model.setup_caches(
        num_micro_batches=1, max_micro_batch_size=batch_size, max_seq_length=seq_length
    )
    model1.setup_caches(
        num_micro_batches=num_micro_batches,
        max_micro_batch_size=micro_batch_size,
        max_seq_length=seq_length,
    )
    model2.setup_caches(
        num_micro_batches=num_micro_batches,
        max_micro_batch_size=micro_batch_size,
        max_seq_length=seq_length,
    )

    # Test tokens
    test_tokens = torch.randint(0, model1.config.vocab_size, (batch_size, 4))
    input_pos = torch.arange(test_tokens.size(-1))

    from torch.nn.attention.flex_attention import create_block_mask

    def causal_mask(b, h, q, kv):
        return q >= kv

    mask = create_block_mask(
        causal_mask, 1, 1, input_pos.shape[0], model1.max_seq_length, device=device
    )

    out = model(0, mask, test_tokens, input_pos)

    split_test_tokens = torch.split(test_tokens, micro_batch_size, dim=0)
    shards_out = []
    for i, split_test_token in enumerate(split_test_tokens):
        hidden_states = model1(i, mask, split_test_token, input_pos)
        shard_out = model2(i, mask, hidden_states, input_pos)
        shards_out.append(shard_out)

    shards_out = torch.cat(shards_out, dim=0)

    import code

    code.interact(local=dict(globals(), **locals()))
