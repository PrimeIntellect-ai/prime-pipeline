from time import perf_counter
from typing import Dict, Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from comm import get_comm
from logger import get_logger
from world import get_world

# Setup compile flags
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True


def multinomial_sample_one_no_sync(probs_sort: Tensor) -> Tensor:
    """Sample from a multinomial distribution without a cuda synchronization from (B, S) to (B, 1)"""
    # Draw exponential random variables
    q = torch.empty_like(probs_sort).exponential_(1)

    # Select the argmax
    next_tokens = torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.long)

    return next_tokens


def logits_to_probs(logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
    """Convert logits to probabilities from (B, S, H) to (B, S)"""
    # Temperature scaling (capped at minimum temperature of 1e-5)
    logits = logits / max(temperature, 1e-5)

    # Top-k sampling
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    # Softmax
    probs = torch.nn.functional.softmax(logits, dim=-1)

    return probs


def sample(logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
    """Sample from a multinomial distribution without a cuda synchronization"""
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next


def causal_mask(b, h, q, kv):
    """Causal mask using for flex attention"""
    return q >= kv


def adjust_mask(block_mask: BlockMask, input_pos: Tensor, max_seq_length: int) -> BlockMask:
    """Adjust the mask for the given input position"""
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, max_seq_length)

    return mask


def model_forward(
    model: nn.Module,
    micro_batch_idx: int,
    block_mask: BlockMask,
    prompt_tokens: Optional[Tensor],
    input_pos: Tensor,
    hidden_states: Optional[Tensor] = None,
) -> Tensor:
    """Wrapper of forward pass of the model for torch.compile"""
    return model(micro_batch_idx, block_mask, input_pos, prompt_tokens, hidden_states)


def adjust_mask_and_model_forward(
    model: nn.Module,
    micro_batch_idx: int,
    block_mask: BlockMask,
    input_pos: Tensor,
    prompt_tokens: Tensor,
    hidden_states: Optional[Tensor] = None,
) -> Tensor:
    """Wrapper for adjust mask and forward pass of the model for torch.compile"""
    mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
    return model_forward(model, micro_batch_idx, mask, input_pos, prompt_tokens, hidden_states)


def prefill(
    model: nn.Module,
    decoded_tokens: Tensor,
    micro_batch_size: int,
    num_prompt_tokens: int,
    **sampling_kwargs,
) -> Dict:
    """
    Prefill prompt tokens to start inference. Assumes that the model caches are
    set up, the prompt is tokenized and put into a tensor of shape [batch_size, seq_length].
    Automatically handles pipelining using synchronous communication primitives. Returns
    a dictionary of metrics.

    Args:
        model: Transformer
        decoded_tokens: Tensor of shape [batch_size, seq_length]
        input_pos: Tensor of shape [num_prompt_tokens]
        micro_batch_size: int
        **sampling_kwargs: Dict of kwargs for the sample function

    Returns:
        metrics: Dict of metrics
    """
    world, logger, comm = get_world(), get_logger(), get_comm()
    logger.debug(f"prefill(model, {decoded_tokens=}, {num_prompt_tokens=}, {micro_batch_size=})")

    # Create input positions
    input_pos = torch.arange(num_prompt_tokens, device=decoded_tokens.device)

    # No new tokens to generate
    if decoded_tokens.size(-1) == input_pos.shape[0]:
        logger.info("No new token to generate.")
        return

    # Create block mask
    block_mask = create_block_mask(
        causal_mask,
        1,
        1,
        input_pos.shape[0],
        model.max_seq_length,
        device=decoded_tokens.device,
    )

    batch_size = decoded_tokens.size(0)
    num_prompt_tokens = input_pos.size(0)
    num_micro_batches = batch_size // micro_batch_size
    next_token, hidden_states = None, None
    start_prefill = perf_counter()
    for micro_batch_idx in range(num_micro_batches):
        # Receive hidden states from previous stage
        if not world.is_first_stage:
            hidden_states = comm.recv(tag=micro_batch_idx, prefill=True)

        # Get micro-batch prompt tokens
        start_idx = micro_batch_idx * micro_batch_size
        end_idx = start_idx + micro_batch_size
        prompt_tokens = decoded_tokens[start_idx:end_idx, :num_prompt_tokens]

        # Forward pass
        outputs = model_forward(
            model,
            micro_batch_idx,
            block_mask,
            prompt_tokens,
            input_pos,
            hidden_states,
        )

        # Sample next token
        if world.is_last_stage:
            outputs = sample(outputs, **sampling_kwargs)

        # Send hidden states or next token to next stage
        comm.isend(outputs, tag=micro_batch_idx)

        if world.is_first_stage:
            # Receive next token from last stage
            if world.size == 1:
                next_token = outputs
            else:
                next_token = comm.recv(tag=micro_batch_idx, prefill=True)
            decoded_tokens[start_idx:end_idx, num_prompt_tokens] = next_token.squeeze()

    logger.debug(f"Prefilled tokens {decoded_tokens[:, num_prompt_tokens]=}")

    return {
        "prefill_time": perf_counter() - start_prefill,
        "prefill_tokens": batch_size * num_prompt_tokens,
    }


def decode(
    model: nn.Module,
    decoded_tokens: Tensor,
    num_prompt_tokens: int,
    micro_batch_size: int,
    **sampling_kwargs,
) -> Dict:
    """
    Auto-regressively decodes tokens until max new tokens are reached. Assumes that
    the model caches are set up, the prompt is tokenized and split into micro-batches
    and the input_pos is a tensor of shape [num_prompt_tokens]. Automatically handles
    interleaved pipelining to achieve near peak device utilization. Returns a dictionary
    of metrics.

    Args:
        model: Transformer
        batched_cur_tokens: List[Tensor], each of shape [micro_batch_size, 1]
        input_pos: Tensor of shape [1]
        num_new_tokens: int
        **sampling_kwargs: Dict of kwargs for the sample function

    Returns:
        metrics: Dict of metrics
    """
    world, logger, comm = get_world(), get_logger(), get_comm()
    logger.debug(f"decode(model, {decoded_tokens=}, {num_prompt_tokens=}, {micro_batch_size=})")

    starting_pos = num_prompt_tokens + 1
    if starting_pos >= decoded_tokens.size(-1):
        logger.info("No new token to generate.")
        return

    batch_size = decoded_tokens.size(0)
    num_micro_batches = batch_size // micro_batch_size
    device = model.layers[0].feed_forward.w1.weight.device

    # Create block mask
    block_mask = create_block_mask(
        causal_mask,
        1,
        1,
        model.max_seq_length,
        model.max_seq_length,
        device=device,
    )

    # Single-node
    if world.size == 1:
        time_per_token = []
        for token_idx in range(starting_pos, decoded_tokens.size(-1)):
            time_per_micro_batch = []
            input_pos = torch.tensor([token_idx], device=device, dtype=torch.long)
            mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
            for micro_batch_idx in range(num_micro_batches):
                start_time = perf_counter()
                start_idx = micro_batch_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                cur_tokens = decoded_tokens[start_idx:end_idx, token_idx - 1].unsqueeze(-1)
                logits = model_forward(model, micro_batch_idx, mask, cur_tokens, input_pos)
                next_token = sample(logits, **sampling_kwargs)
                decoded_tokens[start_idx:end_idx, token_idx] = next_token.squeeze()
                time_per_micro_batch.append(perf_counter() - start_time)
            time_per_token.append(time_per_micro_batch)

        logger.debug(f"Decoded tokens {decoded_tokens[:, num_prompt_tokens:]=}")

    # Multi-node
    else:
        recv_reqs = [None] * num_micro_batches
        if world.is_first_stage:
            input_pos = torch.tensor([starting_pos], device=device, dtype=torch.long)
            mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
            for micro_batch_idx in range(num_micro_batches):
                start_idx = micro_batch_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                cur_tokens = decoded_tokens[start_idx:end_idx, starting_pos - 1].unsqueeze(-1)
                hidden_states = model_forward(
                    model,
                    micro_batch_idx,
                    mask,
                    cur_tokens,
                    input_pos,
                )
                comm.isend(hidden_states, tag=micro_batch_idx)
                recv_reqs[micro_batch_idx] = comm.irecv(tag=micro_batch_idx)
                recv_start = perf_counter()

        # Decode interleaved
        next_token, hidden_states = None, None
        time_per_token = []
        for token_idx in range(starting_pos, decoded_tokens.size(-1)):
            time_per_micro_batch = []
            input_pos = torch.tensor([token_idx], device=device, dtype=torch.long)
            mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
            if world.is_first_stage:
                input_pos += 1
            for micro_batch_idx in range(num_micro_batches):
                start_time = perf_counter()
                if world.is_first_stage:
                    start_idx = micro_batch_idx * micro_batch_size
                    end_idx = start_idx + micro_batch_size
                    next_token = recv_reqs[micro_batch_idx].wait()
                    logger.debug(f"Waited for next token {micro_batch_idx}: {perf_counter() - recv_start:.2f} seconds")
                    decoded_tokens[start_idx:end_idx, token_idx] = next_token.squeeze()
                else:
                    recv_start = perf_counter()
                    hidden_states = comm.recv(tag=micro_batch_idx)
                    logger.debug(f"Waited for hidden states {micro_batch_idx}: {perf_counter() - recv_start:.2f} seconds")

                # Skip forward and send on last iteration for first stage
                if world.is_first_stage and token_idx >= decoded_tokens.size(-1) - 1:
                    continue

                # Forward pass
                outputs = model_forward(
                    model,
                    micro_batch_idx,
                    mask,
                    next_token,
                    input_pos,
                    hidden_states,
                )

                if world.is_last_stage:
                    outputs = sample(outputs, **sampling_kwargs)

                # Send hidden states or next token
                comm.isend(outputs, tag=micro_batch_idx)

                # Schedule next recv
                if world.is_first_stage:
                    recv_reqs[micro_batch_idx] = comm.irecv(tag=micro_batch_idx)
                    recv_start = perf_counter()
                time_per_micro_batch.append(perf_counter() - start_time)
            time_per_token.append(time_per_micro_batch)

    logger.debug(f"Decoded tokens {decoded_tokens[:, num_prompt_tokens:]=}")
    return {
        "time_per_token": time_per_token,
        "num_decode_tokens": batch_size * (decoded_tokens.size(-1) - num_prompt_tokens),
    }


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_tokens: Tensor,
    batch_size: int,
    num_new_tokens: int,
    micro_batch_size: int,
    **sampling_kwargs,
) -> Tuple[Tensor, Dict, Dict]:
    """
    Generate tokens.

    Args:
        model: Transformer
        prompt_tokens: Tensor of shape [batch_size, num_prompt_tokens]
        num_new_tokens: int
        micro_batch_size: int
        **sampling_kwargs: Dict of kwargs for the sample function
    """
    # Setup model cache
    num_prompt_tokens = prompt_tokens.size(-1)
    num_micro_batches = batch_size // micro_batch_size
    num_total_tokens = min(num_prompt_tokens + num_new_tokens, model.config.block_size)
    device = model.layers[0].feed_forward.w1.weight.device
    with torch.device(device):
        model.setup_caches(
            num_micro_batches=num_micro_batches,
            max_micro_batch_size=micro_batch_size,
            max_seq_length=num_total_tokens,
        )

    # Allocate tensor for decoded tokens
    decoded_tokens = torch.empty(batch_size, num_total_tokens, dtype=prompt_tokens.dtype, device=device)
    decoded_tokens[:, :num_prompt_tokens] = prompt_tokens

    # Prefill prompt tokens in-place
    prefill_metrics = prefill(
        model=model,
        decoded_tokens=decoded_tokens,
        micro_batch_size=micro_batch_size,
        num_prompt_tokens=num_prompt_tokens,
        **sampling_kwargs,
    )

    # Decode remaining tokens in-place
    decode_metrics = decode(
        model=model,
        decoded_tokens=decoded_tokens,
        micro_batch_size=micro_batch_size,
        num_prompt_tokens=num_prompt_tokens,
        **sampling_kwargs,
    )

    return decoded_tokens, prefill_metrics, decode_metrics
