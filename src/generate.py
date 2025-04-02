from concurrent.futures import Future
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from tqdm import tqdm

from .comm import get_comm
from .logger import get_logger
from .utils import fake_future, get_device
from .world import get_world

# Setup compile flags
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch._dynamo.config.cache_size_limit = 64


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
    """Adjust the mask for the given input position during decode"""
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, max_seq_length)

    return mask


def micro_step(
    model: nn.Module,
    mask: BlockMask,
    input_pos: Tensor,
    inputs: Tensor,
    micro_batch_idx: int,
    sampling_kwargs: Dict,
    recv_kwargs: Dict,
    skip_send: bool = False,
    skip_recv: bool = False,
) -> Tuple[Optional[Future], Optional[Future], Optional[List[int]]]:
    """
    Computes one micro-step in the decode pipeline. It gets a future of the next
    inputs - tokens for the first stage, hidden states for all other stages. The
    future may be created manually, e.g. when filling up the pipeline, or a
    represent a pending receive request. The function awaits the inputs
    (blocking) and then computes the forward pass. Optionally, on the last
    stage, it also samples a token.  The output - whether hidden state or token
    - is then asynchronously sent to the corresponding next stage and a new
    receive request is scheduled, unless the skip_recv flag is set. The function
    returns the send future, the receive future and decoded micro batch tokens (on
    the stage worker). Sends and receives may be skipped, in which case the
    corresponding future is None.
    """
    world, logger = get_world(), get_logger()

    # Wait to receive next token or hidden states
    start_recv = perf_counter()
    inputs = inputs.result()
    logger.debug(f"Blocked for receiving {micro_batch_idx=} for {(perf_counter() - start_recv) * 1000:.02f}ms")

    # Save results on rank 0
    next_tokens = []
    if world.is_first_stage:
        next_tokens = inputs

    # Forward pass + sample
    start_forward = perf_counter()
    outputs = forward(model, micro_batch_idx, mask, input_pos, inputs)
    if world.is_last_stage:
        outputs = sample(outputs, **sampling_kwargs)
    torch.cuda.synchronize()
    logger.debug(f"Forward {micro_batch_idx=} took {(perf_counter() - start_forward) * 1000:.02f}ms")

    # Send outputs to next stage, unless otherwise specified
    send_future = None
    if not skip_send:
        start_send = perf_counter()
        send_future = get_comm().isend(outputs, tag=micro_batch_idx)
        logger.debug(f"Blocked for sending {micro_batch_idx=} for {(perf_counter() - start_send) * 1000:.02f}ms")

    # Schedule next receive, unless otherwise specified
    recv_future = None
    if not skip_recv:
        if world.is_first_stage and world.is_last_stage:
            recv_future = fake_future(outputs)
        else:
            recv_future = get_comm().irecv(tag=micro_batch_idx, **recv_kwargs)

    return send_future, recv_future, next_tokens


def prefill(
    model: nn.Module,
    prompt_tokens: List[Tensor],
    num_prompt_tokens: int,
    num_micro_batches: int,
    micro_batch_size: int,
    pbar: Optional[tqdm] = None,
    **sampling_kwargs,
) -> Tuple[Optional[List[Tensor]], float]:
    """
    Prefill prompt tokens to start inference. Assumes that the model caches are
    set up, the prompt is tokenized and put into a tensor of shape [batch_size,
    seq_length].  Automatically handles pipelining using synchronous
    communication primitives. Returns the batched decoded tokens and the time
    taken to prefill.

    Args:
        model: Transformer
        prompt_tokens: Tensor[batch_size, num_prompt_tokens]
        micro_batch_size: int
        pbar: Optional progress bar to update
        **sampling_kwargs: Dict of kwargs for the sample function

    Returns:
        decoded_tokens: Tensor[batch_size, 1]
        prefill_time: float
    """
    world, logger, comm, device = get_world(), get_logger(), get_comm(), get_device()
    logger.debug("Calling prefill")
    start_prefill = perf_counter()

    # Create input positions
    input_pos = torch.arange(num_prompt_tokens, device=device)

    # Create block mask
    B, H, Q_LEN, KV_LEN = 1, 1, input_pos.shape[0], model.max_seq_length
    block_mask = create_block_mask(causal_mask, B, H, Q_LEN, KV_LEN, device=device)

    # Setup receive kwargs
    bwd_prefill_shape = (micro_batch_size, 1)
    fwd_prefill_shape = (micro_batch_size, num_prompt_tokens, model.config.dim)
    bwd_dtype = prompt_tokens[0].dtype
    fwd_dtype = model.layers[0].feed_forward.w1.weight.dtype
    shape = bwd_prefill_shape if world.is_first_stage else fwd_prefill_shape
    dtype = bwd_dtype if world.is_first_stage else fwd_dtype
    recv_kwargs = {"shape": shape, "dtype": dtype}

    # Warmup
    recv_futures = []
    for micro_batch_idx in range(num_micro_batches):
        if world.is_first_stage:
            recv_futures.append(fake_future(prompt_tokens[micro_batch_idx].clone()))
        else:
            recv_futures.append(comm.irecv(tag=micro_batch_idx, **recv_kwargs))

    for micro_batch_idx in range(num_micro_batches):
        _, recv_future, _ = micro_step(
            model=model,
            mask=block_mask,
            input_pos=input_pos,
            inputs=recv_futures[micro_batch_idx],
            micro_batch_idx=micro_batch_idx,
            sampling_kwargs=sampling_kwargs,
            recv_kwargs=recv_kwargs,
            skip_recv=not world.is_first_stage,
        )
        recv_futures[micro_batch_idx] = recv_future
        if pbar is not None:
            pbar.update(1)

    # Cooldown
    next_tokens = []
    for micro_batch_idx in range(num_micro_batches):
        if world.is_first_stage:
            micro_batch_tokens = recv_futures[micro_batch_idx].result()
            next_tokens.append(micro_batch_tokens)

    torch.cuda.synchronize()
    prefill_time = perf_counter() - start_prefill
    logger.debug(f"Prefill took {prefill_time:.02f}s")

    return next_tokens if world.is_first_stage else None, prefill_time


def decode(
    model: nn.Module,
    prefill_tokens: List[Tensor],
    num_prompt_tokens: int,
    num_new_tokens: int,
    num_micro_batches: int,
    micro_batch_size: int,
    pbar: Optional[tqdm] = None,
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
        pbar: Optional progress bar to update
        **sampling_kwargs: Dict of kwargs for the sample function

    Returns:
        metrics: Dict of metrics
    """
    world, logger, comm, device = get_world(), get_logger(), get_comm(), get_device()
    logger.debug("Calling decode")
    start_decode = perf_counter()

    starting_pos = num_prompt_tokens + 1

    # Create block mask
    B, H, Q_LEN, KV_LEN = 1, 1, model.max_seq_length, model.max_seq_length
    block_mask = create_block_mask(causal_mask, B, H, Q_LEN, KV_LEN, device=device)

    # Setup receive kwargs
    bwd_shape = (micro_batch_size, 1)
    fwd_shape = (micro_batch_size, 1, model.config.dim)
    bwd_dtype = torch.long
    fwd_dtype = model.layers[0].feed_forward.w1.weight.dtype
    shape = bwd_shape if world.is_first_stage else fwd_shape
    dtype = bwd_dtype if world.is_first_stage else fwd_dtype
    recv_kwargs = {"shape": shape, "dtype": dtype}

    # Prepare futures
    token_idxs = range(starting_pos, starting_pos + num_new_tokens - 1)

    # Warmup pipeline
    send_futures, recv_futures = [None] * num_micro_batches, [None] * num_micro_batches
    for micro_batch_idx in range(num_micro_batches):
        if world.is_first_stage:
            recv_futures[micro_batch_idx] = fake_future(prefill_tokens[micro_batch_idx].clone())
        else:
            recv_futures[micro_batch_idx] = comm.irecv(tag=micro_batch_idx, **recv_kwargs)

    # Interleaved decode pipeline schedule
    all_tokens = [[None for _ in range(len(token_idxs))] for _ in range(num_micro_batches)]
    for i, token_idx in enumerate(token_idxs):
        start_decode_step = perf_counter()
        logger.debug(f"Starting decode step {i + 1} for {token_idx=}")
        input_pos = torch.tensor([token_idx], device=device, dtype=torch.long)
        mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
        for micro_batch_idx in range(num_micro_batches):
            send_future, recv_future, micro_batch_tokens = micro_step(
                model=model,
                mask=mask,
                input_pos=input_pos,
                inputs=recv_futures[micro_batch_idx],
                micro_batch_idx=micro_batch_idx,
                sampling_kwargs=sampling_kwargs,
                recv_kwargs=recv_kwargs,
                skip_recv=token_idx == token_idxs[-1] and not world.is_first_stage,
            )
            send_futures[micro_batch_idx] = send_future
            recv_futures[micro_batch_idx] = recv_future
            if i == 0:
                continue
            all_tokens[micro_batch_idx][i - 1] = micro_batch_tokens
        torch.cuda.synchronize()
        logger.debug(f"Decode step {i + 1} for {token_idx=} took {(perf_counter() - start_decode_step) * 1000:.02f}ms")
        if pbar is not None:
            pbar.update(1)

    # Cooldown
    for micro_batch_idx in range(num_micro_batches):
        if world.is_first_stage:
            final_token = recv_futures[micro_batch_idx].result()
            all_tokens[micro_batch_idx][len(token_idxs) - 1] = final_token
        else:
            send_futures[micro_batch_idx].result()

    # Concatenate tokens
    if world.is_first_stage:
        decoded_tokens = [torch.cat(tokens, dim=-1) for tokens in all_tokens]
    else:
        decoded_tokens = None

    decode_time = perf_counter() - start_decode
    logger.debug(f"Decode took {decode_time:.02f}s")

    return decoded_tokens, decode_time


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_tokens: Tensor,
    num_prompt_tokens: int,
    num_new_tokens: int,
    num_micro_batches: int,
    micro_batch_size: int,
    disable_tqdm: bool = False,
    **sampling_kwargs,
) -> Tuple[Optional[Tensor], float, float]:
    """
    Generate tokens.

    Args:
        model: Transformer
        prompt_tokens: Tensor of shape [batch_size, num_prompt_tokens]
        num_new_tokens: int
        micro_batch_size: int
        **sampling_kwargs: Dict of kwargs for the sample function
    """
    world = get_world()
    # Create progress bar
    pbar = None if disable_tqdm else tqdm(total=num_new_tokens, desc="Generating")

    # Prefill prompt tokens
    prefill_tokens, prefill_time = prefill(
        model=model,
        prompt_tokens=prompt_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_micro_batches=num_micro_batches,
        micro_batch_size=micro_batch_size,
        pbar=pbar,
        **sampling_kwargs,
    )

    # Decode remaining tokens
    decoded_tokens, decode_time = decode(
        model=model,
        prefill_tokens=prefill_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_new_tokens=num_new_tokens,
        num_micro_batches=num_micro_batches,
        micro_batch_size=micro_batch_size,
        pbar=pbar,
        **sampling_kwargs,
    )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Concatenate prompt tokens and decoded tokens
    if world.is_first_stage:
        decoded_tokens = torch.cat([torch.cat(prompt_tokens), torch.cat(prefill_tokens), torch.cat(decoded_tokens)], dim=-1)
    else:
        decoded_tokens = None

    return decoded_tokens, prefill_time, decode_time


@torch.no_grad()
def forward(model: nn.Module, micro_batch_idx: int, mask: BlockMask, input_pos: Tensor, inputs: Tensor) -> Tensor:
    """Forward pass for the model for torch.compile"""
    return model(micro_batch_idx, mask, input_pos, inputs)


def full_compile():
    """Compile block mask generation, adjustment and model forward pass"""
    global create_block_mask
    create_block_mask = torch.compile(create_block_mask)

    global adjust_mask
    adjust_mask = torch.compile(adjust_mask)

    global forward
    forward = torch.compile(forward)


def fake_prefill(model: nn.Module, num_prompt_tokens: int, num_micro_batches: int, micro_batch_size: int):
    """Fakes prefill to trigger torch.compile"""
    world, device, logger = get_world(), get_device(), get_logger()
    logger.debug("Fake prefill...")

    # Create fake inputs
    input_pos = torch.arange(num_prompt_tokens, device=device)
    if world.is_first_stage:
        fake_inputs = torch.randint(0, model.config.vocab_size, (micro_batch_size, num_prompt_tokens), device=device)
    else:
        fake_inputs = torch.randn(micro_batch_size, num_prompt_tokens, model.config.dim, device=device, dtype=torch.bfloat16)
    fake_inputs = fake_future(fake_inputs)

    # Create block mask
    B, H, Q_LEN, KV_LEN = 1, 1, input_pos.shape[0], model.max_seq_length
    block_mask = create_block_mask(causal_mask, B, H, Q_LEN, KV_LEN, device=device)

    # Run micro steps
    for micro_batch_idx in range(num_micro_batches):
        micro_step(
            model,
            block_mask,
            input_pos,
            fake_inputs,
            micro_batch_idx,
            sampling_kwargs={},
            recv_kwargs={},
            skip_send=True,
            skip_recv=True,
        )


def fake_decode(model: nn.Module, num_prompt_tokens: int, num_micro_batches: int, micro_batch_size: int):
    """Fakes decode to trigger torch.compile"""
    world, device, logger = get_world(), get_device(), get_logger()
    logger.debug("Fake decode...")

    # Create fake inputs
    input_pos = torch.arange(num_prompt_tokens, device=device)
    if world.is_first_stage:
        fake_inputs = fake_future(torch.randint(0, model.config.vocab_size, (micro_batch_size, 1), device=device, dtype=torch.long))
    else:
        fake_inputs = fake_future(torch.randn(micro_batch_size, 1, model.config.dim, dtype=torch.bfloat16, device=device))

    # Create block mask
    B, H, Q_LEN, KV_LEN = 1, 1, model.max_seq_length, model.max_seq_length
    block_mask = create_block_mask(causal_mask, B, H, Q_LEN, KV_LEN, device=device)

    # Decode three tokens using micro batching
    num_new_tokens = 3  # Set small for small compile time
    for token_idx in range(num_new_tokens):
        input_pos = torch.tensor([num_prompt_tokens + token_idx], device=device, dtype=torch.long)
        mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
        for micro_batch_idx in range(num_micro_batches):
            micro_step(
                model,
                mask,
                input_pos,
                fake_inputs,
                micro_batch_idx,
                sampling_kwargs={},
                recv_kwargs={},
                skip_send=True,
                skip_recv=True,
            )


@torch.no_grad()
def fake_generate(model: nn.Module, num_prompt_tokens: int, num_micro_batches: int, micro_batch_size: int):
    """Fakes prefill and decode to trigger torch.compile"""
    fake_prefill(model, num_prompt_tokens, num_micro_batches, micro_batch_size)
    fake_decode(model, num_prompt_tokens, num_micro_batches, micro_batch_size)
