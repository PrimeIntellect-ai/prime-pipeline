from concurrent.futures import Future
from time import perf_counter
from typing import Dict, Optional, Tuple

import torch
import torch._dynamo.config
import torch._inductor.config
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from tqdm import tqdm

from .comm import get_comm
from .logger import get_logger
from .utils import fake_future
from .world import get_world

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
    input_pos: Tensor,
    inputs: Tensor,
) -> Tensor:
    """Wrapper of forward pass of the model for torch.compile"""
    input_ids, hidden_states = None, None
    if inputs.shape[-1] > 1:
        hidden_states = inputs
    else:
        input_ids = inputs
    return model(micro_batch_idx, block_mask, input_pos, input_ids, hidden_states)


def save_result(decoded_tokens: Tensor, token_idx: int, micro_batch_idx: int, micro_batch_size: int, inputs: Tensor):
    start_idx = micro_batch_idx * micro_batch_size
    end_idx = start_idx + micro_batch_size
    decoded_tokens[start_idx:end_idx, token_idx] = inputs.squeeze()


def micro_step(
    inputs: Future,
    model: nn.Module,
    mask: BlockMask,
    input_pos: Tensor,
    micro_batch_idx: int,
    sampling_kwargs: Dict,
    recv_kwargs: Dict,
    skip_send: bool = False,
    skip_recv: bool = False,
) -> Tuple[Optional[Future], Optional[Future], Optional[Tensor]]:
    """
    Computes one micro-step in the decode pipeline. It gets a future of the next
    inputs - tokens for the first stage, hidden states for all other stages. The
    future may be created manually, e.g. when filling up the pipeline, or a
    represent a pending receive request. The function awaits the inputs
    (blocking) and then computes the forward pass. Optionally, on the last
    stage, it also samples a token.  The output - whether hidden state or token
    - is then asynchronously sent to the corresponding next stage and a new
    receive request is scheduled, unless the skip_recv flag is set. The function
    returns the send future, the receive future and possible result.
    """
    world, logger, comm = get_world(), get_logger(), get_comm()
    logger.debug(f"Micro step {micro_batch_idx=}")

    # Wait to receive next token or hidden states
    inputs = inputs.result()

    # Save results on rank 0
    result = None
    if world.is_first_stage:
        result = inputs

    # Forward pass + sample
    start_forward = perf_counter()
    outputs = model_forward(model, micro_batch_idx, mask, input_pos, inputs)
    if world.is_last_stage:
        outputs = sample(outputs, **sampling_kwargs)
    torch.cuda.synchronize()
    logger.debug(f"Forward {micro_batch_idx=} took {(perf_counter() - start_forward) * 1000:.02f}ms")

    # Send hidden states or next token
    send_future = None
    if not skip_send:
        send_future = comm.isend(outputs, tag=micro_batch_idx)

    # Schedule next recv
    recv_future = None
    if not skip_recv:
        if world.is_first_stage and world.is_last_stage:
            fut = Future()
            fut.set_result(outputs)
            recv_future = fut
        else:
            recv_future = comm.irecv(tag=micro_batch_idx, **recv_kwargs)

    return send_future, recv_future, result


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
    logger.debug("Calling prefill")

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

    bwd_prefill_shape = (micro_batch_size, 1)
    fwd_prefill_shape = (micro_batch_size, num_prompt_tokens, model.config.dim)
    bwd_dtype = decoded_tokens.dtype
    fwd_dtype = model.layers[0].feed_forward.w1.weight.dtype
    shape = bwd_prefill_shape if world.is_first_stage else fwd_prefill_shape
    dtype = bwd_dtype if world.is_first_stage else fwd_dtype
    recv_kwargs = {"shape": shape, "dtype": dtype}

    # Warmup
    recv_futures = []
    send_futures = []
    for micro_batch_idx in range(num_micro_batches):
        if world.is_first_stage:
            start_idx = micro_batch_idx * micro_batch_size
            end_idx = start_idx + micro_batch_size
            recv_futures.append(fake_future(decoded_tokens[start_idx:end_idx, :num_prompt_tokens].clone()))
        else:
            recv_futures.append(comm.irecv(tag=micro_batch_idx, shape=shape, dtype=dtype))

    for micro_batch_idx in range(num_micro_batches):
        send_future, recv_future, _ = micro_step(
            inputs=recv_futures[micro_batch_idx],
            model=model,
            mask=block_mask,
            input_pos=input_pos,
            micro_batch_idx=micro_batch_idx,
            sampling_kwargs=sampling_kwargs,
            recv_kwargs=recv_kwargs,
            skip_recv=True if not world.is_first_stage else False,
        )
        send_futures.append(send_future)
        recv_futures[micro_batch_idx] = recv_future

    # Cooldown
    for micro_batch_idx in range(num_micro_batches):
        if world.is_first_stage:
            logger.debug("Waiting for recv")
            final_token = recv_futures[micro_batch_idx].result()
            save_result(decoded_tokens, num_prompt_tokens, micro_batch_idx, micro_batch_size, final_token)
        else:
            logger.debug("Waiting for send")
            send_futures[micro_batch_idx].result()

    return {}


def decode(
    model: nn.Module, decoded_tokens: Tensor, num_prompt_tokens: int, micro_batch_size: int, use_tqdm: bool = False, **sampling_kwargs
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
    logger.debug("Calling decode")

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

    send_futures = [Future() for _ in range(num_micro_batches)]
    recv_futures = [Future() for _ in range(num_micro_batches)]
    token_idxs = range(starting_pos, decoded_tokens.size(-1))

    bwd_shape = (micro_batch_size, 1)
    fwd_shape = (micro_batch_size, 1, model.config.dim)
    bwd_dtype = decoded_tokens.dtype
    fwd_dtype = model.layers[0].feed_forward.w1.weight.dtype
    shape = bwd_shape if world.is_first_stage else fwd_shape
    dtype = bwd_dtype if world.is_first_stage else fwd_dtype
    recv_kwargs = {"shape": shape, "dtype": dtype}

    # Warmup
    start_decode = perf_counter()
    for micro_batch_idx in range(num_micro_batches):
        if world.is_first_stage:
            start_idx = micro_batch_idx * micro_batch_size
            end_idx = start_idx + micro_batch_size
            recv_futures[micro_batch_idx] = Future()
            recv_futures[micro_batch_idx].set_result(decoded_tokens[start_idx:end_idx, starting_pos - 1].unsqueeze(-1).clone())
        else:
            recv_futures[micro_batch_idx] = comm.irecv(tag=micro_batch_idx, **recv_kwargs)

    # Interleaved decode pipeline schedule
    for token_idx in tqdm(token_idxs):
        start_decode_step = perf_counter()
        input_pos = torch.tensor([token_idx], device=device, dtype=torch.long)
        mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
        for micro_batch_idx in range(num_micro_batches):
            send_future, recv_future, result = micro_step(
                inputs=recv_futures[micro_batch_idx],
                model=model,
                mask=mask,
                input_pos=input_pos,
                micro_batch_idx=micro_batch_idx,
                sampling_kwargs=sampling_kwargs,
                recv_kwargs=recv_kwargs,
                skip_recv=token_idx == token_idxs[-1] and not world.is_first_stage,
            )
            if world.is_first_stage:
                save_result(decoded_tokens, token_idx - 1, micro_batch_idx, micro_batch_size, result)
            send_futures[micro_batch_idx] = send_future
            recv_futures[micro_batch_idx] = recv_future
        torch.cuda.synchronize()
        logger.debug(f"Decode {token_idx=} took {(perf_counter() - start_decode_step) * 1000:.02f}ms")

    # Cooldown
    for micro_batch_idx in range(num_micro_batches):
        if world.is_first_stage:
            logger.debug("Waiting for recv")
            final_token = recv_futures[micro_batch_idx].result()
            save_result(decoded_tokens, token_idx, micro_batch_idx, micro_batch_size, final_token.squeeze())
        else:
            logger.debug("Waiting for send")
            send_futures[micro_batch_idx].result()

    decode_time = perf_counter() - start_decode
    logger.debug(f"Decode took {decode_time}s")

    return decoded_tokens, decode_time


def compile_generate():
    """Compile the model"""
    global create_block_mask
    create_block_mask = torch.compile(create_block_mask, fullgraph=True)

    global adjust_mask
    adjust_mask = torch.compile(adjust_mask, fullgraph=True)

    global model_forward
    model_forward = torch.compile(model_forward)


@torch.no_grad()
def generate(
    model: nn.Module,
    decoded_tokens: Tensor,
    num_prompt_tokens: int,
    micro_batch_size: int,
    use_tqdm: bool = False,
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
        use_tqdm=use_tqdm,
        **sampling_kwargs,
    )

    return decoded_tokens, prefill_metrics, decode_metrics
