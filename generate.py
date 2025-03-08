# From: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
import argparse
import sys
import time
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

# External imports
import torch
import torch._dynamo.config
import torch._inductor.config
import torch.distributed as dist
from loguru import logger
from lovely_tensors import monkey_patch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

# Local imports
from model import Transformer
from tokenizer import get_tokenizer
from utils import get_device, get_precision, load_model, seed_everything, shard_model

# Use lovely tensors
monkey_patch()

# Setup compile flags
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True

# Support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def multinomial_sample_one_no_sync(probs_sort: Tensor) -> Tensor:
    """Sample from a multinomial distribution without a cuda synchronization from (B, S) to (B, 1)"""
    # Draw exponential random variables
    q = torch.empty_like(probs_sort).exponential_(1)

    # Select the argmax
    next_tokens = torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(
        dtype=torch.long
    )

    return next_tokens


def logits_to_probs(
    logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None
) -> Tensor:
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


def sample(
    logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None
) -> Tensor:
    """Sample from a multinomial distribution without a cuda synchronization"""
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next


def maybe_sample(logits: Tensor, **sampling_kwargs) -> Optional[Tensor]:
    if dist.get_rank() == dist.get_world_size() - 1:
        return sample(logits, **sampling_kwargs)
    return None


class WaitIfNecessary:
    def __init__(self, req):
        self.req = req

    def wait(self):
        if hasattr(self.req, "wait"):
            self.req.wait()


def maybe_recv_fwd(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    tag: int,
    do_async: bool = True,
) -> Tuple[Optional[Tensor], WaitIfNecessary]:
    if dist.get_world_size() == 1 or dist.get_rank() == 0:
        return None, None

    # Initialize receive tensor
    recv_tensor = torch.empty(size=shape, dtype=dtype, device=device)

    # Receive tensor
    recv = dist.irecv if do_async else dist.recv
    logger.debug(
        f"Scheduled recv_fwd {tag=} {recv_tensor.shape=} {recv_tensor.dtype=} {recv_tensor.device=} {tag=}"
    )
    recv_req = recv(recv_tensor, src=dist.get_rank() - 1, tag=tag)

    return recv_tensor, WaitIfNecessary(recv_req)


def maybe_send_fwd(tensor: Tensor, tag: int) -> None:
    if dist.get_world_size() == 1 or dist.get_rank() == dist.get_world_size() - 1:
        return
    logger.debug(f"Call send_fwd {tensor=} {tag=}")
    dist.send(tensor, dst=dist.get_rank() + 1, tag=tag)


def maybe_recv_bwd(
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    tag: int,
    do_async: bool = True,
) -> Tuple[Optional[Tensor], WaitIfNecessary]:
    if dist.get_world_size() == 1 or dist.get_rank() != 0:  # only first stage
        return None, None
    recv_tensor = torch.empty(size=shape, dtype=dtype, device=device)
    recv = dist.irecv if do_async else dist.recv
    logger.debug(
        f"Scheduled recv_bwd {recv_tensor.shape=} {recv_tensor.dtype=} {recv_tensor.device=} {tag=}"
    )
    recv_req = recv(recv_tensor, src=dist.get_world_size() - 1, tag=tag)
    return recv_tensor, WaitIfNecessary(recv_req)


def maybe_send_bwd(tensor: Tensor, tag: int) -> None:
    if dist.get_world_size() == 1 or dist.get_rank() != dist.get_world_size() - 1:
        return
    logger.debug(f"Call send_bwd {tensor=} {tag=}")
    dist.send(tensor, dst=0, tag=tag)


def causal_mask(b, h, q, kv):
    """Causal mask using for flex attention"""
    return q >= kv


def adjust_mask(
    block_mask: BlockMask, input_pos: Tensor, max_seq_length: int
) -> BlockMask:
    """Adjust the mask for the given input position"""
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, max_seq_length)

    return mask


def model_forward(
    model: Transformer,
    micro_batch_idx: int,
    block_mask: BlockMask,
    prompt_tokens: Tensor,
    input_pos: Tensor,
    hidden_states: Optional[Tensor] = None,
) -> Tensor:
    """Wrapper of forward pass of the model for torch.compile"""
    return model(micro_batch_idx, block_mask, prompt_tokens, input_pos, hidden_states)


def adjust_mask_and_model_forward(
    model: Transformer,
    micro_batch_idx: int,
    block_mask: BlockMask,
    prompt_tokens: Tensor,
    input_pos: Tensor,
    hidden_states: Optional[Tensor] = None,
) -> Tensor:
    """Wrapper for adjust mask and forward pass of the model for torch.compile"""
    mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
    return model_forward(
        model, micro_batch_idx, mask, prompt_tokens, input_pos, hidden_states
    )


def prefill(
    model: Transformer,
    batched_prompt_tokens: List[Tensor],
    input_pos: Tensor,
    **sampling_kwargs,
) -> Tensor:
    """
    Prefill prompt tokens to start inference. Assumes that the model caches are
    set up, the prompt is tokenized and split into micro-batches and the
    input_pos is a tensor of shape [num_prompt_tokens]. Automatically handles
    pipelining using synchronous communication primitives.

    Args:
        model: Transformer
        batched_prompt_tokens: List[Tensor], each of shape [micro_batch_size, num_prompt_tokens]
        input_pos: Tensor of shape [num_prompt_tokens]
        **sampling_kwargs: Dict of kwargs for the sample function

    Returns:
        next_token: Tensor of shape [batch_size, 1]
    """
    logger.debug(
        f"Call prefill(model, {len(batched_prompt_tokens)=}, {batched_prompt_tokens[0]=}, {input_pos=})"
    )

    # Setup shapes, dtypes, device
    micro_batch_size = batched_prompt_tokens[0].shape[0]
    tokens_shape = (micro_batch_size, 1)
    tokens_dtype = torch.long  # TODO: dont hardcode
    hidden_states_shape = (micro_batch_size, batched_prompt_tokens[0].size(1), 4096)
    hidden_states_dtype = model.layers[
        0
    ].feed_forward.w1.weight.dtype  # TODO: Cleaner way
    device = batched_prompt_tokens[0].device

    # Partially apply recv functions
    rank_maybe_recv_fwd = partial(
        maybe_recv_fwd,
        shape=hidden_states_shape,
        dtype=hidden_states_dtype,
        device=device,
        do_async=False,
    )
    rank_maybe_recv_bwd = partial(
        maybe_recv_bwd,
        shape=tokens_shape,
        dtype=tokens_dtype,
        device=device,
        do_async=False,
    )

    # Create block mask
    block_mask = create_block_mask(
        causal_mask,
        1,
        1,
        input_pos.shape[0],
        model.max_seq_length,
        device=batched_prompt_tokens[0].device,
    )

    batched_next_tokens: List[Tensor] = []  # List[Tensor[micro_batch_size, 1]]
    for micro_batch_idx, prompt_tokens in enumerate(batched_prompt_tokens):
        hidden_states, _ = rank_maybe_recv_fwd(tag=micro_batch_idx)
        outputs = model_forward(
            model,
            micro_batch_idx,
            block_mask,
            prompt_tokens,
            input_pos,
            hidden_states,
        )
        maybe_send_fwd(outputs, tag=micro_batch_idx)
        next_token = maybe_sample(outputs, **sampling_kwargs)
        if next_token is None:
            next_token, _ = rank_maybe_recv_bwd(tag=micro_batch_idx)
        assert next_token is not None, "Should have sampled or received"
        maybe_send_bwd(next_token, tag=micro_batch_idx)
        batched_next_tokens.append(next_token)

    return torch.cat(batched_next_tokens, dim=0)  # [batch_size, 1]


def decode_n_tokens(
    model: Transformer,
    batched_cur_tokens: List[Tensor],
    input_pos: Tensor,
    num_new_tokens: int,
    **sampling_kwargs,
) -> Tensor:
    """
    Auto-regressively decodes tokens until max new tokens are reached. Assumes that
    the model caches are set up, the prompt is tokenized and split into micro-batches
    and the input_pos is a tensor of shape [num_prompt_tokens]. Automatically handles
    interleaved pipelining to achieve near peak device utilization.

    Args:
        model: Transformer
        batched_cur_tokens: List[Tensor], each of shape [micro_batch_size, 1]
        input_pos: Tensor of shape [1]
        num_new_tokens: int
        **sampling_kwargs: Dict of kwargs for the sample function

    Returns:
        next_tokens: Tensor of shape [batch_size, num_new_tokens]
    """
    logger.debug(
        f"Calling decode_n_tokens(model, {len(batched_cur_tokens)=}, {batched_cur_tokens[0]=}, {input_pos=}, {num_new_tokens=})"
    )

    # Setup shapes, dtypes, device
    num_micro_batches = len(batched_cur_tokens)
    micro_batch_size = batched_cur_tokens[0].shape[0]
    tokens_shape = (micro_batch_size, 1)
    tokens_dtype = torch.long  # TODO: dont hardcode
    hidden_states_shape = (micro_batch_size, 1, model.config.dim)
    hidden_states_dtype = model.layers[
        0
    ].feed_forward.w1.weight.dtype  # TODO: Cleaner way
    device = batched_cur_tokens[0].device

    # Setup receive and send functions
    recv_func = maybe_recv_bwd if dist.get_rank() == 0 else maybe_recv_fwd
    do_async = True if dist.get_rank() == 0 else False
    shape = tokens_shape if dist.get_rank() == 0 else hidden_states_shape
    dtype = tokens_dtype if dist.get_rank() == 0 else hidden_states_dtype
    recv = partial(
        recv_func,
        shape=shape,
        dtype=dtype,
        device=device,
        do_async=do_async,
    )
    send = (
        maybe_send_bwd
        if dist.get_rank() == dist.get_world_size() - 1
        else maybe_send_fwd
    )

    # Create block mask
    block_mask = create_block_mask(
        causal_mask,
        1,
        1,
        model.max_seq_length,
        model.max_seq_length,
        device=device,
    )

    # Setup receive buffers
    recv_reqs = [None] * num_micro_batches
    recv_buffers = [None] * num_micro_batches

    # Prefill pipeline on first stage
    if dist.get_rank() == 0 and dist.get_world_size() > 1:
        for micro_batch_idx in range(num_micro_batches):
            mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
            hidden_states = model_forward(
                model,
                micro_batch_idx,
                mask,
                batched_cur_tokens[micro_batch_idx],
                input_pos,
            )
            send(hidden_states, tag=micro_batch_idx)
            recv_tensor, recv_req = recv(tag=micro_batch_idx)
            recv_reqs[micro_batch_idx] = recv_req
            recv_buffers[micro_batch_idx] = recv_tensor
        input_pos += 1

    # Decode auto-regressively
    batched_decoded_tokens = []  # List[Tensor[batch_size, 1]] of length num_new_tokens
    for token_idx in range(num_new_tokens):
        batched_next_tokens = []
        mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
        for micro_batch_idx in range(num_micro_batches):
            # Wait for requests
            if dist.get_world_size() == 1:
                logits = model_forward(
                    model,
                    micro_batch_idx,
                    mask,
                    batched_cur_tokens[micro_batch_idx],
                    input_pos,
                )
                next_token = sample(logits, **sampling_kwargs)
                batched_next_tokens.append(next_token)
                batched_cur_tokens[micro_batch_idx] = next_token
            else:
                if dist.get_rank() == 0:
                    recv_reqs[micro_batch_idx].wait()
                    next_token = recv_buffers[micro_batch_idx].clone()
                    batched_cur_tokens[micro_batch_idx] = next_token
                    batched_next_tokens.append(next_token)

                    if token_idx < num_new_tokens - 1:
                        hidden_states = model_forward(
                            model,
                            micro_batch_idx,
                            mask,
                            batched_cur_tokens[micro_batch_idx],
                            input_pos,
                        )
                        send(hidden_states, tag=micro_batch_idx)

                        recv_tensor, recv_req = recv(tag=micro_batch_idx)
                        recv_reqs[micro_batch_idx] = recv_req
                        recv_buffers[micro_batch_idx] = recv_tensor
                elif dist.get_rank() == dist.get_world_size() - 1:
                    hidden_states, _ = recv(tag=micro_batch_idx)
                    logits = model_forward(
                        model,
                        micro_batch_idx,
                        mask,
                        batched_cur_tokens[micro_batch_idx],
                        input_pos,
                        hidden_states,
                    )
                    next_token = maybe_sample(logits, **sampling_kwargs)
                    batched_cur_tokens[micro_batch_idx] = next_token
                    batched_next_tokens.append(next_token)
                    send(next_token, tag=micro_batch_idx)
                else:
                    raise NotImplementedError

        batched_decoded_tokens.append(torch.cat(batched_next_tokens, dim=0))
        input_pos += 1

    return torch.cat(batched_decoded_tokens, dim=-1)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: Tensor,
    max_new_tokens: int,
    **sampling_kwargs,
) -> Tensor:
    """
    Generate tokens from a (batched) encoded prompt

    Args:
        model: Transformer
        prompt_tokens: Tensor of shape [batch_size, num_prompt_tokens]
        max_new_tokens: int
        **sampling_kwargs: Dict of kwargs for the sample function
    """
    # Setup sequence length and microbatching
    batch_size, num_prompt_tokens = prompt_tokens.shape
    num_micro_batches = (
        dist.get_world_size() if batch_size >= dist.get_world_size() else 1
    )
    seq_length = min(num_prompt_tokens + max_new_tokens, model.config.block_size)
    micro_batch_size = batch_size // num_micro_batches

    # Setup model cache
    device, dtype = prompt_tokens.device, prompt_tokens.dtype
    with torch.device(device):
        model.setup_caches(
            num_micro_batches=num_micro_batches,
            max_micro_batch_size=micro_batch_size,
            max_seq_length=seq_length,
        )

    # Setup input positions
    batched_prompt_tokens = list(torch.split(prompt_tokens, micro_batch_size, dim=0))
    input_pos = torch.arange(num_prompt_tokens, device=device)  # [num_prompt_tokens]

    # Allocate tensor for all decoded tokens
    decoded_tokens = torch.empty(batch_size, seq_length, dtype=dtype, device=device)
    decoded_tokens[:, :num_prompt_tokens] = prompt_tokens

    # Return early if no new tokens are generated
    if max_new_tokens == 0:
        return decoded_tokens

    # Prefill prompt tokens
    next_token = prefill(
        model=model,
        batched_prompt_tokens=batched_prompt_tokens,
        input_pos=input_pos,
        **sampling_kwargs,
    )
    logger.debug(f"Prefilled tokens {next_token=}")
    decoded_tokens[:, num_prompt_tokens] = next_token.squeeze()

    # Decode remaining tokens
    batched_cur_tokens = list(torch.split(next_token, micro_batch_size, dim=0))
    input_pos = torch.tensor([num_prompt_tokens], device=device, dtype=torch.long)
    num_new_tokens = max_new_tokens - 1
    if num_new_tokens == 0:
        return decoded_tokens
    next_tokens = decode_n_tokens(
        model=model,
        batched_cur_tokens=batched_cur_tokens,
        input_pos=input_pos,
        num_new_tokens=max_new_tokens - 1,
        **sampling_kwargs,
    )
    logger.debug(f"Decoded tokens {next_tokens=}")
    decoded_tokens[:, num_prompt_tokens + 1 :] = next_tokens

    return decoded_tokens


def main(args: argparse.Namespace) -> None:
    # Setup process group
    dist.init_process_group()

    # Initialize logger
    global logger
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> [<level>{level}</level>] [<level>Rank {extra[rank]}</level>] - <level>{message}</level>"
    logger.remove()
    logger.add(sys.stdout, level=args.log_level, format=console_format)
    logger = logger.bind(rank=dist.get_rank())
    logger.info("Starting")
    logger.info(f"Config: {vars(args)}")

    # Set seeds for reproducibility across all processes
    seed_everything(args.seed)

    # Set device and precision
    device = get_device(args.device)
    precision = get_precision(args.precision)

    # Load model
    t0 = time.time()
    checkpoint_path = Path(f"checkpoints/{args.model_repo}/model.pth")
    model = load_model(checkpoint_path, device, precision)
    torch.cuda.synchronize(device)
    logger.info(f"Loaded model in {time.time() - t0:.02f} seconds")

    # Shard model
    if dist.get_world_size() > 1:
        t0 = time.time()
        model = shard_model(model, dist.get_rank(), dist.get_world_size())
        torch.cuda.synchronize(device)

    # Load tokenizer
    t0 = time.time()
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    if args.compile:
        global create_block_mask
        create_block_mask = torch.compile(create_block_mask, fullgraph=True)

        global adjust_mask
        adjust_mask = torch.compile(adjust_mask, fullgraph=True)

        global model_forward
        model_forward = torch.compile(model_forward, fullgraph=True)

    # Encode prompt (batch_size, num_prompt_tokens)
    prompt_tokens = [tokenizer.bos_id()] + tokenizer.encode(args.prompt)
    prompt_tokens = torch.tensor(
        prompt_tokens,
        device=device,
    ).repeat(args.batch_size, 1)

    start = -1 if args.compile else 0
    metrics = {"latency": [], "tps": []}
    for i in range(start, args.num_samples):
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        decoded_tokens = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        if i == -1:
            logger.info(
                f"Compiled model in {time.perf_counter() - start_time:.2f} seconds"
            )
            continue

        # Calculate metrics
        torch.cuda.synchronize(device)
        time_taken = time.perf_counter() - start_time
        num_prompt_tokens = args.batch_size * prompt_tokens.size(-1)
        num_generated_tokens = args.batch_size * (
            decoded_tokens.size(-1) - prompt_tokens.size(-1)
        )
        num_total_tokens = num_prompt_tokens + num_generated_tokens
        metrics["latency"].append(time_taken)
        metrics["tps"].append(num_generated_tokens / time_taken)

        # Print generations (on main rank)
        if dist.get_rank() == 0:
            logger.info(
                f"First generation: {tokenizer.decode(decoded_tokens[0].tolist())}"
            )

    # Print metrics (on main rank)
    if dist.get_rank() == 0:
        logger.info(f"Batch Size: {args.batch_size}")
        logger.info(f"Number of prompt tokens: {num_prompt_tokens}")
        logger.info(f"Number of generated tokens: {num_generated_tokens}")
        logger.info(f"Number of total tokens: {num_total_tokens}")
        logger.info(
            f"Latency: {torch.mean(torch.tensor(metrics['latency'])).item():.2f} seconds"
        )
        logger.info(
            f"Throughput: {torch.mean(torch.tensor(metrics['tps'])).item():.2f} tokens/second"
        )
        logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_repo",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model repository.",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=50, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Whether to compile the model.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for the model."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        help="Precision to use for the model.",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Seed for reproducibility."
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Log level.")

    main(parser.parse_args())
