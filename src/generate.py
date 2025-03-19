# From: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
import pickle
import argparse
from time import perf_counter
import time
import asyncio
from pathlib import Path
from typing import Any, Optional

# External imports
import torch
import torch._dynamo.config
import torch._inductor.config
from lovely_tensors import monkey_patch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from comm import TorchP2PComm, IrohP2PComm, P2PComm, get_comm, setup_comm
from logger import get_logger, setup_logger
from model import Transformer
from serializer import PickleSerializer
from tokenizer import get_tokenizer
from utils import get_device, get_precision, load_model, seed_everything, shard_model
from world import World, get_world, setup_world

# Use lovely tensors
monkey_patch()

# Setup compile flags
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True

# Globals
logger: Optional[Any] = None
comm: Optional[P2PComm] = None
world: Optional[World] = None


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
    prompt_tokens: Optional[Tensor],
    input_pos: Tensor,
    hidden_states: Optional[Tensor] = None,
) -> Tensor:
    """Wrapper of forward pass of the model for torch.compile"""
    return model(micro_batch_idx, block_mask, input_pos, prompt_tokens, hidden_states)


def adjust_mask_and_model_forward(
    model: Transformer,
    micro_batch_idx: int,
    block_mask: BlockMask,
    input_pos: Tensor,
    prompt_tokens: Tensor,
    hidden_states: Optional[Tensor] = None,
) -> Tensor:
    """Wrapper for adjust mask and forward pass of the model for torch.compile"""
    mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
    return model_forward(
        model, micro_batch_idx, mask, input_pos, prompt_tokens, hidden_states
    )


async def wrap_future(future):
    """Wrapper to convert Future to coroutine"""
    return await future


async def prefill(
    model: Transformer,
    decoded_tokens: Tensor,
    micro_batch_size: int,
    num_prompt_tokens: int,
    **sampling_kwargs,
) -> None:
    """
    Prefill prompt tokens to start inference. Assumes that the model caches are
    set up, the prompt is tokenized and put into a tensor of shape [batch_size, seq_length].
    Automatically handles pipelining using synchronous communication primitives.

    Args:
        model: Transformer
        decoded_tokens: Tensor of shape [batch_size, seq_length]
        input_pos: Tensor of shape [num_prompt_tokens]
        micro_batch_size: int
        **sampling_kwargs: Dict of kwargs for the sample function

    Returns:
        next_token: Tensor of shape [batch_size, 1]
    """
    logger.debug(
        f"prefill(model, {decoded_tokens=}, {num_prompt_tokens=}, {micro_batch_size=})"
    )

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

    device = decoded_tokens.device
    batch_size = decoded_tokens.size(0)
    num_prompt_tokens = input_pos.size(0)
    num_micro_batches = batch_size // micro_batch_size
    next_token, hidden_states = None, None
    send_tasks = []
    for micro_batch_idx in range(num_micro_batches):
        # Receive hidden states from previous stage
        if not world.is_first_stage:
            hidden_states = pickle.loads(await comm.recv()).to(device)

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
        t0 = perf_counter()
        bytes = pickle.dumps(outputs.cpu())
        logger.debug(f"Serialization took {(perf_counter() - t0) * 1000:.2f}ms")
        t0 = perf_counter()
        send_tasks.append(asyncio.create_task(wrap_future(comm.send(bytes))))
        logger.debug(f"Send took {(perf_counter() - t0) * 1000:.2f}ms")

        if world.is_first_stage:
            # Receive next token from last stage
            if world.size == 1:
                next_token = outputs
            else:
                next_token = pickle.loads(await wrap_future(comm.recv())).to(device)
            decoded_tokens[start_idx:end_idx, num_prompt_tokens] = next_token.squeeze()
    await asyncio.gather(*send_tasks)

    logger.debug(f"Prefilled tokens {decoded_tokens[:, num_prompt_tokens]=}")


async def decode(
    model: Transformer,
    decoded_tokens: Tensor,
    num_prompt_tokens: int,
    micro_batch_size: int,
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
        f"decode(model, {decoded_tokens=}, {num_prompt_tokens=}, {micro_batch_size=})"
    )

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
        for token_idx in range(starting_pos, decoded_tokens.size(-1)):
            input_pos = torch.tensor([token_idx], device=device, dtype=torch.long)
            mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
            for micro_batch_idx in range(num_micro_batches):
                start_idx = micro_batch_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                cur_tokens = decoded_tokens[start_idx:end_idx, token_idx - 1].unsqueeze(
                    -1
                )
                logits = model_forward(
                    model,
                    micro_batch_idx,
                    mask,
                    cur_tokens,
                    input_pos
                )
                next_token = sample(logits, **sampling_kwargs)
                decoded_tokens[start_idx:end_idx, token_idx] = next_token.squeeze()
        logger.debug(f"Decoded tokens {decoded_tokens[:, num_prompt_tokens:]=}")
        return

    # Multi-node
    # Prefill pipeline
    recv_reqs = [None] * num_micro_batches
    serialize_times = []
    recv_times = []
    send_times = []
    forward_times = []
    if world.is_first_stage:
        send_tasks = []
        input_pos = torch.tensor([starting_pos], device=device, dtype=torch.long)
        mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
        for micro_batch_idx in range(num_micro_batches):
            start_idx = micro_batch_idx * micro_batch_size
            end_idx = start_idx + micro_batch_size
            cur_tokens = decoded_tokens[start_idx:end_idx, starting_pos - 1].unsqueeze(
                -1
            )
            hidden_states = model_forward(
                model,
                micro_batch_idx,
                mask,
                cur_tokens,
                input_pos,
            )
            t0 = perf_counter()
            bytes = pickle.dumps(hidden_states.cpu())
            serialize_time = (perf_counter() - t0) * 1000
            serialize_times.append(serialize_time) 
            logger.debug(f"Serialization took {serialize_time:.2f}ms")
            t0 = perf_counter() 
            send_tasks.append(asyncio.create_task(wrap_future(comm.send(bytes))))
            send_time = (perf_counter() - t0) * 1000
            send_times.append(send_time)
            logger.debug(f"Send took {send_time:.2f}ms")
            recv_start = perf_counter()
            recv_reqs[micro_batch_idx] = comm.recv()

        await asyncio.gather(*send_tasks)
        
    # Decode interleaved
    next_token, hidden_states = None, None
    for token_idx in range(starting_pos, decoded_tokens.size(-1)):
        input_pos = torch.tensor([token_idx], device=device, dtype=torch.long)
        mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
        send_tasks = []
        for micro_batch_idx in range(num_micro_batches):
            if world.is_first_stage:
                start_idx = micro_batch_idx * micro_batch_size
                end_idx = start_idx + micro_batch_size
                next_token = pickle.loads(await recv_reqs[micro_batch_idx]).to(device)
                recv_time = (perf_counter() - recv_start) * 1000
                logger.debug(f"Recv took {recv_time:.2f}ms")
                recv_times.append(recv_time)
                decoded_tokens[start_idx:end_idx, token_idx] = next_token.squeeze()
                input_pos += 1
            else:
                recv_start = perf_counter()
                hidden_states = pickle.loads(await wrap_future(comm.recv())).to(device)
                recv_time = (perf_counter() - recv_start) * 1000
                logger.debug(f"Recv took {recv_time:.2f}ms")
                recv_times.append(recv_time)

            # Skip forward and send on last iteration for first stage
            if world.is_first_stage and token_idx >= decoded_tokens.size(-1) - 1:
                continue

            # Forward pass
            t0 = perf_counter()
            outputs = model_forward(
                model,
                micro_batch_idx,
                mask,
                next_token,
                input_pos,
                hidden_states,
            )
            torch.cuda.synchronize()
            forward_time = (perf_counter() - t0) * 1000
            logger.debug(f"Forward pass took {forward_time:.2f}ms")
            forward_times.append(forward_time)

            if world.is_last_stage:
                outputs = sample(outputs, **sampling_kwargs)

            # Send hidden states or next token
            t0 = perf_counter()
            bytes = pickle.dumps(outputs.cpu())
            serialize_time = (perf_counter() - t0) * 1000
            serialize_times.append(serialize_time)
            logger.debug(f"Serialization took {serialize_time:.2f}ms")
            t0 = perf_counter()
            send_tasks.append(asyncio.create_task(wrap_future(comm.send(bytes))))
            send_time = (perf_counter() - t0) * 1000
            send_times.append(send_time)
            logger.debug(f"Send took {send_time:.2f}ms")
            # Schedule next recv
            if world.is_first_stage:
                recv_start = perf_counter()
                recv_reqs[micro_batch_idx] = comm.recv()

        await asyncio.gather(*send_tasks)


    stats = lambda x: f"{x.mean().item():.1f}ms (std: {x.std().item():.2f}ms)"
    logger.debug("")
    logger.debug(f"Serialize times: {stats(torch.tensor(serialize_times))}")
    logger.debug(f"Recv times: {stats(torch.tensor(recv_times))}")
    logger.debug(f"Send times: {stats(torch.tensor(send_times))}")
    logger.debug(f"Forward times: {stats(torch.tensor(forward_times))}")

@torch.no_grad()
def generate(
    model: Transformer,
    prompt_tokens: Tensor,
    num_new_tokens: int,
    micro_batch_size: int,
    **sampling_kwargs,
) -> Tensor:
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
    batch_size, num_prompt_tokens = prompt_tokens.shape
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
    decoded_tokens = torch.empty(
        batch_size, num_total_tokens, dtype=prompt_tokens.dtype, device=device
    )
    decoded_tokens[:, :num_prompt_tokens] = prompt_tokens

    # Prefill prompt tokens
    asyncio.run(prefill(
        model=model,
        decoded_tokens=decoded_tokens,
        micro_batch_size=micro_batch_size,
        num_prompt_tokens=num_prompt_tokens,
        **sampling_kwargs,
    ))

    # Decode remaining tokens
    asyncio.run(decode(
        model=model,
        decoded_tokens=decoded_tokens,
        micro_batch_size=micro_batch_size,
        num_prompt_tokens=num_prompt_tokens,
        **sampling_kwargs,
    ))

    return decoded_tokens


def main(args: argparse.Namespace) -> None:
    # Setup world and communication
    setup_world()
    global world
    world = get_world()

    # Initialize logger
    setup_logger(world.rank, log_level=args.log_level)
    global logger
    logger = get_logger()
    logger.info("Starting")
    torch.cuda.synchronize()
    logger.info(f"Args: {vars(args)}")

    # Set seeds for reproducibility across all processes
    seed_everything(args.seed)

    # Set device and precision
    device = get_device(args.device, world)
    logger.info(f"Device: {device}")
    precision = get_precision(args.precision)

    # Load model
    t0 = time.time()
    checkpoint_path = Path(f"checkpoints/{args.model_repo}/model.pth")
    model = load_model(checkpoint_path, device, precision)
    logger.info(f"Loaded model in {time.time() - t0:.02f} seconds")

    # Shard model
    if world.size > 1:
        t0 = time.time()
        model = shard_model(model, world.rank, world.size)
        logger.info(f"Sharded model in {time.time() - t0:.02f} seconds")
        logger.info(f"Model: {model}")

    # Compile model
    if args.compile:
        global create_block_mask
        create_block_mask = torch.compile(create_block_mask, fullgraph=True)

        global adjust_mask
        adjust_mask = torch.compile(adjust_mask, fullgraph=True)

        global model_forward
        model_forward = torch.compile(model_forward, fullgraph=True)

    # Load tokenizer
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    # Encode prompt (batch_size, num_prompt_tokens)
    prompt_tokens = [tokenizer.bos_id()] + tokenizer.encode(args.prompt)
    prompt_tokens = torch.tensor(
        prompt_tokens,
        device=device,
    ).repeat(args.batch_size, 1)

    # Setup communication
    micro_batch_size = (
        args.micro_batch_size
        if args.micro_batch_size is not None
        else args.batch_size // world.size
        if args.batch_size >= world.size
        else 1
    )
    logger.info(f"Micro-batch size: {micro_batch_size}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Num micro-batches: {args.batch_size // micro_batch_size}")
    num_prompt_tokens = prompt_tokens.size(-1)
    # hidden_states_shape = (micro_batch_size, 1, model.config.dim)
    # tokens_shape = (micro_batch_size, 1)
    # hidden_states_dtype = model.layers[0].feed_forward.w1.weight.dtype
    # tokens_dtype = torch.long
    # if args.backend == "torch":
    #     setup_comm(TorchP2PComm, 
    #             fwd_shape=hidden_states_shape,
    #             bwd_shape=tokens_shape,
    #             fwd_dtype=hidden_states_dtype,
    #             bwd_dtype=tokens_dtype,
    #             device=device,
    #             num_prompt_tokens=num_prompt_tokens,
    #     )
    # elif args.backend == "iroh":
    #     setup_comm(IrohP2PComm, serializer=PickleSerializer(), device=device)
    # else:
    #     raise ValueError(f"Invalid backend: {args.backend}")

    global comm
    from iroh_py import create_node
    comm = create_node()
    node_id = comm.get_node_id()
    logger.info(f"Connect to: {node_id}")
    peer_id = input().strip()
    time.sleep(1)
    comm.connect(peer_id)
    while not comm.is_ready():
        time.sleep(0.1)
    logger.info(f"Connected to: {peer_id}")

    start = -1 if args.compile else 0
    metrics = {"latency": [], "tps": []}
    for i in range(start, args.num_samples):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        decoded_tokens = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            num_new_tokens=args.num_new_tokens,
            micro_batch_size=micro_batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        if i == -1:
            logger.info(f"Compiled in {time.perf_counter() - start_time:.2f} seconds")
            continue

        # Calculate metrics
        torch.cuda.synchronize()
        time_taken = time.perf_counter() - start_time
        num_prompt_tokens = args.batch_size * prompt_tokens.size(-1)
        num_generated_tokens = args.batch_size * (
            decoded_tokens.size(-1) - prompt_tokens.size(-1)
        )
        num_total_tokens = num_prompt_tokens + num_generated_tokens
        metrics["latency"].append(time_taken)
        metrics["tps"].append(num_generated_tokens / time_taken)

        # Print generations (on main rank)
        if world.rank == 0:
            logger.info(
                f"First generation: {tokenizer.decode(decoded_tokens[0].tolist())}"
            )

    # Print metrics (on main rank)
    if world.rank == 0:
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

    # Destroy communication
    comm.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-repo",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model repository.",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is", help="Input prompt."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Number of micro-batches. If not provided, will use world.size if batch_size >= world.size else 1.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1, help="Number of samples to generate."
    )
    parser.add_argument(
        "--num-new-tokens",
        type=int,
        default=50,
        help="Number of new tokens to generate.",
    )
    parser.add_argument("--top-k", type=int, default=200, help="Top-k for sampling.")
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
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        help="Either `torch` or `iroh`.",
    )
    main(parser.parse_args())
