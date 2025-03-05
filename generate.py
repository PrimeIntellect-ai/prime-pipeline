# From: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import loguru
from torch import Tensor
import torch.distributed as dist
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer
from utils import seed_everything, setup_world, setup_logger, load_model

def multinomial_sample_one_no_sync(probs_sort: Tensor) -> Tensor:
    """Sample from a multinomial distribution without a cuda synchronization from (B, S) to (B, 1)"""
    # Draw exponential random variables
    q = torch.empty_like(probs_sort).exponential_(1)
    # Select the argmax
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

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

def sample(logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    """Sample from a multinomial distribution without a cuda synchronization"""
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def roundup(val: float, multiplier: float) -> int:
    """Round up to the nearest multiple of multiplier"""
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv

def pipelined_forward(model: Transformer, mask: Tensor, x: Tensor, input_pos: Tensor, logger: "loguru.Logger", **sampling_kwargs) -> Tensor:
    """Pipelined forward pass"""
    logger.debug(f"Calling pipelined_forward(model, {mask=}, {x.shape=}, {input_pos.shape})")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank != 0: # all but first rank receive hidden states
        x = torch.empty((x.size(0), x.size(1), 4096), device=x.device, dtype=torch.bfloat16)
        logger.debug(f"Receiving hidden states {x.shape=}, {x.dtype=}, {x.device=}")
        dist.recv(x, src=(rank-1) % world_size)
        logger.debug(f"Got hidden states {x.shape=}, {x.dtype=}, {x.device=}")

    output = model(mask, x, input_pos)
    logger.debug(f"Computed output {output.shape=}")

    if rank != world_size - 1: # all but last rank
        logger.debug(f"Sending output to 1")
        dist.send(output, dst=1)
        next_token = torch.empty(size=(x.size(0), 1), dtype=x.dtype, device=x.device)
        logger.debug(f"Receiving tokens from 1")
        dist.recv(next_token, src=1)
        logger.debug(f"Got tokens {next_token=}, {next_token.shape=}, {next_token.dtype=}, {next_token.device=}")
    else:
        next_token = sample(output, **sampling_kwargs)[0]
        logger.debug(f"Sampled next token {next_token=}, {next_token.shape=}, {next_token.dtype=}, {next_token.device=}")
        if dist.get_world_size() > 1:
            logger.debug(f"Sending next token to 0")
            dist.send(next_token, dst=0)

    return next_token

def prefill(model: Transformer, x: Optional[Tensor], input_pos: Tensor, logger: "loguru.Logger", **sampling_kwargs) -> Optional[Tensor]:
    """Prefill the model with the given input """
    # input_pos: [B, S]
    logger.debug(f"Calling prefill(model, {x.shape=}, {input_pos.shape=})")
    logger.debug(f"Creating mask")
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logger.debug(f"{mask=}")
    return pipelined_forward(model, mask, x, input_pos, logger, **sampling_kwargs)

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, logger: "loguru.Logger", **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode one token"""
    # input_pos: [B, 1]
    logger.debug(f"Calling decode_one_token(model, {x.shape=}, {input_pos.shape=}, {block_mask=})")
    assert input_pos.shape[-1] == 1
    logger.debug(f"Adjusting mask")
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logger.debug(f"{mask=}")

    next_token = pipelined_forward(model, mask, x, input_pos, logger, **sampling_kwargs)
    logger.debug(f"Decoded next token {next_token.squeeze().tolist()}")
    return next_token


def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, logger: "loguru.Logger", **sampling_kwargs):
    """Decode n tokens"""
    logger.debug(f"Calling decode_n_tokens({cur_token=} ({cur_token.shape=}), {input_pos=} ({input_pos.shape=}), {num_new_tokens=}, {sampling_kwargs=})")
    logger.debug(f"Creating block mask")
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
    logger.debug(f"{block_mask=}")

    new_tokens = []
    for i in range(num_new_tokens):
        next_token = decode_one_token(
            model, cur_token, input_pos, block_mask, logger, **sampling_kwargs
        )
        input_pos += 1
        new_tokens.append(next_token.clone())
        cur_token = next_token.clone()

    return new_tokens

@torch.no_grad()
def generate(
    model: Transformer,
    prompt: Tensor,
    logger: "loguru.Logger",
    max_new_tokens: int,
    batch_size: int,
    **sampling_kwargs
) -> torch.Tensor:
    # Get number of tokens in prompt and to generate
    num_prompt_tokens = prompt.size(-1)
    num_new_tokens = max_new_tokens
    seq_length = min(num_prompt_tokens + num_new_tokens, model.config.block_size) # Cap at model context

    # Setup cache
    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=seq_length)

    # Create empty tensor of prompt tokens
    prompt = prompt.view(1, -1).repeat(batch_size, 1) # Repeat prompt for each batch
    empty = torch.empty(batch_size, seq_length, dtype=dtype, device=device) # Create empty tensor
    empty[:, :num_prompt_tokens] = prompt # Fill in prompt tokens
    input_pos = torch.arange(num_prompt_tokens, device=device)

    # Prefill prompt tokens
    seq = empty
    next_token = prefill(model=model, x=prompt.view(batch_size, -1), input_pos=input_pos, logger=logger, **sampling_kwargs)

    # Decode remaining tokens
    seq[:, num_prompt_tokens] = next_token.squeeze()
    input_pos = torch.tensor([num_prompt_tokens], device=device, dtype=torch.int)
    logger.debug("Decoding remaining tokens")
    generated_tokens = decode_n_tokens(model=model, cur_token=next_token.view(batch_size, -1), input_pos=input_pos, num_new_tokens=max_new_tokens - 1, logger=logger, **sampling_kwargs)
    logger.debug(f"Done decoding, generated tokens={generated_tokens}")
    seq[:, num_prompt_tokens+1:] = torch.cat(generated_tokens, dim=-1)

    return seq

def main(
    prompt: str = "Hello, my name is",
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    precision: torch.dtype = torch.bfloat16,
    seed: int = 1234,
    log_level: str = "INFO", # console log level
) -> None:
    # Initialize world
    world = setup_world()

    # Initialize logger
    logger = setup_logger(world.rank, log_level)
    logger.info(f"Starting")

    # Set seeds for reproducibility across all processes
    seed_everything(seed)
    logger.info(f"Seeded with {seed}")
    
    # Set device and precision
    device = f"cuda:{world.rank}"
    logger.info(f"Using device {device}")
    logger.info(f"Using precision {precision}")

    # Load model
    logger.debug("Loading model ...")
    t0 = time.time()
    model = load_model(checkpoint_path, device, precision)
    torch.cuda.synchronize(device)
    logger.info(f"Loaded model in {time.time() - t0:.02f} seconds")

    # Shard model
    if world.size > 1:
        from model import shard_model
        logger.debug("Sharding model ...")
        model = shard_model(model, world.rank, world.size)
        torch.cuda.synchronize(device)
        logger.info(f"Sharded model in {time.time() - t0:.02f} seconds")

    # Load tokenizer
    logger.debug("Loading tokenizer ...")
    t0 = time.time()
    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
    logger.info(f"Loaded tokenizer in {time.time() - t0:.02f} seconds")

    # Encode prompt
    tokens = [tokenizer.bos_id()] + tokenizer.encode(prompt)
    encoded = torch.tensor(tokens, dtype=torch.int, device=device)

    if compile:
        global create_block_mask
        create_block_mask = torch.compile(create_block_mask)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead") # fullgraph=True

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)

    start = -1 if compile else 0
    metrics = {"latency": [], "tps": []}
    for i in range(start, num_samples):
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        decoded = generate(
            model=model,
            prompt=encoded,
            logger=logger,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
        )
        if i == -1:
            logger.info(f"Compiled model in {time.perf_counter() - start_time:.2f} seconds")
            continue

        # Calculate metrics
        torch.cuda.synchronize(device)
        time_taken = time.perf_counter() - start_time
        num_prompt_tokens = batch_size * encoded.size(-1)
        num_generated_tokens = batch_size * (decoded.size(-1) - encoded.size(-1))
        num_total_tokens = num_prompt_tokens + num_generated_tokens
        metrics["latency"].append(time_taken)
        metrics["tps"].append(num_generated_tokens / time_taken)

        # Print generations (on main rank)
        for i in range(batch_size):
            if world.rank == 0:
                logger.info(f"Sample {i + 1}: {tokenizer.decode(decoded[i].tolist())}")

    # Print metrics (on main rank)
    if world.rank == 0:
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Number of prompt tokens: {num_prompt_tokens}")
        logger.info(f"Number of generated tokens: {num_generated_tokens}")
        logger.info(f"Number of total tokens: {num_total_tokens}")
        logger.info(f"Latency: {torch.mean(torch.tensor(metrics['latency'])).item():.2f} seconds")
        logger.info(f"Throughput: {torch.mean(torch.tensor(metrics['tps'])).item():.2f} tokens/second")
        logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=Path, required=True, help='Model checkpoint path.')
    parser.add_argument('--prompt', type=str, default="Hello, my name is", help="Input prompt.")
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--seed', type=int, default=1234, help='Seed for reproducibility.')
    parser.add_argument('--log_level', type=str, default="INFO", help='Log level.')

    args = parser.parse_args()
    main(
        prompt=args.prompt,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path,
        compile=args.compile,
        compile_prefill=args.compile_prefill,
        seed=args.seed,
        log_level=args.log_level,
    )
