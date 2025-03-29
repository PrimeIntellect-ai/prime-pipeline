import os
from concurrent.futures import Future
from time import perf_counter
from typing import Optional, Tuple

import torch
from torch.nn.attention.flex_attention import create_block_mask

from .comm import setup_comm
from .generate import adjust_mask, causal_mask, compile_generate, micro_step
from .logger import setup_logger
from .model import get_model, get_model_shard
from .offload import get_offload
from .serializer import get_serializer
from .utils import get_device, get_precision, get_tokenizer, seed_everything
from .world import setup_world


def setup(
    rank: int,
    local_rank: int,
    world_size: int,
    log_level: str,
    seed: int,
    device: str,
    precision: str,
    model_name: str,
    dummy: bool,
    prompt: str,
    backend: str,
    num_micro_batches: int,
    num_new_tokens: int,
    num_cache_tokens: int,
    batch_size: int,
    compile: bool,
    latency: Optional[int] = None,  # Only set for benchmark
) -> Tuple:
    """
    Setup world, logger and communication backend globally. Also, loads model (shard), tokenizes
    prompt and compiles model if requested.
    """
    # Setup world
    world = setup_world(rank, local_rank, world_size)

    # Set OMP_NUM_THREADS
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // world_size)

    # Initialize logger
    logger = setup_logger(rank=rank, log_level=log_level)
    torch.cuda.synchronize()

    # Set seeds for reproducibility across all processes
    logger.info(f"Seeding with {seed}")
    seed_everything(seed)

    # Setup device
    device = get_device(device=device, world=world)
    logger.info(f"Using device {device}")

    # Setup precision
    precision = get_precision(precision=precision)
    logger.info(f"Using precision {precision}")

    # Load model
    t0 = perf_counter()
    logger.info(f"Loading model {model_name}...")
    if world.size == 1:
        model = get_model(model_name=model_name, device=device, precision=precision, dummy=dummy)
        logger.info(f"Loaded model in {perf_counter() - t0:.02f} seconds")
    else:
        model = get_model_shard(model_name=model_name, rank=rank, world_size=world_size, device=device, precision=precision, dummy=dummy)
        logger.info(f"Loaded model shard in {perf_counter() - t0:.02f} seconds")

    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = get_tokenizer(model_name=model_name)

    # Encode prompt (batch_size, num_prompt_tokens)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device).repeat(batch_size, 1)
    num_prompt_tokens = prompt_tokens.size(-1)

    # Setup model cache
    assert batch_size >= num_micro_batches, "Batch size must be at least as large as number of micro batches"
    assert batch_size % num_micro_batches == 0, f"Batch size {batch_size} must be divisible by number of micro batches {num_micro_batches}"
    micro_batch_size = batch_size // num_micro_batches
    num_total_tokens = num_prompt_tokens + num_new_tokens
    assert num_total_tokens <= model.config.block_size, (
        f"Total tokens {num_total_tokens} must be less than or equal to model block size {model.config.block_size}"
    )
    if num_cache_tokens == 0:
        num_cache_tokens = num_total_tokens
    assert num_cache_tokens >= num_total_tokens, (
        f"Number of cache tokens {num_cache_tokens} must be greater than or equal to total tokens {num_total_tokens}"
    )
    logger.info(f"Setting up KV cache for {num_cache_tokens} tokens...")
    t0 = perf_counter()
    with torch.device(device):
        model.setup_caches(
            num_micro_batches=num_micro_batches,
            max_micro_batch_size=micro_batch_size,
            max_seq_length=num_cache_tokens,
        )
    logger.info(f"Set up KV cache in {perf_counter() - t0:.02f} seconds")

    # Allocate tensor for decoded tokens
    decoded_tokens = torch.empty(batch_size, num_total_tokens, dtype=prompt_tokens.dtype, device=device)
    decoded_tokens[:, :num_prompt_tokens] = prompt_tokens

    # Setup serializer
    logger.info("Setting up serializer...")
    serializer = get_serializer()

    # Setup offload
    logger.info("Setting up offloader...")
    offloader = get_offload(device)

    # Setup communication
    logger.info(f"Setting up communication backend {backend}...")
    setup_comm(backend, device=device, serializer=serializer, offload=offloader, latency=latency, num_micro_batches=num_micro_batches)

    # Compile model
    logger.info("Compiling model...")
    if compile:
        compile_generate()
    t0 = perf_counter()

    # Fake prefill
    with torch.no_grad():
        logger.info("Faking prefill...")
        input_pos = torch.arange(num_prompt_tokens, device=device)
        if world.is_first_stage:
            inputs = torch.randint(0, model.config.vocab_size, (micro_batch_size, num_prompt_tokens), device=device)
        else:
            inputs = torch.randn(micro_batch_size, num_prompt_tokens, model.config.dim, device=device, dtype=torch.bfloat16)
        input_future = Future()
        input_future.set_result(inputs)
        block_mask = create_block_mask(
            causal_mask,
            1,
            1,
            input_pos.shape[0],
            model.max_seq_length,
            device=decoded_tokens.device,
        )
        for micro_batch_idx in range(num_micro_batches):
            micro_step(
                input_future,
                model,
                block_mask,
                input_pos,
                micro_batch_idx,
                sampling_kwargs={},
                recv_kwargs={},
                skip_send=True,
                skip_recv=True,
            )

        # Fake decode one step
        logger.info("Faking decode step...")
        block_mask = create_block_mask(
            causal_mask,
            1,
            1,
            model.max_seq_length,
            model.max_seq_length,
            device=device,
        )
        if world.is_first_stage:
            inputs = torch.randint(0, model.config.vocab_size, (micro_batch_size, 1), device=device, dtype=torch.long)
        else:
            inputs = torch.randn(micro_batch_size, 1, model.config.dim, dtype=torch.bfloat16, device=device)
        input_future = Future()
        input_future.set_result(inputs)
        for token_idx in range(3):
            input_pos = torch.tensor([num_prompt_tokens + token_idx], device=device, dtype=torch.long)
            mask = adjust_mask(block_mask, input_pos, model.max_seq_length)
            for micro_batch_idx in range(num_micro_batches):
                micro_step(
                    input_future,
                    model,
                    mask,
                    input_pos,
                    micro_batch_idx,
                    sampling_kwargs={},
                    recv_kwargs={},
                    skip_send=True,
                    skip_recv=True,
                )

    logger.info(f"Model compiled in {perf_counter() - t0} seconds")

    return model, tokenizer, decoded_tokens, num_prompt_tokens, micro_batch_size
