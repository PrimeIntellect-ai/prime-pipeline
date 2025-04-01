from time import perf_counter
from typing import Optional, Tuple

import torch

from .comm import setup_comm
from .generate import fake_generate, full_compile
from .logger import setup_logger
from .model import get_model, get_model_shard
from .offload import get_offload
from .serializer import get_serializer
from .utils import get_precision, get_tokenizer, seed_everything, setup_device
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
    batch_size: Optional[int],
    micro_batch_size: Optional[int],
    num_micro_batches: int,
    num_new_tokens: int,
    num_cache_tokens: int,
    compile: bool,
    latency: Optional[int] = None,  # Only set for benchmark
) -> Tuple:
    """
    Setup world, logger and communication backend globally. Also, loads model (shard), tokenizes
    prompt and compiles model if requested.
    """
    # Setup world
    world = setup_world(rank, local_rank, world_size)

    # Initialize logger
    logger = setup_logger(rank=rank, log_level=log_level)
    torch.cuda.synchronize()

    # Set seeds for reproducibility across all processes
    logger.info(f"Seeding with {seed}")
    seed_everything(seed)

    # Setup device
    device = setup_device(device=device, world=world)
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
    logger.info("Encoding prompt...")
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    num_prompt_tokens = prompt_tokens.size(-1)
    
    assert (batch_size is not None) ^ (micro_batch_size is not None), "Either batch_size or micro_batch_size must be provided, but not both"
    if batch_size is not None:
        logger.info(f"Auto-detecting micro batch size from global batch size {batch_size}...")
        micro_batch_size = batch_size // num_micro_batches
    else:
        logger.info(f"Auto-detecting global batch size from micro batch size {micro_batch_size}...")
        batch_size = num_micro_batches * micro_batch_size
    logger.info(f"Setting {batch_size=}, {micro_batch_size=}, {num_micro_batches=}")
    prompt_tokens = list(prompt_tokens.repeat(batch_size, 1).split(micro_batch_size))

    # Setup model cache
    assert batch_size >= num_micro_batches, "Batch size must be at least as large as number of micro batches"
    assert batch_size % num_micro_batches == 0, f"Batch size {batch_size} must be divisible by number of micro batches {num_micro_batches}"
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

    # Compile model
    logger.info("Compiling model...")
    start_compile = perf_counter()
    if compile:
        full_compile()
    fake_generate(model, num_prompt_tokens, num_micro_batches, micro_batch_size)
    compile_time = perf_counter() - start_compile
    logger.info(f"Model compiled in {compile_time:.02f} seconds")

    # Setup communication
    logger.info(f"Setting up communication backend {backend}...")
    setup_comm(
        backend,
        device=device,
        serializer=get_serializer(),
        offload=get_offload(device),
        latency=latency,
        num_micro_batches=num_micro_batches,
    )

    return model, tokenizer, prompt_tokens, num_prompt_tokens, batch_size, micro_batch_size, compile_time
