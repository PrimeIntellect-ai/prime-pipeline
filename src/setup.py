from time import perf_counter
from typing import Optional, Tuple

import torch

from .comm import setup_comm
from .logger import setup_logger
from .model import get_model, get_model_shard
from .serializer import get_serializer
from .utils import get_device, get_precision, get_tokenizer, seed_everything
from .world import setup_world


def setup(
    rank: int,
    world_size: int,
    log_level: str,
    seed: int,
    device: str,
    precision: str,
    model_name: str,
    dummy: bool,
    prompt: str,
    compile: bool,
    backend: str,
    micro_batch_size: int,
    batch_size: int,
    latency: Optional[int] = None,  # Only set for benchmark
) -> Tuple:
    """
    Setup world, logger and communication backend globally. Also, loads model (shard), tokenizes
    prompt and compiles model if requested.
    """
    # Setup world
    world = setup_world(rank, world_size)

    # Initialize logger
    logger = setup_logger(rank=rank, log_level=log_level)
    torch.cuda.synchronize()

    # Set seeds for reproducibility across all processes
    logger.info(f"Seeding with {seed}")
    seed_everything(seed)

    # Setup device
    logger.info(f"Using device {device}")
    device = get_device(device=device, world=world)

    # Setup precision
    logger.info(f"Using precision {precision}")
    precision = get_precision(precision=precision)

    # Load model
    t0 = perf_counter()
    if world.size == 1:
        model = get_model(model_name=model_name, device=device, precision=precision, dummy=dummy)
        logger.info(f"Loaded model in {perf_counter() - t0:.02f} seconds")
    else:
        model = get_model_shard(model_name=model_name, rank=rank, world_size=world_size, device=device, precision=precision, dummy=dummy)
        logger.info(f"Loaded model shard in {perf_counter() - t0:.02f} seconds")

    # Compile model
    if compile:
        global create_block_mask
        create_block_mask = torch.compile(create_block_mask, fullgraph=True)

        global adjust_mask
        adjust_mask = torch.compile(adjust_mask, fullgraph=True)

        global model_forward
        model_forward = torch.compile(model_forward, fullgraph=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = get_tokenizer(model_name=model_name)

    # Encode prompt (batch_size, num_prompt_tokens)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device).repeat(batch_size, 1)

    # Setup communication
    num_prompt_tokens = prompt_tokens.size(-1)
    micro_batch_size = micro_batch_size if micro_batch_size > 0 else batch_size
    num_micro_batches = batch_size // micro_batch_size
    hidden_states_shape = (micro_batch_size, 1, model.config.dim)
    tokens_shape = (micro_batch_size, 1)
    hidden_states_dtype = model.layers[0].feed_forward.w1.weight.dtype
    tokens_dtype = torch.long
    if backend == "torch":
        kwargs = dict(
            fwd_shape=hidden_states_shape,
            bwd_shape=tokens_shape,
            fwd_dtype=hidden_states_dtype,
            bwd_dtype=tokens_dtype,
            device=device,
            num_prompt_tokens=num_prompt_tokens,
        )
    elif backend == "iroh":
        serializer = get_serializer()
        kwargs = dict(
            serializer=serializer,
            num_micro_batches=num_micro_batches,
            latency=latency,
            device=device,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
    setup_comm(backend, **kwargs)

    return model, tokenizer, prompt_tokens, micro_batch_size
