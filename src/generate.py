import argparse
import time
from typing import Any, List

import torch
import torch.nn as nn
from lovely_tensors import monkey_patch
from torch.nn.attention.flex_attention import create_block_mask

from comm import setup_comm
from decode import decode, prefill
from logger import setup_logger
from model import get_model, get_model_shard
from serializer import get_serializer
from tokenizer import get_tokenizer
from utils import get_device, get_precision, seed_everything
from world import get_world

# Use lovely tensors
monkey_patch()


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    batch_size: int,
    num_new_tokens: int,
    micro_batch_size: int,
    **sampling_kwargs,
) -> List[str]:
    """
    Generate tokens.

    Args:
        model: Transformer
        prompt_tokens: Tensor of shape [batch_size, num_prompt_tokens]
        num_new_tokens: int
        micro_batch_size: int
        **sampling_kwargs: Dict of kwargs for the sample function
    """
    # Encode prompt
    device = model.layers[0].feed_forward.w1.weight.device
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tokens = torch.tensor(
        prompt_tokens,
        device=device,
    ).repeat(batch_size, 1)

    # Setup model cache
    num_prompt_tokens = prompt_tokens.size(-1)
    num_micro_batches = batch_size // micro_batch_size
    num_total_tokens = min(num_prompt_tokens + num_new_tokens, model.config.block_size)
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
    prefill(
        model=model,
        decoded_tokens=decoded_tokens,
        micro_batch_size=micro_batch_size,
        num_prompt_tokens=num_prompt_tokens,
        **sampling_kwargs,
    )

    # Decode remaining tokens in-place
    decode(
        model=model,
        decoded_tokens=decoded_tokens,
        micro_batch_size=micro_batch_size,
        num_prompt_tokens=num_prompt_tokens,
        **sampling_kwargs,
    )

    # Decode generations
    generations = [tokenizer.decode(decoded_tokens[i].tolist()) for i in range(batch_size)]

    return generations


def main(args: argparse.Namespace) -> None:
    # Setup world
    world = get_world()

    # Initialize logger
    logger = setup_logger(world.rank, log_level=args.log_level)
    logger.info("Starting generation...")
    torch.cuda.synchronize()
    logger.info(f"Args: {vars(args)}")

    # Set seeds for reproducibility across all processes
    logger.info(f"Setting seeds for reproducibility {args.seed=}")
    seed_everything(args.seed)

    # Load model
    device = get_device(args.device, world)
    precision = get_precision(args.precision)
    logger.info(f"Loading model {args.model_name} on device {device} and precision {precision}")
    t0 = time.time()
    if world.size == 1:
        model = get_model(args.model_name, device, precision)
        logger.info(f"Loaded model in {time.time() - t0:.02f} seconds")
    else:
        model = get_model_shard(args.model_name, world.rank, world.size, device, precision)
        logger.info(f"Loaded model shard in {time.time() - t0:.02f} seconds")

    # Compile model
    if args.compile:
        global create_block_mask
        create_block_mask = torch.compile(create_block_mask, fullgraph=True)

        global adjust_mask
        adjust_mask = torch.compile(adjust_mask, fullgraph=True)

        global model_forward
        model_forward = torch.compile(model_forward, fullgraph=True)

    # Load tokenizer
    logger.info(f"Loading tokenizer for {args.model_name}")
    tokenizer = get_tokenizer(args.model_name)

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
    num_prompt_tokens = prompt_tokens.size(-1)
    num_micro_batches = args.batch_size // micro_batch_size
    hidden_states_shape = (micro_batch_size, 1, model.config.dim)
    tokens_shape = (micro_batch_size, 1)
    hidden_states_dtype = model.layers[0].feed_forward.w1.weight.dtype
    tokens_dtype = torch.long
    if args.backend == "torch":
        kwargs = dict(
            fwd_shape=hidden_states_shape,
            bwd_shape=tokens_shape,
            fwd_dtype=hidden_states_dtype,
            bwd_dtype=tokens_dtype,
            device=device,
            num_prompt_tokens=num_prompt_tokens,
        )
    elif args.backend == "iroh":
        serializer = get_serializer(device=device)
        kwargs = dict(
            serializer=serializer,
            num_micro_batches=num_micro_batches,
        )
    else:
        raise ValueError(f"Invalid backend: {args.backend}")
    comm = setup_comm(args.backend, **kwargs)

    for sample_idx in range(-1 if args.compile else 0, args.num_samples):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        generations = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            num_new_tokens=args.num_new_tokens,
            batch_size=args.batch_size,
            micro_batch_size=micro_batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        if sample_idx == -1:
            logger.info(f"Compiled in {time.perf_counter() - start_time:.2f} seconds")
            continue

        if world.is_master:
            logger.info(f"Generations {sample_idx + 1}")
            for batch_idx, generation in enumerate(generations):
                logger.info(f"Generation {batch_idx + 1}: {generation}")

    # Destroy communication
    comm.destroy()
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/llama-2-7b-chat-hf",
        help="Model name.",
    )
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Input prompt.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Number of micro-batches. If not provided, will use world.size if batch_size >= world.size else 1.",
    )
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument(
        "--num-new-tokens",
        type=int,
        default=50,
        help="Number of new tokens to generate.",
    )
    parser.add_argument("--top-k", type=int, default=200, help="Top-k for sampling.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling.")
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Whether to compile the model.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model.")
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        help="Precision to use for the model.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        help="Either `torch` or `iroh`.",
    )
    main(parser.parse_args())
