import argparse
import os
import time

import torch
from lovely_tensors import monkey_patch

from .comm import get_comm
from .generate import generate
from .logger import get_logger
from .setup import setup
from .world import get_world

# Use lovely tensors
monkey_patch()


def main(args: argparse.Namespace) -> None:
    # Setup
    model, tokenizer, prompt_tokens = setup(
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        log_level=args.log_level,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        model_name=args.model_name,
        prompt=args.prompt,
        compile=args.compile,
        backend=args.backend,
        micro_batch_size=args.micro_batch_size,
        batch_size=args.batch_size,
    )

    # Get world, logger and comm
    world, logger, comm = get_world(), get_logger(), get_comm()

    # Generate
    for sample_idx in range(-1 if args.compile else 0, 1):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        decoded_tokens, _, _ = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            num_new_tokens=args.num_new_tokens,
            micro_batch_size=args.micro_batch_size,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        if sample_idx == -1:
            logger.info(f"Compiled in {time.perf_counter() - start_time:.2f} seconds")
            continue

        logger.debug(f"Decoded tokens: {decoded_tokens.tolist()}")
        if world.is_master:
            for batch_idx, generation in enumerate(decoded_tokens):
                logger.info(f"Generation {batch_idx + 1}: {tokenizer.decode(generation.tolist())}")

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
        default=1,
        help="Number of micro-batches.",
    )
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
