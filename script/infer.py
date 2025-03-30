import argparse
import os
from time import perf_counter

import autorootcwd  # noqa: F401
import torch
from lovely_tensors import monkey_patch

from src.comm import get_comm
from src.generate import generate
from src.logger import get_logger
from src.setup import setup
from src.utils import to_int_or_none
from src.world import get_world

# Use lovely tensors
monkey_patch()


def main(args: argparse.Namespace) -> None:
    # Setup
    model, tokenizer, prompt_tokens, num_prompt_tokens, micro_batch_size = setup(
        rank=int(os.environ.get("RANK", 0)),
        local_rank=to_int_or_none(os.environ.get("LOCAL_RANK")),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
        log_level=args.log_level,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        model_name=args.model_name,
        dummy=args.dummy,
        prompt=args.prompt,
        backend=args.backend,
        num_micro_batches=args.num_micro_batches,
        num_new_tokens=args.num_new_tokens,
        num_cache_tokens=args.num_cache_tokens,
        batch_size=args.batch_size,
        compile=args.compile,
    )

    # Get world, logger and comm
    world, logger, comm = get_world(), get_logger(), get_comm()

    # Generate
    torch.cuda.synchronize()
    start_generate = perf_counter()
    decoded_tokens, prefill_time, decode_time = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_new_tokens=args.num_new_tokens,
        num_micro_batches=args.num_micro_batches,
        micro_batch_size=micro_batch_size,
        disable_tqdm=args.disable_tqdm,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # Print generations
    if world.is_master:
        for batch_idx, generation in enumerate(decoded_tokens):
            logger.info(f"Generation {batch_idx + 1}: {tokenizer.decode(generation.tolist(), skip_special_tokens=True)}")

    # Print metrics
    generate_time = perf_counter() - start_generate
    logger.info(f"Time: {generate_time:.02f}s (Prefill: {prefill_time:.02f}s, Decode: {decode_time:.02f}s)")
    logger.info(f"Generated Tokens: {args.batch_size * args.num_new_tokens}")
    logger.info(f"Throughput: {(args.batch_size * args.num_new_tokens) / generate_time:.02f} T/s")

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
        "--num-micro-batches",
        type=int,
        default=1,
        help="Number of micro-batches.",
    )
    parser.add_argument("--num-cache-tokens", type=int, default=0, help="Number of cache tokens.")
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
    parser.add_argument("--dummy", action="store_true", help="Use dummy weights.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm for progress bar.")
    args = parser.parse_args()

    main(args)
