import argparse
import json
import os
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import torch
from lovely_tensors import monkey_patch
from tabulate import tabulate
from torch.nn.attention.flex_attention import create_block_mask
from tqdm import tqdm

from comm import setup_comm
from decode import generate
from logger import setup_logger
from model import get_model, get_model_shard
from serializer import get_serializer
from tokenizer import get_tokenizer
from utils import get_device, get_precision, seed_everything
from world import get_world

# Use lovely tensors
monkey_patch()


def run_benchmark(
    config_idx: int,
    model_name: str,
    num_iterations: int,
    num_prompt_tokens: int,
    num_new_tokens: int,
    batch_size: int,
    micro_batch_size: int,
    device: str,
    precision: str,
    backend: str,
    compile: bool,
    warmup: bool,
    seed: int,
    log_level: str,
) -> Dict:
    # Get world
    world = get_world()

    # Setup logger
    logger = setup_logger(world.rank, log_level=log_level)

    # Setup logger
    startup_start = perf_counter()

    # Set seeds for reproducibility across all processes
    seed_everything(seed)

    # Load model
    device = get_device(device, world)
    precision = get_precision(precision)
    startup_metrics = {}

    if world.size == 1:
        model = get_model(model_name, device, precision)
    else:
        model = get_model_shard(model_name, world.rank, world.size, device, precision)

    # Compile model
    if compile:
        global create_block_mask
        create_block_mask = torch.compile(create_block_mask, fullgraph=True)

        global adjust_mask
        adjust_mask = torch.compile(adjust_mask, fullgraph=True)

        global model_forward
        model_forward = torch.compile(model_forward, fullgraph=True)

    # Load tokenizer
    tokenizer = get_tokenizer(model_name)

    # Encode prompt (batch_size, num_prompt_tokens)
    prompt_tokens = [tokenizer.bos_id()] * num_prompt_tokens
    prompt_tokens = torch.tensor(prompt_tokens, device=device)
    prompt_tokens = prompt_tokens.repeat(batch_size, 1)

    # Setup communication
    num_prompt_tokens = prompt_tokens.size(-1)
    if micro_batch_size == "auto":
        micro_batch_size = batch_size // world.size if batch_size >= world.size else 1
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
        serializer = get_serializer(device=device)
        kwargs = dict(
            serializer=serializer,
            num_micro_batches=num_micro_batches,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")
    comm = setup_comm(backend, **kwargs)
    startup_time = perf_counter() - startup_start

    metrics = []
    desc = f"Configuration {config_idx + 1} ({batch_size=}, {micro_batch_size=}, {backend=}, {compile=})"
    for sample_idx in tqdm(range(-1 if compile else 0, num_iterations), desc=desc):
        torch.cuda.synchronize()
        start_time = perf_counter()
        _, prefill_metrics, decode_metrics = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            num_new_tokens=num_new_tokens,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
        )
        if sample_idx == -1:
            startup_metrics["compile_time"] = perf_counter() - start_time
            logger.info(f"Compiled in {startup_metrics['compile_time']:.2f} seconds")
            continue

        # Compute metrics for this iteration
        time_per_token_list = list(map(sum, decode_metrics["time_per_token"][int(warmup) :]))
        decode_time = sum(time_per_token_list)
        time_per_token = decode_time / len(time_per_token_list)
        decode_tps = (decode_metrics["num_decode_tokens"] - int(warmup) * batch_size) / decode_time

        metrics.append(
            {
                "iteration": sample_idx + 1,
                "startup_time": startup_time,
                "prefill_time": prefill_metrics["prefill_time"],
                "prefill_tokens": prefill_metrics["prefill_tokens"],
                "decode_time": decode_time,
                "decode_tokens": decode_metrics["num_decode_tokens"],
                "throughput": decode_tps,
                "time_per_token": time_per_token,
            }
        )

    # Destroy communication
    comm.destroy()

    return metrics


def main(args: argparse.Namespace) -> None:
    # Prepare static configuration to identify benchmark run
    timestamp = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()[:7],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    static_config = {
        "model_name": args.model_name,
        "num_prompt_tokens": args.num_prompt_tokens,
        "num_new_tokens": args.num_new_tokens,
        "precision": args.precision,
        "device": args.device,
        "seed": args.seed,
    }

    # Generate all combinations for dynamic arguments
    list_args = {arg: getattr(args, arg) for arg in vars(args) if isinstance(getattr(args, arg), list)}
    dynamic_config_combinations = list(product(*list_args.values()))
    dynamic_config_names = list(list_args.keys())

    if args.save:
        file_path = Path(f"benchmark/{args.model_name}.jsonl")
        os.makedirs(file_path.parent, exist_ok=True)

    # Run benchmarks for each combination
    aggregated_results = []
    for config_idx, config_combination in enumerate(dynamic_config_combinations):
        # Create configuration dict
        config = dict(zip(dynamic_config_names, config_combination))

        # Run benchmark
        metrics = run_benchmark(
            config_idx=config_idx,
            model_name=args.model_name,
            num_iterations=args.num_iterations,
            num_prompt_tokens=args.num_prompt_tokens,
            num_new_tokens=args.num_new_tokens,
            device=args.device,
            precision=args.precision,
            seed=args.seed,
            warmup=args.warmup,
            log_level=args.log_level,
            **config,
        )

        # Aggregate metrics
        def mean(values: List[float]) -> float:
            return sum(values) / len(values)

        aggregated_metrics = {key: mean([result[key] for result in metrics]) for key in metrics[0].keys()}
        aggregated_metrics.pop("iteration")
        aggregated_results.append(aggregated_metrics)

        # Extend metrics with static and dynamic configuration
        metrics = [{**timestamp, **static_config, **config, **metric} for metric in metrics]

        # Save results
        if args.save:
            with open(file_path, "a") as f:
                for metric in metrics:
                    f.write(json.dumps(metric) + "\n")

    # Display aggregated results
    headers = aggregated_results[0].keys()
    table = [list(result.values()) for result in aggregated_results]
    print(tabulate(table, headers=headers))

    print("\nAll benchmarks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Constant arguments
    parser.add_argument("--model-name", type=str, default="meta-llama/llama-2-7b-chat-hf", help="HF model name.")
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--num-prompt-tokens", type=int, default=1, help="Number of prompt tokens.")
    parser.add_argument("--num-new-tokens", type=int, default=10, help="Number of tokens to generate.")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision to use for the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    parser.add_argument("--warmup", type=bool, default=True, help="Whether to exclude the first iteration from the metrics.")
    parser.add_argument("--save", action="store_true", help="Save results to CSV.")

    # Combination arguments
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1], help="Batch size.")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        nargs="+",
        default=["auto"],
        help="Number of micro-batches. If 'auto', will determine automatically.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        default=["torch"],
        help="Either `torch` or `iroh`.",
    )
    parser.add_argument(
        "--compile",
        type=bool,
        nargs="+",
        default=[False],
        help="Whether to compile the model.",
    )

    args = parser.parse_args()
    world = get_world()

    main(args)
