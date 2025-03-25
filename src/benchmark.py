import argparse
import json
import os
import subprocess
from datetime import datetime
from itertools import product
from multiprocessing import Process
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Literal

import torch
from lovely_tensors import monkey_patch
from tabulate import tabulate
from tqdm import tqdm

from .comm import destroy_comm
from .generate import generate
from .logger import get_logger
from .setup import setup
from .utils import flatten_list, mean
from .world import setup_world

# Use lovely tensors
monkey_patch()

# Node IDs given a seed parameter, used for connecting nodes into pipelines
IROH_PAIRS = {
    0: "ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03",
    1: "ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337",
    2: "c5bbbb60e412879bbec7bb769804fa8e36e68af10d5477280b63deeaca931bed",
    3: "4f44e6c7bdfed3d9f48d86149ee3d29382cae8c83ca253e06a70be54a301828b",
}


# Compute decode metrics for this iteration
def compute_metrics(metrics: Dict, warmup: bool, stage: Literal["prefill", "decode"]) -> Dict:
    total_time = sum(flatten_list(metrics["times"][int(warmup) :]))
    mean_forward_time = mean(flatten_list(metrics["forward_times"][int(warmup) :]))
    mean_wait_time = mean(flatten_list(metrics["wait_times"][int(warmup) :]))
    tps = (metrics["num_new_tokens"] - int(warmup)) / total_time
    return {
        f"{stage}_time": total_time,
        f"{stage}_tps": tps,
        f"{stage}_forward_time": mean_forward_time,
        f"{stage}_wait_time": mean_wait_time,
        f"{stage}_num_new_tokens": metrics["num_new_tokens"],
    }


def run_benchmark(
    config_idx: int,
    rank: int,
    world_size: int,
    model_name: str,
    num_iterations: int,
    prompt: str,
    num_new_tokens: int,
    batch_size: int,
    micro_batch_size: int,
    device: str,
    precision: str,
    backend: str,
    compile: bool,
    latency: int,
    warmup: bool,
    seed: int,
    log_level: str,
) -> Dict:
    # Populate environment variables for multi-node setup
    assert rank in IROH_PAIRS, f"Node {rank} is not in the list of known nodes: {IROH_PAIRS.keys()}"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if backend == "iroh":
        os.environ["IROH_SEED"] = str(rank)
        os.environ["IROH_PEER_ID"] = IROH_PAIRS[(rank + 1) % world_size]

    # Setup world, logger, comm and load model
    start_setup = perf_counter()
    model, _, prompt_tokens = setup(
        rank=rank,
        world_size=world_size,
        log_level=log_level,
        seed=seed,
        device=device,
        precision=precision,
        model_name=model_name,
        prompt=prompt,
        compile=compile,
        backend=backend,
        micro_batch_size=micro_batch_size,
        batch_size=batch_size,
        latency=latency,
    )
    setup_time = {"setup_time": perf_counter() - start_setup, "compile_time": 0}

    metrics = []
    desc = f"Configuration {config_idx} ({batch_size=}, {micro_batch_size=}, {backend=}, {compile=})"
    iter = range(-1 if compile else 0, num_iterations)
    iter = tqdm(iter, desc=desc) if rank == 0 else iter
    for sample_idx in iter:
        torch.cuda.synchronize()
        start_generate = perf_counter()
        _, prefill_metrics, decode_metrics = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            num_new_tokens=num_new_tokens,
            micro_batch_size=micro_batch_size,
        )
        if sample_idx == -1:
            setup_time["compile_time"] = perf_counter() - start_generate
            get_logger().info(f"Compiled in {setup_time['compile_time']:.2f} seconds")
            continue

        # Compute metrics
        computed_prefill_metrics = compute_metrics(prefill_metrics, warmup=False, stage="prefill")
        computed_decode_metrics = compute_metrics(decode_metrics, warmup=warmup, stage="decode")

        metrics.append(
            {
                "iteration": sample_idx + 1,
                "setup_time": setup_time["setup_time"],
                "compile_time": setup_time["compile_time"],
                **computed_prefill_metrics,
                **computed_decode_metrics,
            }
        )

    # Destroy communication (necessary for iroh backend, because it adjusts streams to number of micro batches)
    destroy_comm()

    return metrics


def main(rank: int, args: argparse.Namespace) -> None:
    # Setup world
    world = setup_world(rank=rank, size=args.num_devices)

    # Prepare static configuration to identify benchmark run
    timestamp = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()[:7],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    static_config = {
        "model_name": args.model_name,
        "num_devices": args.num_devices,
        "prompt": args.prompt,
        "num_new_tokens": args.num_new_tokens,
        "precision": args.precision,
        "device": args.device,
        "seed": args.seed,
    }

    # Generate all combinations for dynamic arguments
    list_args = {arg: getattr(args, arg) for arg in vars(args) if isinstance(getattr(args, arg), list)}
    dynamic_config_combinations = list(product(*list_args.values()))
    dynamic_config_names = list(list_args.keys())

    if world.is_master and args.save:
        file_path = Path(f"benchmark/{args.model_name}.jsonl")
        os.makedirs(file_path.parent, exist_ok=True)

    # Run benchmarks for each combination
    aggregated_results = []
    for config_idx, config_combination in enumerate(dynamic_config_combinations, start=1):
        # Create configuration dict
        config = dict(zip(dynamic_config_names, config_combination))

        # Run benchmark
        metrics = run_benchmark(
            config_idx=config_idx,
            rank=rank,
            world_size=args.num_devices,
            model_name=args.model_name,
            num_iterations=args.num_iterations,
            prompt=args.prompt,
            num_new_tokens=args.num_new_tokens,
            device=args.device,
            precision=args.precision,
            seed=args.seed,
            warmup=args.warmup,
            log_level=args.log_level,
            **config,
        )

        # Ensure cache is emptied
        torch.cuda.empty_cache()

        # Aggregate metrics
        def mean(values: List[float]) -> float:
            return sum(values) / len(values)

        aggregated_metrics = {key: mean([result[key] for result in metrics]) for key in metrics[0].keys()}
        aggregated_metrics.pop("iteration")
        aggregated_results.append(aggregated_metrics)

        # Extend metrics with static and dynamic configuration
        metrics = [{**timestamp, **static_config, **config, **metric} for metric in metrics]

        # Save results
        if world.is_master and args.save:
            with open(file_path, "a") as f:
                for metric in metrics:
                    f.write(json.dumps(metric) + "\n")

    # Display aggregated results
    if world.is_master:
        headers = ["config_idx"] + list(aggregated_results[0].keys())
        table = [[config_idx] + list(result.values()) for config_idx, result in enumerate(aggregated_results, start=1)]
        print(tabulate(table, headers=headers))

        print("\nAll benchmarks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Constant arguments
    parser.add_argument("--model-name", type=str, default="meta-llama/llama-2-7b-chat-hf", help="HF model name.")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of pipeline stages.")
    parser.add_argument("--num-iterations", type=int, default=3, help="Number of samples to generate.")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Prompt to generate from.")
    parser.add_argument("--num-new-tokens", type=int, default=10, help="Number of tokens to generate.")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision to use for the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility.")
    parser.add_argument("--log-level", type=str, default="CRITICAL", help="Log level.")
    parser.add_argument("--warmup", type=bool, default=True, help="Whether to exclude the first iteration from the metrics.")
    parser.add_argument("--save", action="store_true", help="Save results to CSV.")

    # Combination arguments
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1], help="Batch size.")
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        nargs="+",
        default=[1],
        help="Number of micro batches.",
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
    parser.add_argument(
        "--latency", type=int, nargs="+", default=[0], help="Add artificial latency (ms) to the network (only works for iroh backend)."
    )

    args = parser.parse_args()

    ps = [
        Process(
            target=main,
            args=(
                rank,
                args,
            ),
        )
        for rank in range(args.num_devices)
    ]

    for p in ps:
        p.start()
    for p in ps:
        p.join()
