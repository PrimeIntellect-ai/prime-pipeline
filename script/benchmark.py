import argparse
import json
import os
import subprocess
from datetime import datetime
from itertools import product
from multiprocessing import Process, Queue
from pathlib import Path
from time import perf_counter
from typing import Dict, Literal

import autorootcwd  # noqa: F401
import torch
from lovely_tensors import monkey_patch
from tabulate import tabulate

from src.generate import generate
from src.logger import get_logger
from src.setup import setup
from src.utils import flatten_list, mean

# Use lovely tensors
monkey_patch()

# Node IDs given a seed parameter, used for connecting nodes into pipelines
IROH_PAIRS = {
    0: "ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03",
    1: "ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337",
    2: "191fc38f134aaf1b7fdb1f86330b9d03e94bd4ba884f490389de964448e89b3f",
    3: "c5bbbb60e412879bbec7bb769804fa8e36e68af10d5477280b63deeaca931bed",
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
    rank: int,
    queue: Queue,
    local_rank: int,
    world_size: int,
    model_name: str,
    dummy: bool,
    num_iterations: int,
    prompt: str,
    num_new_tokens: int,
    num_cache_tokens: int,
    batch_size: int,
    num_micro_batches: int,
    device: str,
    precision: str,
    backend: str,
    compile: bool,
    latency: int,
    warmup: bool,
    seed: int,
    log_level: str,
    **kwargs
) -> None:
    # Populate environment variables for multi-node setup
    assert rank in IROH_PAIRS, f"Node {rank} is not in the list of known nodes: {IROH_PAIRS.keys()}"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["CACHE_DIR"] = "/workspace"
    if backend == "iroh":
        os.environ["IROH_SEED"] = str(rank)
        os.environ["IROH_PEER_ID"] = IROH_PAIRS[(rank + 1) % world_size]

    # Setup world, logger, comm and load model
    start_setup = perf_counter()
    model, _, decoded_tokens, num_prompt_tokens, micro_batch_size = setup(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        log_level=log_level,
        seed=seed,
        device=device,
        precision=precision,
        model_name=model_name,
        dummy=dummy,
        prompt=prompt,
        backend=backend,
        num_micro_batches=num_micro_batches,
        num_new_tokens=num_new_tokens,
        num_cache_tokens=num_cache_tokens,
        batch_size=batch_size,
        compile=compile,
        latency=latency,
    )
    setup_time = {"setup_time": perf_counter() - start_setup, "compile_time": 0}

    metrics = []
    for sample_idx in range(num_iterations):
        torch.cuda.synchronize()
        _, prefill_metrics, decode_metrics = generate(
            model=model,
            decoded_tokens=decoded_tokens,
            num_prompt_tokens=num_prompt_tokens,
            micro_batch_size=micro_batch_size,
            use_tqdm=True,
        )

        # Compute metrics
        computed_prefill_metrics = compute_metrics(prefill_metrics, warmup=False, stage="prefill")
        computed_decode_metrics = compute_metrics(decode_metrics, warmup=warmup, stage="decode")

        metrics.append(
            {
                "rank": rank,
                "iteration": sample_idx + 1,
                "setup_time": setup_time["setup_time"],
                "compile_time": setup_time["compile_time"],
                "prefill_metrics": prefill_metrics,
                "decode_metrics": decode_metrics,
                **computed_prefill_metrics,
                **computed_decode_metrics,
            }
        )

    # Send metrics to main process
    queue.put((rank, torch.cuda.get_device_name(), metrics))

def main(args: argparse.Namespace) -> None:
    # Prepare static configuration to identify benchmark run
    timestamp = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()[:7],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    static_config = {
        "model_name": args.model_name,
        "world_size": args.world_size,
        "prompt": args.prompt,
        "num_new_tokens": args.num_new_tokens,
        "precision": args.precision,
        "device": args.device,
        "seed": args.seed,
    }

    # Get static and dynamic arguments
    dynamic_args = {arg: getattr(args, arg) for arg in vars(args) if isinstance(getattr(args, arg), list)}
    static_config = {arg: getattr(args, arg) for arg in vars(args) if not isinstance(getattr(args, arg), list)}
    
    file_path = Path(f"benchmark/{args.model_name}.jsonl")
    if args.save:
        os.makedirs(file_path.parent, exist_ok=True)

    # Run benchmarks for each combination
    aggregated_results = []
    for config_idx, dynamic_values in enumerate(list(product(*dynamic_args.values())), start=1):
        # Create configuration dict
        dynamic_config = dict(zip(dynamic_args.keys(), dynamic_values))
        print(f"Running configuration {config_idx} with {dynamic_config}")

        # Skip if batch size is less than number of micro batches
        if dynamic_config["batch_size"] < dynamic_config["num_micro_batches"]:
            print("Skipping because batch size is less than number of micro batches")
            continue

        # Run benchmark
        try:
            queue = Queue()
            ps = [Process(target=run_benchmark, args=(rank, queue), kwargs={**static_config, **dynamic_config}) for rank in range(args.world_size)]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
        except Exception as e:
            print(f"Error: {e}")
            for p in ps:
                p.terminate()
            raise e

        # Get results
        all_metrics = {}
        while not queue.empty():
            rank, gpu, metrics = queue.get()
            all_metrics[rank] = metrics

        # Aggregate metrics
        for rank, metrics in all_metrics.items():

            # Extend metrics with static and dynamic configuration
            metrics = [{**timestamp, **static_config, **dynamic_config, **metric, "gpu": gpu} for metric in metrics]

            # Save results
            if args.save:
                with open(file_path, "a") as f:
                    for metric in metrics:
                        f.write(json.dumps(metric) + "\n")

        # Aggregate most relevant metrics
        metrics_to_aggregate = ["setup_time", "compile_time", "prefill_time", "decode_time", "prefill_tps", "decode_tps"]
        aggregated_metrics = {key: mean([result[key] for result in all_metrics[0]]) for key in metrics_to_aggregate}
        aggregated_results.append(aggregated_metrics)


    # Display aggregated results
    headers = ["config_idx"] + list(aggregated_results[0].keys())
    table = [[config_idx] + list(result.values()) for config_idx, result in enumerate(aggregated_results, start=1)]
    print(tabulate(table, headers=headers))

    print("\nAll benchmarks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Static arguments
    parser.add_argument("--model-name", type=str, default="meta-llama/llama-2-7b-chat-hf", help="HF model name.")
    parser.add_argument("--local-rank", type=int, default=0, help="Sets local rank in CUDA process.")
    parser.add_argument("--world-size", type=int, default=1, help="Number of pipeline stages.")
    parser.add_argument("--num-iterations", type=int, default=3, help="Number of samples to generate.")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Prompt to generate from.")
    parser.add_argument("--num-new-tokens", type=int, default=10, help="Number of tokens to generate.")
    parser.add_argument("--num-cache-tokens", type=int, default=0, help="Number of cache tokens.")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision to use for the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility.")
    parser.add_argument("--log-level", type=str, default="CRITICAL", help="Log level.")
    parser.add_argument("--warmup", type=bool, default=True, help="Whether to exclude the first iteration from the metrics.")
    parser.add_argument("--dummy", action="store_true", help="Use dummy weights.")
    parser.add_argument("--save", action="store_true", help="Save results to CSV.")

    # Dynamic arguments
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1], help="Batch size.")
    parser.add_argument(
        "--num-micro-batches",
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
        type=str,
        nargs="+",
        default=["False"],
        help="Whether to compile the model.",
    )
    parser.add_argument(
        "--latency", type=int, nargs="+", default=[0], help="Add artificial latency (ms) to the network (only works for iroh backend)."
    )

    args = parser.parse_args()

    # Convert compile to bool
    args.compile = [True if compile == "True" else False for compile in args.compile]

    main(args)

