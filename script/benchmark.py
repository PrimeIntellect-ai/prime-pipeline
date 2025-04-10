import argparse
import json
import os
import subprocess
from datetime import datetime
from itertools import product
from multiprocessing import Process, Queue
from pathlib import Path
from time import perf_counter

import autorootcwd  # noqa: F401
import torch
from lovely_tensors import monkey_patch
from tabulate import tabulate

from src.generate import generate
from src.setup import setup
from src.utils import mean

# Use lovely tensors
monkey_patch()

# Node IDs given a seed parameter, used for connecting nodes into pipelines
IROH_PAIRS = {
    0: "ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03",
    1: "ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337",
    2: "191fc38f134aaf1b7fdb1f86330b9d03e94bd4ba884f490389de964448e89b3f",
    3: "c5bbbb60e412879bbec7bb769804fa8e36e68af10d5477280b63deeaca931bed",
    4: "4f44e6c7bdfed3d9f48d86149ee3d29382cae8c83ca253e06a70be54a301828b",
    5: "e2e8aa145e1ec5cb01ebfaa40e10e12f0230c832fd8135470c001cb86d77de00",
    6: "17888c2ca502371245e5e35d5bcf35246c3bc36878e859938c9ead3c54db174f",
    7: "478243aed376da313d7cf3a60637c264cb36acc936efb341ff8d3d712092d244",
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
    micro_batch_size: int,
    num_micro_batches: int,
    device: str,
    precision: str,
    backend: str,
    compile: bool,
    disable_tqdm: bool,
    latency: int,
    seed: int,
    log_level: str,
    **kwargs,
) -> None:
    try:
        # Populate environment variables for multi-node setup
        assert rank in IROH_PAIRS, f"Node {rank} is not in the list of known nodes: {IROH_PAIRS.keys()}"
        assert world_size <= len(IROH_PAIRS), (
            f"World size {world_size} is greater than the configured number of known nodes {len(IROH_PAIRS)}"
        )
        os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"] = str(rank), str(4 + rank), str(world_size)
        os.environ["CACHE_DIR"] = os.environ.get("CACHE_DIR", "/workspace")
        if backend == "iroh":
            os.environ["IROH_SEED"] = str(rank)
            os.environ["IROH_PEER_ID"] = IROH_PAIRS[(rank + 1) % world_size]

        # Setup world, logger, comm and load model
        start_setup = perf_counter()
        model, _, prompt_tokens, num_prompt_tokens, batch_size, micro_batch_size, compile_time = setup(
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
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            num_micro_batches=num_micro_batches,
            num_new_tokens=num_new_tokens,
            num_cache_tokens=num_cache_tokens,
            compile=compile,
            latency=latency,
        )
        setup_time = perf_counter() - start_setup

        metrics = []
        for sample_idx in range(num_iterations):
            torch.cuda.synchronize()
            start_generate = perf_counter()
            _, prefill_time, decode_time = generate(
                model=model,
                prompt_tokens=prompt_tokens,
                num_prompt_tokens=num_prompt_tokens,
                num_new_tokens=num_new_tokens,
                num_micro_batches=num_micro_batches,
                micro_batch_size=micro_batch_size,
                disable_tqdm=disable_tqdm or rank != 0,
            )
            generate_time = perf_counter() - start_generate

            # Compute metrics
            generated_tokens = batch_size * num_new_tokens
            throughput = generated_tokens / generate_time

            metrics.append(
                {
                    "rank": rank,
                    "iteration": sample_idx + 1,
                    "setup_time": setup_time,
                    "compile_time": compile_time,
                    "prefill_time": prefill_time,
                    "decode_time": decode_time,
                    "generate_time": generate_time,
                    "generated_tokens": generated_tokens,
                    "throughput": throughput,
                }
            )

        # Send metrics to main process
        queue.put((rank, torch.cuda.get_device_name(), metrics))
    except Exception as e:
        # If an exception occurs, send it to the main process
        queue.put(("ERROR", rank, str(e)))


def main(args: argparse.Namespace) -> None:
    # Prepare static configuration to identify benchmark run
    timestamp = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()[:7],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Get static and dynamic arguments
    dynamic_args = {arg: getattr(args, arg) for arg in vars(args) if isinstance(getattr(args, arg), list)}
    static_config = {arg: getattr(args, arg) for arg in vars(args) if not isinstance(getattr(args, arg), list)}
    print(f"Running benchmark with config {static_config}")

    # Prepare file path
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
        if dynamic_config.get("batch_size") is not None and dynamic_config["batch_size"] < dynamic_config["num_micro_batches"]:
            print("Skipping because batch_size < num_micro_batches")
            continue

        # Run benchmark
        try:
            queue = Queue()
            ps = [
                Process(target=run_benchmark, args=(rank, queue), kwargs={**static_config, **dynamic_config})
                for rank in range(args.world_size)
            ]
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
            result = queue.get()
            if result[0] == "ERROR":
                # If we received an error from a subprocess
                _, rank, error_msg = result
                raise RuntimeError(f"Error in subprocess (rank {rank}): {error_msg}")
            else:
                rank, gpu, metrics = result
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
        metrics_to_aggregate = ["setup_time", "compile_time", "prefill_time", "decode_time", "generate_time", "throughput"]
        aggregated_metrics = {key: mean([result[key] for result in all_metrics[0]]) for key in metrics_to_aggregate}
        aggregated_results.append(aggregated_metrics)

    # Display aggregated results
    headers = ["config_idx"] + list(aggregated_results[0].keys())
    table = [[config_idx] + [f"{r:0.2f}" for r in result.values()] for config_idx, result in enumerate(aggregated_results, start=1)]
    print(tabulate(table, headers=headers))

    print("\nAll benchmarks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Static arguments
    parser.add_argument("--model-name", type=str, default="meta-llama/llama-2-7b-chat-hf", help="HF model name.")
    parser.add_argument("--local-rank", type=int, default=None, help="Sets local rank in CUDA process.")
    parser.add_argument("--world-size", type=int, default=1, help="Number of pipeline stages.")
    parser.add_argument("--num-iterations", type=int, default=3, help="Number of samples to generate.")
    parser.add_argument("--prompt", type=str, default="Hello, my name is", help="Prompt to generate from.")
    parser.add_argument("--num-new-tokens", type=int, default=256, help="Number of tokens to generate.")
    parser.add_argument("--num-cache-tokens", type=int, default=0, help="Number of cache tokens.")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Precision to use for the model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for the model.")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for reproducibility.")
    parser.add_argument("--log-level", type=str, default="CRITICAL", help="Log level.")
    parser.add_argument("--dummy", action="store_true", help="Use dummy weights.")
    parser.add_argument("--save", action="store_true", help="Save results to CSV.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar.")

    # Dynamic arguments
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1], help="Batch size.")
    parser.add_argument("--micro-batch-size", type=int, nargs="+", help="Micro batch size.")
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
        default=["iroh"],
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
