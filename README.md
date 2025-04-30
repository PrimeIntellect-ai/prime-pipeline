<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

---

<p align="center">

<h3 align="center">
PRIME-PIPELINE: Research Sandbox for Decentralized Pipelined Inference
</h3>

---

This is a open-source research repository designed for quickly validating research ideas for pipelined inference over public networks. The codebase initially built upon [GPT-Fast](https://github.com/pytorch-labs/gpt-fast), but has since diverged. Most notably, the codebase implements synchronous and asynchronous communication protocols for pipeline parallel inference.

The codebase has two main entrypoints: 
- `script/generate.py` is used to generate text given model and generation parameters.
- `script/benchmark.py` is used to benchmark performance in varying (artificial) network conditions.


# Usage

## Installation

**Quick Install:** Run the following command for a quick install:


```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-pipeline/refs/heads/main/script/install.sh | bash
```

**Manual Install:** First, install `uv` to build the project.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Then, clone the repository and install the dependencies.

```bash
git clone https://github.com/PrimeIntellect-ai/prime-pipeline.git && cd prime-pipeline
uv sync
```

Also, if you plan to use a private model, you will need to set the `HF_TOKEN` environment variable. Also, we recommend setting the `CACHE_DIR` environment variable to a local directory with enough disk space to store the model weights.

```bash
export CACHE_DIR=<path-to-cache-dir>
export HF_TOKEN=<your-token>
```

## Inference

To check that your installation has succeeded, you can run the following command to generate text with a small model on a single node:

```bash
RANK=0 WORLD_SIZE=1 uv run python script/generate.py
```

Run `uv run python script/generate.py --help` for more information on the available options.


Running distributed inference is as easy as adjusting the environment variables to your setup. For peer-to-peer communication using `iroh`, it's easiest for testing to seed the public key generation required for connecting the nodes. For example, if you have two nodes, set your environment variables as follows:

```bash
# On the first node
export IROH_SEED=0
export IROH_PEER_ID=ff87a0b0a3c7c0ce827e9cada5ff79e75a44a0633bfcb5b50f99307ddb26b337
export RANK=0
export WORLD_SIZE=2
```

```bash
# On the second node
export IROH_SEED=1
export IROH_PEER_ID=ee1aa49a4459dfe813a3cf6eb882041230c7b2558469de81f87c9bf23bf10a03
export RANK=1
export WORLD_SIZE=2
```

Then, run the following command to start the inference (defaults to `meta-llama/Llama-2-7B-chat-hf`):

```bash
uv run python script/generate.py
```

Run `uv run python script/generate.py --help` for more information on the available options.

## Benchmark

To benchmark the inference performance, you can use the `script/benchmark.py` script. It will generate a given number of new tokens from a given prompt and times various aspects of the inference, like startup, prefill, and decode time. Some of the parameters are static, others are dynamic and can be specified as a list. The benchmark will automatically run all combinations of the dynamic parameters and save the benchmark results in the `benchmark` directory under the model name. Repeated benchmark runs will append to the existing results. The benchmark script will not run over the network, but on colocated nodes and simulate network latency (only for the `iroh` backend).

```bash
uv run python script/benchmark.py
```

Run `uv run python script/benchmark.py --help` for more information on the available options.