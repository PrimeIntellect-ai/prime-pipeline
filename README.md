<p align="center">
</p>

<img src="https://github.com/user-attachments/assets/51e44795-5206-49d6-a12a-ecacd2799df2" alt="Prime Intellect" style="width: 100%; height: auto;"/>

---

<p align="center">

<h3 align="center">
[WIP]: Pipelined Inference over the Internet
</h3>
<p align="center">
| <a href=""><b>Blog</b></a> | <a href=""><b>X Thread</b></a> |
</p>

---

This is a library for pipelined inference over the Internet. The repository is still under active development.

The library has two main entrypoints: 
- `src/infer.py` is used to generate text using a given model and generation parameters.
- `src/benchmark.py` is used to benchmark the inference performance in various configurations.


# Usage

## Installation

**Quick Install:** Run the following command for a quick install:


```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/pipelined-gpt-fast/refs/heads/main/script/install.sh | bash
```

**Manual Install:** First, install `uv` and `cargo` to build the project.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

```bash
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
```

Then, clone the repository and install the dependencies.

```bash
git clone git@github.com:PrimeIntellect-ai/pipelined-gpt-fast.git && cd pipelined-gpt-fast
uv sync
```

Also, if you plan to use a private model, you will need to set the `HF_TOKEN` environment variable. To do this, run 

```bash
export HF_TOKEN=<your-token>
```

## Inference

To check that your installation has succeeded, you can run the following command to generate text with a small model on a single node:

```bash
RANK=0 WORLD_SIZE=1 uv run python src/infer.py
```

Run `uv run python src/infer.py --help` for more information on the available options.


If you want to run distributed inference, adjust your environment variables to your setup. For example, if you have two nodes, you can run the following command:

```bash
RANK=0 WORLD_SIZE=2 uv run python src/infer.py # On the first node
RANK=1 WORLD_SIZE=2 uv run python src/infer.py # On the second node
```


## Benchmark

To benchmark the inference performance, you can use the `src/benchmark.py` script. It will generate a given number of new tokens from a given prompt and time various aspects of the inference, like startup, prefill, and decode time. Some of the parameters are static, others are dynamic and can be specified as a list. The benchmark will automatically run all combinations of the dynamic parameters and save the benchmark results in the `benchmark` directory under the model name. Repeated benchmark runs will append to the existing results. The benchmark script will not run over the network, but on colocated nodes and simulate network latency (only for the `iroh` backend).

```bash
uv run python src/benchmark.py
```

Run `uv run python src/benchmark.py --help` for more information on the available options.