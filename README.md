# Pipelined GPT-Fast

Private fork of [gpt-fast](https://github.com/pytorch-labs/gpt-fast) for pipelined inference over the Internet.

## Usage

```bash
git clone git@github.com:primeintellect-ai/pipelined-gpt-fast.git && cd pipelined-gpt-fast
```

Install rust

```bash
curl https://sh.rustup.rs -sSf | sh
```

And add to path

```bash
. "$HOME/.cargo/env"
```

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies

```bash
uv sync
```

Set your Hugging Face token if model has restricted access

```bash
export HF_TOKEN=...
```

Download and convert model to required format

```bash
export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
uv run bash scripts/prepare.sh $MODEL_REPO
```

Test inference on single-node

```bash
RANK=0 WORLD_SIZE=1 uv run python src/generate.py
```

Test distributed inference by running in two terminals (tmux sessions recommended)

```bash
RANK=0 WORLD_SIZE=2 uv run python src/generate.py
RANK=1 WORLD_SIZE=2 uv run python src/generate.py
```

To checkout out all generation options, run `uv run python src/generate.py -h`.