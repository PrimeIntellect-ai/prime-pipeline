# Pipelined GPT-Fast

Private fork of [gpt-fast](https://github.com/pytorch-labs/gpt-fast) for pipelined inference over the Internet.

## Usage

```bash
git clone git@github.com:primeintellect-ai/gpt-fast.git && cd gpt-fast
```

Install dependencies via `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Prepare model for generation
```
export MODEL_REPO=meta-llama/Llama-3.2-3B
export HF_TOKEN=...
uv run bash scripts/prepare.sh $MODEL_REPO
```

Test inference on single-node by generating 200 tokens

```
uv run generate.py \
	--checkpoint_path checkpoints/$MODEL_REPO/model.pth \
	--num_samples 1 \
	--batch_size 1 \
	--compile
```
