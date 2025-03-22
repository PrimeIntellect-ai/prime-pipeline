uv run python -m scripts.download --repo_id $1 && uv run python -m scripts.convert_hf_checkpoint --checkpoint_dir checkpoints/$1
