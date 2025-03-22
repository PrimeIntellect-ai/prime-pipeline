#!/bin/bash

# Capture both stdout and stderr while also printing to terminal
RANK=0 WORLD_SIZE=1 uv run python src/pipelined_gpt_fast/generate.py 