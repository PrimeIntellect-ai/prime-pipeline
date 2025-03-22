#!/usr/bin/env bash

set -e

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

main() {

    log_info "Cloning repository..."
    git clone https://github.com/primeintellect-ai/pipelined-gpt-fast.git
    
    log_info "Entering project directory..."
    cd pipelined-gpt-fast
    
    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi
    
    log_info "Creating virtual environment..."
    uv venv
    
    log_info "Activating virtual environment..."
    source .venv/bin/activate
    
    log_info "Installing dependencies..."
    uv sync
        
    log_info "Installation completed! You can double check that everything is install correctly by running 'RANK=0 WORLD_SIZE=1 uv run python src/pipelined_gpt_fast/generate.py"
}

main