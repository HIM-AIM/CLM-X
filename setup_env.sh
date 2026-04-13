#!/bin/bash
# install uv
if ! command -v uv &> /dev/null
then
    echo "uv Not installed, installing now..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Clean up old environment (optional)
rm -rf .venv uv.lock

# Create new env
uv venv
source .venv/bin/activate

# Synchronize dependencie
uv sync 

# Install flash-attn
uv pip install flash-attn==2.5.8 --no-build-isolation
echo "Env done!type .venv/bin/activate to activate env"