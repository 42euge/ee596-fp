#!/bin/bash
# Setup script for TPU VM environment
# Uses uv to install dependencies from pyproject.toml

set -e  # Exit on error

echo "=========================================="
echo "Setting up TPU VM for GRPO Training"
echo "=========================================="

# Wait for apt locks to be released (unattended-upgrades runs on boot)
echo "[1/5] Waiting for apt locks..."
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "  Waiting for apt lock to be released..."
    sleep 5
done

# Install Python 3.11 and uv
echo "[2/5] Installing Python 3.11 and uv..."
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev git curl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Pin Python version and sync dependencies
echo "[3/5] Installing dependencies with uv..."
cd ~/training
uv python pin 3.11
uv sync

# Verify installation
echo "[4/5] Verifying installation..."
uv run python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
print(f'TPU detected: {any(d.platform == \"tpu\" for d in jax.devices())}')
"

uv run python -c "
import flax
import optax
print(f'Flax version: {flax.__version__}')
print(f'Optax version: {optax.__version__}')
"

uv run python -c "
from tunix.models.gemma3 import model as gemma_lib
print(f'Tunix ModelConfig methods: {[m for m in dir(gemma_lib.ModelConfig) if m.startswith(\"gemma\")]}')
"

# Setup W&B if API key is provided
echo "[5/5] Setting up W&B..."
if [ -n "$WANDB_API_KEY" ]; then
    uv run python -c "import wandb; wandb.login(key='$WANDB_API_KEY')"
    echo "W&B authentication: OK"
else
    echo "WANDB_API_KEY not set, skipping W&B login"
fi

echo ""
echo "=========================================="
echo "TPU VM setup complete!"
echo "=========================================="
