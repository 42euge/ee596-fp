#!/bin/bash
# Setup script for TPU VM environment
# This script prepares the TPU VM for running GRPO training

set -e  # Exit on error

echo "=========================================="
echo "Setting up TPU VM for GRPO Training"
echo "=========================================="

# Update system packages and install Python 3.11
echo "[1/7] Installing Python 3.11..."
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev git

# Create and activate virtual environment with Python 3.11
echo "[2/7] Creating Python 3.11 virtual environment..."
python3.11 -m venv ~/venv
source ~/venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install JAX with TPU support
echo "[3/7] Installing JAX with TPU support..."
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q

# Install Flax and Optax
echo "[4/7] Installing Flax and Optax..."
pip install flax optax grain-nightly -q

# Install core dependencies
echo "[5/8] Installing project dependencies..."
pip install transformers>=4.40.0 -q
pip install datasets>=2.18.0 -q
pip install huggingface_hub>=0.21.0 -q
pip install safetensors>=0.4.0 -q
pip install tqdm>=4.66.0 -q
pip install wandb>=0.16.0 -q
pip install tensorflow tensorflow_datasets -q

# Install Tunix (Google's training framework)
echo "[6/8] Installing Tunix..."
pip install google-tunix -q

# Install TunRex (local package) if it exists
echo "[7/8] Installing TunRex..."
if [ -d "TunRex" ]; then
    pip install -e TunRex -q
    echo "TunRex installed from local package"
else
    echo "TunRex directory not found, skipping"
fi

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
print(f'TPU detected: {any(d.platform == \"tpu\" for d in jax.devices())}')
"

python3 -c "
import flax
import optax
print(f'Flax version: {flax.__version__}')
print(f'Optax version: {optax.__version__}')
"

python3 -c "
from transformers import AutoTokenizer
print('Transformers: OK')
"

# Setup HuggingFace token if provided
if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo "Setting up HuggingFace authentication..."
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
    echo "HuggingFace authentication: OK"
fi

# Setup W&B if API key is provided
echo ""
echo "[8/8] Setting up W&B..."
if [ -n "$WANDB_API_KEY" ]; then
    python3 -c "import wandb; wandb.login(key='$WANDB_API_KEY')"
    echo "W&B authentication: OK"
else
    echo "WANDB_API_KEY not set, skipping W&B login"
fi

echo ""
echo "=========================================="
echo "TPU VM setup complete!"
echo "=========================================="
