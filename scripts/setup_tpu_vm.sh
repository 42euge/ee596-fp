#!/bin/bash
# Setup script for TPU VM environment
# This script prepares the TPU VM for running GRPO training

set -e  # Exit on error

echo "=========================================="
echo "Setting up TPU VM for GRPO Training"
echo "=========================================="

# Update system packages
echo "[1/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv git

# Create and activate virtual environment
echo "[2/6] Creating Python virtual environment..."
python3 -m venv ~/venv
source ~/venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install JAX with TPU support
echo "[3/6] Installing JAX with TPU support..."
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q

# Install Flax and Optax
echo "[4/6] Installing Flax and Optax..."
pip install flax optax grain-nightly -q

# Install core dependencies
echo "[5/6] Installing project dependencies..."
pip install transformers>=4.40.0 -q
pip install datasets>=2.18.0 -q
pip install huggingface_hub>=0.21.0 -q
pip install safetensors>=0.4.0 -q
pip install tqdm>=4.66.0 -q

# Install Tunix (Google's training framework)
echo "[6/6] Installing Tunix..."
pip install google-tunix -q

# Install TunRex (local package) if it exists
if [ -d "TunRex" ]; then
    echo "Installing TunRex local package..."
    pip install -e TunRex -q
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

echo ""
echo "=========================================="
echo "TPU VM setup complete!"
echo "=========================================="
