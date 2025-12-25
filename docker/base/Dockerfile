# Layer 1: Base dependencies
# Rebuild when: pyproject.toml, uv.lock, tunix/pyproject.toml, TunRex/pyproject.toml change

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Copy submodules (required for editable installs)
COPY tunix/ ./tunix/
COPY TunRex/ ./TunRex/

# Install all dependencies
# uv sync installs jax[tpu]==0.8.1 as specified in pyproject.toml
RUN uv sync --frozen

# Pre-compile Python bytecode for faster startup
RUN uv run python -c "import jax; import flax; import optax; print('JAX deps OK')"
RUN uv run python -c "from tunix.rl.grpo.grpo_learner import GRPOLearner; print('Tunix OK')"
RUN uv run python -c "from tunrex.datasets import get_train_val_test_datasets; print('TunRex OK')"

# Set environment for TPU
ENV JAX_PLATFORMS=tpu
ENV TF_CPP_MIN_LOG_LEVEL=2

LABEL org.opencontainers.image.source="https://github.com/42euge/tunix-hackathon-google"
LABEL layer="base"
