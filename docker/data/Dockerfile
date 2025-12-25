# Layer 2: Pre-downloaded models and datasets
# Rebuild when: scripts/model_utils.py, TunRex/src/tunrex/datasets/loaders.py change

ARG BASE_IMAGE=us-central1-docker.pkg.dev/kaggle-euge/tunix-training/grpo-base:latest
FROM ${BASE_IMAGE}

# Build arguments for configurability
ARG MODELS=gemma-3-1b-it
ARG DATASET_SOURCE=tfds
ARG HF_TOKEN=""

ENV MODELS=${MODELS}
ENV DATASET_SOURCE=${DATASET_SOURCE}
ENV HF_TOKEN=${HF_TOKEN}

WORKDIR /app

# Copy the download script
COPY docker/data/download_assets.py ./docker/data/

# Copy model_utils.py for download_model function
COPY scripts/model_utils.py ./scripts/

# Copy dataset loaders (already in base, but ensure latest)
COPY TunRex/src/tunrex/datasets/ ./TunRex/src/tunrex/datasets/
COPY scripts/training_config.py ./scripts/

# Create cache directories
RUN mkdir -p /app/models /app/data/train /app/data/test

# Download models and datasets at build time
RUN uv run python docker/data/download_assets.py \
    --models "${MODELS}" \
    --dataset-source "${DATASET_SOURCE}"

# Store metadata about downloaded assets
RUN echo "MODELS=${MODELS}" > /app/.asset_metadata && \
    echo "DATASET_SOURCE=${DATASET_SOURCE}" >> /app/.asset_metadata && \
    echo "BUILD_DATE=$(date -Iseconds)" >> /app/.asset_metadata

LABEL layer="data"
LABEL models="${MODELS}"
LABEL dataset_source="${DATASET_SOURCE}"
