FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Copy submodules
COPY tunix ./tunix
COPY TunRex ./TunRex

# Install dependencies
RUN uv sync --frozen

# Copy the rest of the code
COPY . .

# Default command
CMD ["uv", "run", "python", "scripts/train_grpo.py", "--help"]
