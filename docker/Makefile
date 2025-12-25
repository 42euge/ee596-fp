# Layered Docker Build System for GRPO Training
#
# Usage:
#   make base          - Build Layer 1 (dependencies)
#   make data          - Build Layer 2 (models + datasets)
#   make training      - Build Layer 3 (training runtime)
#   make all           - Build all layers
#   make push-all      - Push all images to registry
#
# Configuration:
#   MODELS=gemma-3-1b-it    - Models to pre-download (gemma-3-1b-it, gemma-3-270m, both)
#   DATASET_SOURCE=tfds     - Dataset source (tfds, huggingface, kaggle)
#   TAG=latest              - Image tag

REGISTRY := us-central1-docker.pkg.dev
PROJECT := kaggle-euge
REPOSITORY := tunix-training

BASE_IMAGE := $(REGISTRY)/$(PROJECT)/$(REPOSITORY)/grpo-base
DATA_IMAGE := $(REGISTRY)/$(PROJECT)/$(REPOSITORY)/grpo-data
TRAINING_IMAGE := $(REGISTRY)/$(PROJECT)/$(REPOSITORY)/grpo-trainer

# Build arguments
MODELS ?= gemma-3-1b-it
DATASET_SOURCE ?= tfds
TAG ?= latest

# Git SHA for versioning
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Root directory (one level up from docker/)
ROOT_DIR := $(shell cd .. && pwd)

.PHONY: all base data training push-all push-base push-data push-training clean login help run-local

help:
	@echo "Layered Docker Build System"
	@echo ""
	@echo "Targets:"
	@echo "  base      - Build Layer 1 (dependencies)"
	@echo "  data      - Build Layer 2 (models + datasets)"
	@echo "  training  - Build Layer 3 (training runtime)"
	@echo "  all       - Build all layers"
	@echo "  push-all  - Push all images to registry"
	@echo "  clean     - Remove local images"
	@echo "  run-local - Run training locally (for testing)"
	@echo ""
	@echo "Build arguments:"
	@echo "  MODELS=$(MODELS)"
	@echo "  DATASET_SOURCE=$(DATASET_SOURCE)"
	@echo "  TAG=$(TAG)"
	@echo ""
	@echo "Images:"
	@echo "  Base:     $(BASE_IMAGE):$(TAG)"
	@echo "  Data:     $(DATA_IMAGE):$(TAG)"
	@echo "  Training: $(TRAINING_IMAGE):$(TAG)"

login:
	gcloud auth configure-docker $(REGISTRY) --quiet

# Layer 1: Base dependencies
base:
	@echo "========================================"
	@echo "Building Layer 1: Base (dependencies)"
	@echo "========================================"
	cd $(ROOT_DIR) && docker build \
		-f docker/base/Dockerfile \
		-t $(BASE_IMAGE):$(TAG) \
		-t $(BASE_IMAGE):$(GIT_SHA) \
		.

# Layer 2: Data and models
data:
	@echo "========================================"
	@echo "Building Layer 2: Data (models + datasets)"
	@echo "  MODELS=$(MODELS)"
	@echo "  DATASET_SOURCE=$(DATASET_SOURCE)"
	@echo "========================================"
	cd $(ROOT_DIR) && docker build \
		-f docker/data/Dockerfile \
		--build-arg BASE_IMAGE=$(BASE_IMAGE):$(TAG) \
		--build-arg MODELS=$(MODELS) \
		--build-arg DATASET_SOURCE=$(DATASET_SOURCE) \
		--build-arg HF_TOKEN=$(HF_TOKEN) \
		-t $(DATA_IMAGE):$(TAG) \
		-t $(DATA_IMAGE):$(GIT_SHA) \
		-t $(DATA_IMAGE):$(MODELS)-$(DATASET_SOURCE) \
		.

# Layer 3: Training runtime
training:
	@echo "========================================"
	@echo "Building Layer 3: Training"
	@echo "========================================"
	cd $(ROOT_DIR) && docker build \
		-f docker/training/Dockerfile \
		--build-arg DATA_IMAGE=$(DATA_IMAGE):$(TAG) \
		-t $(TRAINING_IMAGE):$(TAG) \
		-t $(TRAINING_IMAGE):$(GIT_SHA) \
		.

# Build all layers
all: base data training

# Push all images
push-all: login push-base push-data push-training

push-base: login
	docker push $(BASE_IMAGE):$(TAG)
	docker push $(BASE_IMAGE):$(GIT_SHA)

push-data: login
	docker push $(DATA_IMAGE):$(TAG)
	docker push $(DATA_IMAGE):$(GIT_SHA)
	docker push $(DATA_IMAGE):$(MODELS)-$(DATASET_SOURCE)

push-training: login
	docker push $(TRAINING_IMAGE):$(TAG)
	docker push $(TRAINING_IMAGE):$(GIT_SHA)

# Clean up local images
clean:
	docker rmi $(TRAINING_IMAGE):$(TAG) $(TRAINING_IMAGE):$(GIT_SHA) 2>/dev/null || true
	docker rmi $(DATA_IMAGE):$(TAG) $(DATA_IMAGE):$(GIT_SHA) 2>/dev/null || true
	docker rmi $(BASE_IMAGE):$(TAG) $(BASE_IMAGE):$(GIT_SHA) 2>/dev/null || true

# Run training locally (for testing)
run-local:
	docker run --rm --privileged \
		-e HF_TOKEN=$(HF_TOKEN) \
		-e WANDB_API_KEY=$(WANDB_API_KEY) \
		-e NUM_STEPS=10 \
		-e NO_WANDB=true \
		$(TRAINING_IMAGE):$(TAG)

# Build with both models
build-full:
	$(MAKE) all MODELS=both

# Quick rebuild of training layer only (for code changes)
quick:
	$(MAKE) training
