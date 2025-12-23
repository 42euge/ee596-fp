# Makefile for Reward Model Development Pipeline
# Provides convenient shortcuts for common tasks

.PHONY: help install setup test clean format lint train evaluate deploy monitor

# Default target
help:
	@echo "Reward Model Development Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  install         Install all dependencies"
	@echo "  setup           Set up development environment (pre-commit hooks, etc.)"
	@echo "  test            Run all tests"
	@echo "  format          Format code with black and isort"
	@echo "  lint            Run linters (flake8, mypy)"
	@echo "  clean           Clean generated files"
	@echo ""
	@echo "Dataset Pipeline:"
	@echo "  prepare-gsm8k   Prepare GSM8K dataset"
	@echo "  prepare-openrubrics  Prepare OpenRubrics dataset"
	@echo ""
	@echo "Training Pipeline:"
	@echo "  train           Start training (local)"
	@echo "  train-small     Quick validation training run"
	@echo ""
	@echo "Evaluation Pipeline:"
	@echo "  evaluate        Evaluate model on test set"
	@echo "  evaluate-quick  Quick evaluation (100 samples)"
	@echo ""
	@echo "Deployment Pipeline:"
	@echo "  deploy          Deploy checkpoint to HuggingFace Hub"
	@echo ""
	@echo "Monitoring:"
	@echo "  monitor         Show training metrics dashboard"
	@echo "  logs            Tail latest training log"
	@echo ""
	@echo "Use 'make <target>' to run a specific task"

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e TunRex
	@echo "Dependencies installed!"

install-dev: install
	@echo "Installing development dependencies..."
	pip install pre-commit pytest black isort flake8 mypy bandit
	@echo "Development dependencies installed!"

# Setup
setup: install-dev
	@echo "Setting up development environment..."
	pre-commit install
	chmod +x scripts/*.py scripts/*.sh
	mkdir -p checkpoints logs data
	@echo "Development environment ready!"

# Testing
test:
	@echo "Running tests..."
	python -m pytest TunRex/tests/ -v --tb=short

test-quick:
	@echo "Running quick tests..."
	python -m pytest TunRex/tests/test_imports.py -v

# Code quality
format:
	@echo "Formatting code..."
	black src/ scripts/ --line-length=100 --exclude TunRex/
	isort src/ scripts/ --profile=black --line-length=100 --skip TunRex
	@echo "Code formatted!"

lint:
	@echo "Running linters..."
	flake8 src/ scripts/ --max-line-length=100 --extend-ignore=E203,E501,W503 --exclude=TunRex/
	mypy src/ --ignore-missing-imports --no-strict-optional
	@echo "Linting complete!"

lint-security:
	@echo "Running security checks..."
	bandit -r src/ scripts/ -ll
	@echo "Security check complete!"

# Cleaning
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

clean-all: clean
	@echo "Removing all generated data..."
	rm -rf logs/*.log logs/*.json
	rm -rf data/*/
	@echo "All generated data removed!"

# Dataset Pipeline
prepare-gsm8k:
	@echo "Preparing GSM8K dataset..."
	python scripts/prepare_dataset.py \
		--config gsm8k \
		--output-dir data/gsm8k \
		--validate

prepare-openrubrics:
	@echo "Preparing OpenRubrics dataset..."
	python scripts/prepare_dataset.py \
		--config openrubrics \
		--output-dir data/openrubrics \
		--validate

# Training Pipeline
train:
	@echo "Starting training..."
	python scripts/reward_pipeline.py train \
		--steps 100 \
		--dataset gsm8k \
		--wandb-project reward-model-dev

train-small:
	@echo "Running quick training validation..."
	python scripts/train_small.py

train-full:
	@echo "Starting full training run..."
	python scripts/reward_pipeline.py train \
		--steps 1000 \
		--lr 3e-6 \
		--batch-size 1 \
		--dataset gsm8k \
		--wandb-project reward-model-dev

# Evaluation Pipeline
evaluate:
	@echo "Running full evaluation..."
	python scripts/reward_pipeline.py evaluate \
		--dataset gsm8k \
		--split test \
		--output logs/eval_$(shell date +%Y%m%d_%H%M%S).json

evaluate-quick:
	@echo "Running quick evaluation (100 samples)..."
	python scripts/reward_pipeline.py evaluate \
		--dataset gsm8k \
		--split test \
		--num-samples 100 \
		--output logs/eval_quick_$(shell date +%Y%m%d_%H%M%S).json

# Deployment Pipeline
deploy:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT path required. Usage: make deploy CHECKPOINT=./checkpoints/step_1000"; \
		exit 1; \
	fi
	@if [ -z "$(REPO_ID)" ]; then \
		echo "Error: REPO_ID required. Usage: make deploy CHECKPOINT=./checkpoints/step_1000 REPO_ID=username/model"; \
		exit 1; \
	fi
	@echo "Deploying checkpoint $(CHECKPOINT) to $(REPO_ID)..."
	python scripts/reward_pipeline.py deploy \
		--checkpoint $(CHECKPOINT) \
		--repo-id $(REPO_ID)

deploy-private:
	@echo "Deploying checkpoint (private repository)..."
	python scripts/reward_pipeline.py deploy \
		--checkpoint $(CHECKPOINT) \
		--repo-id $(REPO_ID) \
		--private

# Monitoring
monitor:
	@if [ -z "$(RUN)" ]; then \
		echo "Available runs:"; \
		ls -1 logs/*.log 2>/dev/null | xargs -n1 basename -s .log || echo "No logs found"; \
		echo ""; \
		echo "Usage: make monitor RUN=run_name"; \
	else \
		python scripts/monitor_training.py --run-name $(RUN); \
	fi

monitor-compare:
	@if [ -z "$(RUNS)" ]; then \
		echo "Error: RUNS required. Usage: make monitor-compare RUNS='run1 run2 run3'"; \
		exit 1; \
	fi
	python scripts/monitor_training.py --compare $(RUNS)

logs:
	@LATEST_LOG=$$(ls -t logs/*.log 2>/dev/null | head -1); \
	if [ -n "$$LATEST_LOG" ]; then \
		echo "Tailing $$LATEST_LOG..."; \
		tail -f $$LATEST_LOG; \
	else \
		echo "No log files found in logs/"; \
	fi

# CI/CD
ci-test:
	@echo "Running CI tests..."
	make test-quick
	make lint
	@echo "CI tests passed!"

ci-full:
	@echo "Running full CI suite..."
	make test
	make lint
	make lint-security
	@echo "Full CI suite passed!"

# Documentation
docs:
	@echo "Opening documentation..."
	@if command -v open >/dev/null 2>&1; then \
		open docs/CICD_SETUP.md; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open docs/CICD_SETUP.md; \
	else \
		echo "Documentation available in docs/"; \
	fi

# Quick start
quickstart: setup prepare-gsm8k
	@echo ""
	@echo "✓ Quick start complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run a quick training test: make train-small"
	@echo "  2. Start training: make train"
	@echo "  3. Evaluate model: make evaluate-quick"
	@echo "  4. Monitor training: make monitor RUN=<run_name>"
	@echo ""
	@echo "For more options, run: make help"

# Development workflow
dev: format lint test
	@echo "✓ Development checks passed!"

# Pre-commit check (run before committing)
pre-commit: format lint test-quick
	@echo "✓ Pre-commit checks passed! Ready to commit."

# Show current status
status:
	@echo "Reward Model Development Status"
	@echo "=============================="
	@echo ""
	@echo "Checkpoints: $$(find checkpoints -type f 2>/dev/null | wc -l) files"
	@echo "Logs: $$(find logs -name '*.log' 2>/dev/null | wc -l) training runs"
	@echo "Datasets: $$(find data -type d -mindepth 1 -maxdepth 1 2>/dev/null | wc -l) prepared"
	@echo ""
	@if [ -d .git ]; then \
		echo "Git branch: $$(git branch --show-current)"; \
		echo "Git status: $$(git status --short | wc -l) modified files"; \
	fi
	@echo ""
	@echo "Recent logs:"
	@ls -t logs/*.log 2>/dev/null | head -5 | xargs -n1 basename -s .log || echo "  No logs found"

# Version info
version:
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Project root: $$(pwd)"
	@if command -v git >/dev/null 2>&1 && [ -d .git ]; then \
		echo "Git commit: $$(git rev-parse --short HEAD)"; \
	fi
