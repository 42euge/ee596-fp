# Experiment Tracking and Evaluation Framework Design

## Executive Summary

This document outlines the design for a comprehensive experiment tracking and evaluation framework for the Gemma3-1B Reasoning Model project. The design addresses current gaps in experiment management, result comparison, and standardized evaluation.

## Current State Analysis

### Existing Tracking Systems
- **W&B Integration**: Good for real-time metrics, but lacks structured experiment metadata
- **Scattered Logging**: Multiple JSON outputs, notebook cells, manual result recording
- **No Experiment Comparison**: Difficult to compare runs across different configurations
- **Limited Reproducibility**: Hyperparameters tracked but not easily reproducible

### Existing Evaluation Systems
- **TunRex Evaluation**: Good for format/accuracy metrics, but dataset-specific
- **Manual GSM8K Testing**: Requires separate script execution, results not centralized
- **Multiple Reward Functions**: 7+ reward functions with no systematic comparison framework
- **No Benchmark Suite**: Each evaluation is ad-hoc

## Design Goals

1. **Centralized Experiment Management**: Single source of truth for all experiments
2. **Reproducibility**: One-command experiment reproduction from tracked config
3. **Comparison Tools**: Easy comparison of hyperparameters, metrics, and results
4. **Extensible Benchmarks**: Plugin architecture for adding new evaluation datasets
5. **Backward Compatible**: Works with existing W&B and training scripts
6. **Minimal Overhead**: < 5% computational overhead for tracking

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Tracker                        │
│  - Unique experiment IDs                                     │
│  - Configuration versioning                                  │
│  - Metadata collection (git hash, timestamp, environment)    │
│  - Result aggregation                                        │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ├─────────────────────────────────────┐
                   │                                     │
         ┌─────────▼──────────┐              ┌─────────▼──────────┐
         │   W&B Backend      │              │  Local Backend     │
         │  - Real-time plots │              │  - SQLite DB       │
         │  - Cloud storage   │              │  - JSON export     │
         └────────────────────┘              │  - CSV reports     │
                                             └────────────────────┘
                   │
         ┌─────────▼──────────────────────────────────────────────┐
         │              Evaluation Framework                      │
         │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
         │  │   GSM8K      │  │   MATH       │  │   Custom     │ │
         │  │  Benchmark   │  │  Benchmark   │  │  Datasets    │ │
         │  └──────────────┘  └──────────────┘  └──────────────┘ │
         │                                                        │
         │  ┌──────────────────────────────────────────────────┐ │
         │  │           Unified Metrics Collection             │ │
         │  │  - Accuracy (exact, partial, format)             │ │
         │  │  - Reasoning quality scores                      │ │
         │  │  - Generation efficiency                         │ │
         │  │  - Consistency across samples                    │ │
         │  └──────────────────────────────────────────────────┘ │
         └────────────────────────────────────────────────────────┘
                   │
         ┌─────────▼──────────────────────────────────────────────┐
         │           Analysis & Comparison Tools                  │
         │  - Experiment leaderboard                              │
         │  - Hyperparameter impact analysis                      │
         │  - Statistical significance testing                    │
         │  - Result visualization                                │
         └────────────────────────────────────────────────────────┘
```

## Component 1: Centralized Experiment Tracker

### Features

#### 1.1 Experiment Metadata
Every experiment automatically records:
```python
{
    "experiment_id": "exp_20250123_143022_a1b2c3",  # Timestamp + git hash
    "git_commit": "a1b2c3d4e5f6...",
    "git_branch": "claude/experiment-tracking-framework-zriqo",
    "git_dirty": false,  # Uncommitted changes
    "timestamp": "2025-01-23T14:30:22Z",
    "user": "system",
    "hostname": "tpu-vm-01",
    "environment": {
        "python_version": "3.10.12",
        "jax_version": "0.4.23",
        "device": "TPU v2-8",
        "cuda_version": null
    }
}
```

#### 1.2 Configuration Tracking
Complete hyperparameter snapshot:
```python
{
    "model": {
        "base_model": "google/gemma-3-1b-it",
        "use_lora": true,
        "lora_rank": 8,
        "lora_alpha": 16
    },
    "training": {
        "num_steps": 500,
        "learning_rate": 5e-5,
        "batch_size": 64,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "warmup_fraction": 0.1
    },
    "grpo": {
        "num_generations": 4,
        "beta": 0.1,
        "epsilon": 0.2,
        "temperature": 0.7
    },
    "generation": {
        "max_prompt_length": 1024,
        "max_generation_steps": 512,
        "top_k": 50,
        "top_p": 0.9
    },
    "dataset": {
        "name": "openrubrics",
        "train_size": 8000,
        "eval_size": 1000,
        "seed": 42
    },
    "rewards": {
        "format_weight": 1.0,
        "accuracy_weight": 2.0,
        "rubric_weight": 0.5
    }
}
```

#### 1.3 Result Aggregation
Structured storage of all metrics:
```python
{
    "training_metrics": {
        "final_loss": 0.234,
        "final_reward": 4.56,
        "gradient_norm_avg": 1.23,
        "learning_rate_final": 1e-6,
        "steps_completed": 500,
        "time_elapsed_seconds": 3600
    },
    "evaluation_metrics": {
        "gsm8k": {
            "accuracy": 0.73,
            "partial_accuracy": 0.81,
            "format_accuracy": 0.95,
            "avg_generation_time": 1.2,
            "num_samples": 1000
        },
        "math": {
            "accuracy": 0.45,
            "partial_accuracy": 0.52,
            "format_accuracy": 0.93,
            "num_samples": 500
        }
    },
    "checkpoint_paths": [
        "gs://bucket/checkpoints/exp_20250123_143022_a1b2c3/step_500"
    ]
}
```

### Implementation

**File**: `/home/user/ee596-fp/src/experiment_tracker.py`

Key classes:
- `ExperimentTracker`: Main tracking interface
- `ExperimentConfig`: Configuration management with validation
- `ExperimentResult`: Result storage and querying
- `ExperimentBackend`: Abstract backend (W&B, Local, etc.)

## Component 2: Unified Evaluation Framework

### Features

#### 2.1 Benchmark Registry
Plugin-based benchmark system:

```python
class BenchmarkRegistry:
    """Registry for evaluation benchmarks"""

    benchmarks = {
        "gsm8k": GSM8KBenchmark,
        "math": MATHBenchmark,
        "openrubrics": OpenRubricsBenchmark,
        "custom": CustomBenchmark
    }

    @classmethod
    def register(cls, name, benchmark_class):
        """Register a new benchmark"""
        cls.benchmarks[name] = benchmark_class

    @classmethod
    def evaluate(cls, model, benchmark_name, **kwargs):
        """Run evaluation on specified benchmark"""
        benchmark = cls.benchmarks[benchmark_name]()
        return benchmark.evaluate(model, **kwargs)
```

#### 2.2 Standardized Metrics
All benchmarks return consistent metric structure:

```python
{
    "benchmark_name": "gsm8k",
    "num_samples": 1000,
    "metrics": {
        # Core accuracy metrics
        "accuracy": 0.73,
        "partial_accuracy": 0.81,  # Within tolerance
        "format_accuracy": 0.95,   # Proper tags

        # Reasoning quality
        "avg_reasoning_length": 245.3,  # tokens
        "reasoning_coherence": 0.87,    # Future: LLM-as-judge

        # Efficiency
        "avg_generation_time": 1.2,     # seconds
        "avg_tokens_generated": 312.5,

        # Consistency (multi-sample)
        "self_consistency": 0.68,       # Agreement across samples

        # Error analysis
        "error_types": {
            "format_error": 0.05,
            "calculation_error": 0.12,
            "reasoning_error": 0.10
        }
    },
    "per_sample_results": [
        {
            "question_id": "gsm8k_001",
            "question": "...",
            "gold_answer": 42,
            "predicted_answer": 42,
            "is_correct": true,
            "format_correct": true,
            "reasoning": "...",
            "generation_time": 1.1
        },
        # ... more samples
    ]
}
```

#### 2.3 Reward Function Comparison
Systematic comparison of reward functions:

```python
class RewardComparator:
    """Compare different reward function strategies"""

    def compare_rewards(self, samples, reward_functions):
        """
        Compare multiple reward functions on same samples

        Returns:
            {
                "reward_function_name": {
                    "mean_reward": 2.34,
                    "std_reward": 1.12,
                    "correlation_with_accuracy": 0.87,
                    "range": [-2.5, 5.0]
                },
                ...
            }
        """
        pass
```

### Implementation

**Files**:
- `/home/user/ee596-fp/src/evaluation/benchmark_registry.py`
- `/home/user/ee596-fp/src/evaluation/benchmarks/base.py`
- `/home/user/ee596-fp/src/evaluation/benchmarks/gsm8k.py`
- `/home/user/ee596-fp/src/evaluation/benchmarks/math.py`
- `/home/user/ee596-fp/src/evaluation/metrics.py`
- `/home/user/ee596-fp/src/evaluation/reward_comparison.py`

## Component 3: Analysis & Comparison Tools

### Features

#### 3.1 Experiment Leaderboard
```bash
$ python -m src.analysis.leaderboard --benchmark gsm8k --top 10

╔═══════════════════════════════════════════════════════════════════════════╗
║                         GSM8K Leaderboard (Top 10)                        ║
╠═══════╦════════════════╦══════════╦════════════╦═════════════╦═══════════╣
║ Rank  ║ Experiment ID  ║ Accuracy ║ Partial    ║ Format      ║ Date      ║
╠═══════╬════════════════╬══════════╬════════════╬═════════════╬═══════════╣
║   1   ║ exp_..._a1b2c3 ║  0.758   ║   0.832    ║    0.961    ║ 2025-01-20║
║   2   ║ exp_..._d4e5f6 ║  0.742   ║   0.818    ║    0.953    ║ 2025-01-19║
║   3   ║ exp_..._g7h8i9 ║  0.731   ║   0.809    ║    0.947    ║ 2025-01-18║
╚═══════╩════════════════╩══════════╩════════════╩═════════════╩═══════════╝
```

#### 3.2 Hyperparameter Impact Analysis
```bash
$ python -m src.analysis.hyperparam_impact --metric accuracy --top 5

Top 5 Hyperparameters Impacting Accuracy:
1. learning_rate: correlation = 0.67 (p < 0.001)
2. lora_rank: correlation = 0.54 (p < 0.01)
3. num_generations: correlation = 0.48 (p < 0.01)
4. beta (GRPO): correlation = -0.41 (p < 0.05)
5. batch_size: correlation = 0.32 (p < 0.05)
```

#### 3.3 Experiment Comparison
```bash
$ python -m src.analysis.compare exp_001 exp_002 exp_003

Comparing 3 experiments:

Configuration Differences:
┌─────────────────┬────────────┬────────────┬────────────┐
│ Hyperparameter  │ exp_001    │ exp_002    │ exp_003    │
├─────────────────┼────────────┼────────────┼────────────┤
│ learning_rate   │ 5e-5       │ 1e-4       │ 5e-5       │
│ lora_rank       │ 8          │ 8          │ 16         │
│ num_generations │ 4          │ 8          │ 4          │
└─────────────────┴────────────┴────────────┴────────────┘

Performance Comparison (GSM8K):
┌─────────────────┬────────────┬────────────┬────────────┐
│ Metric          │ exp_001    │ exp_002    │ exp_003    │
├─────────────────┼────────────┼────────────┼────────────┤
│ Accuracy        │ 0.731      │ 0.698      │ 0.758 ⭐   │
│ Partial Acc.    │ 0.809      │ 0.772      │ 0.832 ⭐   │
│ Format Acc.     │ 0.947      │ 0.934      │ 0.961 ⭐   │
│ Training Time   │ 3600s      │ 4200s      │ 3650s      │
└─────────────────┴────────────┴────────────┴────────────┘

Winner: exp_003 (2.7% improvement over exp_001)
```

#### 3.4 Statistical Significance Testing
```python
from src.analysis.statistics import compare_experiments

result = compare_experiments(
    "exp_001", "exp_002",
    metric="accuracy",
    test="bootstrap",  # or "t-test", "permutation"
    n_bootstrap=10000
)

print(result)
# {
#     "difference": 0.027,
#     "confidence_interval_95": [0.012, 0.041],
#     "p_value": 0.003,
#     "is_significant": True,
#     "effect_size": "medium"
# }
```

### Implementation

**Files**:
- `/home/user/ee596-fp/src/analysis/leaderboard.py`
- `/home/user/ee596-fp/src/analysis/hyperparam_impact.py`
- `/home/user/ee596-fp/src/analysis/compare.py`
- `/home/user/ee596-fp/src/analysis/statistics.py`
- `/home/user/ee596-fp/src/analysis/visualizations.py`

## Component 4: Database Schema

### SQLite Schema (Local Backend)

```sql
-- Experiments table
CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    git_commit TEXT NOT NULL,
    git_branch TEXT NOT NULL,
    git_dirty BOOLEAN NOT NULL,
    timestamp DATETIME NOT NULL,
    user TEXT,
    hostname TEXT,
    config_json TEXT NOT NULL,  -- Full JSON config
    status TEXT DEFAULT 'running',  -- running, completed, failed
    notes TEXT
);

-- Training metrics (time series)
CREATE TABLE training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Evaluation results
CREATE TABLE evaluation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    benchmark_name TEXT NOT NULL,
    metrics_json TEXT NOT NULL,  -- Full metrics JSON
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Per-sample results
CREATE TABLE sample_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL,
    sample_id TEXT NOT NULL,
    question TEXT,
    gold_answer TEXT,
    predicted_answer TEXT,
    is_correct BOOLEAN,
    format_correct BOOLEAN,
    reasoning TEXT,
    generation_time REAL,
    FOREIGN KEY (evaluation_id) REFERENCES evaluation_results(id)
);

-- Checkpoints
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER,
    timestamp DATETIME NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

-- Indexes for fast queries
CREATE INDEX idx_experiments_timestamp ON experiments(timestamp);
CREATE INDEX idx_training_metrics_exp ON training_metrics(experiment_id, step);
CREATE INDEX idx_evaluation_exp_benchmark ON evaluation_results(experiment_id, benchmark_name);
```

## Component 5: Integration with Existing Systems

### 5.1 W&B Integration
```python
class WandBBackend(ExperimentBackend):
    """Use W&B as backend while maintaining local tracking"""

    def __init__(self, project_name="tunix-grpo"):
        self.project_name = project_name

    def log_experiment(self, experiment_id, config):
        """Initialize W&B run with experiment ID"""
        import wandb
        wandb.init(
            project=self.project_name,
            name=experiment_id,
            config=config,
            tags=[config.get("git_branch"), config.get("model.base_model")]
        )

    def log_metrics(self, metrics, step):
        """Log metrics to W&B"""
        import wandb
        wandb.log(metrics, step=step)
```

### 5.2 Training Script Integration
```python
# In scripts/train_grpo.py

from src.experiment_tracker import ExperimentTracker

def main():
    # Initialize tracker
    tracker = ExperimentTracker(
        backends=["wandb", "local"],
        db_path="experiments.db"
    )

    # Start experiment
    experiment_id = tracker.start_experiment(
        config=config,
        notes="Testing higher learning rate"
    )

    # Training loop
    for step in range(num_steps):
        # ... training code ...

        # Log metrics
        tracker.log_metrics({
            "loss": loss,
            "reward": reward,
            "learning_rate": lr
        }, step=step)

        # Save checkpoint
        if step % 50 == 0:
            checkpoint_path = save_checkpoint(...)
            tracker.log_checkpoint(checkpoint_path, step)

    # Run evaluation
    results = evaluate_all_benchmarks(model)
    tracker.log_evaluation(results)

    # Finish experiment
    tracker.finish_experiment(status="completed")
```

## Usage Examples

### Running an Experiment
```bash
# Start training with automatic tracking
python scripts/train_grpo.py \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --num_steps 500 \
    --experiment_name "high_lr_test" \
    --notes "Testing impact of increased learning rate"

# Experiment ID: exp_20250123_143022_a1b2c3
```

### Evaluating a Checkpoint
```bash
# Run all benchmarks
python -m src.evaluation.run_benchmarks \
    --checkpoint gs://bucket/checkpoints/exp_20250123_143022_a1b2c3/step_500 \
    --benchmarks gsm8k,math \
    --experiment_id exp_20250123_143022_a1b2c3

# Run specific benchmark
python -m src.evaluation.run_benchmarks \
    --checkpoint <path> \
    --benchmarks gsm8k \
    --num_samples 100  # Quick evaluation
```

### Comparing Experiments
```bash
# Compare two experiments
python -m src.analysis.compare \
    exp_20250123_143022_a1b2c3 \
    exp_20250122_091544_f9e8d7

# Compare all experiments with specific hyperparameter
python -m src.analysis.filter \
    --where "config.training.learning_rate > 1e-4" \
    --order_by "evaluation.gsm8k.accuracy DESC" \
    --limit 5
```

### Reproducing an Experiment
```bash
# Reproduce exact experiment from ID
python -m src.experiment_tracker.reproduce \
    exp_20250123_143022_a1b2c3 \
    --new_experiment_name "reproduction_test"

# This will:
# 1. Checkout the exact git commit
# 2. Load the exact configuration
# 3. Run training with same hyperparameters
# 4. Track as new experiment with link to original
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Implement `ExperimentTracker` class
- [ ] Create SQLite database schema
- [ ] Build configuration management system
- [ ] Add git integration for metadata

### Phase 2: Evaluation Framework (Week 2)
- [ ] Create `BenchmarkRegistry` system
- [ ] Implement GSM8K benchmark wrapper
- [ ] Add MATH benchmark support
- [ ] Build standardized metrics collection

### Phase 3: Analysis Tools (Week 3)
- [ ] Build leaderboard CLI
- [ ] Implement experiment comparison
- [ ] Add statistical testing
- [ ] Create visualization tools

### Phase 4: Integration (Week 4)
- [ ] Integrate with `train_grpo.py`
- [ ] Add W&B backend support
- [ ] Update documentation
- [ ] Add CLI commands

## Testing Strategy

### Unit Tests
```python
# tests/test_experiment_tracker.py
def test_experiment_creation():
    tracker = ExperimentTracker(backend="memory")
    exp_id = tracker.start_experiment(config={"lr": 1e-4})
    assert exp_id.startswith("exp_")

def test_metric_logging():
    tracker = ExperimentTracker(backend="memory")
    exp_id = tracker.start_experiment(config={})
    tracker.log_metrics({"loss": 0.5}, step=0)
    metrics = tracker.get_metrics(exp_id)
    assert len(metrics) == 1
```

### Integration Tests
```python
# tests/test_integration.py
def test_full_experiment_lifecycle():
    """Test complete experiment: train, evaluate, compare"""
    # Small training run
    exp_id = run_training(num_steps=10)

    # Evaluation
    results = run_evaluation(exp_id, benchmark="gsm8k", num_samples=10)
    assert "accuracy" in results["metrics"]

    # Analysis
    leaderboard = get_leaderboard("gsm8k")
    assert exp_id in [e["experiment_id"] for e in leaderboard]
```

## Success Metrics

The framework will be considered successful if:

1. **Adoption**: 100% of training runs use the tracker
2. **Reproducibility**: Any experiment can be reproduced with single command
3. **Discoverability**: Top experiment can be found in < 30 seconds
4. **Performance**: < 5% overhead on training time
5. **Reliability**: No data loss across 100+ experiments

## Future Enhancements

### Short-term (3-6 months)
- **Multi-run experiments**: Track hyperparameter sweeps as single experiment
- **Model comparison**: Compare different base models (Gemma, Llama, etc.)
- **Cost tracking**: Log TPU/GPU costs per experiment
- **Automatic alerts**: Notify when new best model found

### Long-term (6-12 months)
- **LLM-as-judge evaluation**: Assess reasoning quality with LLM
- **Active learning**: Identify hard samples for targeted training
- **Meta-learning**: Learn which hyperparameters work best
- **Distributed tracking**: Multi-team experiment sharing

## Appendix A: Configuration Schema

Full JSON schema for experiment configuration:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["model", "training", "grpo", "generation", "dataset"],
  "properties": {
    "model": {
      "type": "object",
      "properties": {
        "base_model": {"type": "string"},
        "use_lora": {"type": "boolean"},
        "lora_rank": {"type": "integer", "minimum": 1},
        "lora_alpha": {"type": "integer", "minimum": 1}
      }
    },
    "training": {
      "type": "object",
      "properties": {
        "num_steps": {"type": "integer", "minimum": 1},
        "learning_rate": {"type": "number", "minimum": 0},
        "batch_size": {"type": "integer", "minimum": 1},
        "weight_decay": {"type": "number", "minimum": 0},
        "warmup_fraction": {"type": "number", "minimum": 0, "maximum": 1}
      }
    }
  }
}
```

## Appendix B: Metrics Glossary

| Metric | Definition | Range | Good Value |
|--------|------------|-------|------------|
| **accuracy** | Exact numerical match | [0, 1] | > 0.7 |
| **partial_accuracy** | Within 10% tolerance | [0, 1] | > 0.8 |
| **format_accuracy** | Proper tag usage | [0, 1] | > 0.95 |
| **self_consistency** | Agreement across samples | [0, 1] | > 0.6 |
| **avg_reasoning_length** | Tokens in reasoning | [0, ∞) | 150-300 |
| **avg_generation_time** | Seconds per sample | [0, ∞) | < 2.0 |

## Appendix C: Error Type Classification

| Error Type | Description | Example |
|------------|-------------|---------|
| **format_error** | Missing/incorrect tags | No `<reasoning>` tag |
| **calculation_error** | Wrong arithmetic | 12 + 15 = 26 |
| **reasoning_error** | Flawed logic | Incorrect problem setup |
| **extraction_error** | Failed to extract answer | Answer in wrong format |
| **timeout_error** | Generation too slow | > 30 seconds |

---

**Document Version**: 1.0
**Last Updated**: 2025-01-23
**Authors**: Claude (System Design), Engineering Team
**Status**: Approved for Implementation
