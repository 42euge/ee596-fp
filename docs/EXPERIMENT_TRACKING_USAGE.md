# Experiment Tracking and Evaluation Framework - Usage Guide

This guide explains how to use the new experiment tracking and evaluation frameworks for the Gemma3-1B Reasoning Model project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Experiment Tracking](#experiment-tracking)
3. [Evaluation Framework](#evaluation-framework)
4. [Analysis Tools](#analysis-tools)
5. [Integration with Training](#integration-with-training)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### Installation

The frameworks require no additional dependencies beyond the project's existing requirements:

```bash
# Ensure you're in the project directory
cd /home/user/ee596-fp

# Optional: Install scipy for statistical tests
pip install scipy
```

### Run a Tracked Experiment

```bash
# Run a demo training with tracking
python src/train_with_tracking.py \
    --num_steps 100 \
    --learning_rate 5e-5 \
    --experiment_name "my_first_experiment" \
    --notes "Testing the tracking framework"

# Experiment ID will be printed, e.g., exp_20250123_143022_a1b2c3
```

### View Results

```bash
# List all experiments
python src/experiment_cli.py list

# Show leaderboard
python src/experiment_cli.py leaderboard

# Show experiment details
python src/experiment_cli.py show exp_20250123_143022_a1b2c3
```

## Experiment Tracking

### Basic Usage

```python
from src.experiment_tracker import ExperimentTracker, ExperimentConfig

# 1. Create configuration
config = ExperimentConfig(
    base_model="google/gemma-3-1b-it",
    learning_rate=5e-5,
    lora_rank=8,
    num_steps=500
)

# 2. Initialize tracker
tracker = ExperimentTracker(
    backends=["local"],  # or ["local", "wandb"]
    db_path="experiments.db"
)

# 3. Start experiment
experiment_id = tracker.start_experiment(
    config=config,
    notes="Testing higher learning rate"
)

# 4. Log metrics during training
for step in range(num_steps):
    # ... training code ...
    tracker.log_metrics({
        "train/loss": loss,
        "train/reward": reward
    }, step=step)

# 5. Log checkpoints
tracker.log_checkpoint("/path/to/checkpoint", step=500)

# 6. Finish experiment
tracker.finish_experiment(status="completed")
```

### Configuration Options

All hyperparameters are captured in `ExperimentConfig`:

```python
config = ExperimentConfig(
    # Model
    base_model="google/gemma-3-1b-it",
    use_lora=True,
    lora_rank=8,
    lora_alpha=16,

    # Training
    num_steps=500,
    learning_rate=5e-5,
    batch_size=64,
    optimizer="adamw",
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_fraction=0.1,

    # GRPO
    num_generations=4,
    beta=0.1,
    epsilon=0.2,
    temperature=0.7,

    # Generation
    max_prompt_length=1024,
    max_generation_steps=512,
    top_k=50,
    top_p=0.9,

    # Dataset
    dataset_name="openrubrics",
    train_size=8000,
    eval_size=1000,
    seed=42,

    # Rewards
    format_weight=1.0,
    accuracy_weight=2.0,
    rubric_weight=0.5
)
```

### Multiple Backends

Track to both local database and W&B:

```python
tracker = ExperimentTracker(
    backends=["local", "wandb"],
    db_path="experiments.db",
    wandb_project="tunix-grpo"
)
```

### Automatic Git Tracking

The tracker automatically captures:
- Git commit hash
- Git branch name
- Whether working directory is dirty
- Timestamp
- User and hostname
- Python version
- Device type (TPU/CUDA/CPU)

## Evaluation Framework

### Using the Benchmark Registry

```python
from src.evaluation.benchmark_registry import BenchmarkRegistry
import src.evaluation.benchmarks  # Auto-register benchmarks

# List available benchmarks
benchmarks = BenchmarkRegistry.list_benchmarks()
print(benchmarks)  # ['gsm8k']

# Evaluate on a single benchmark
result = BenchmarkRegistry.evaluate(
    model=my_model,
    benchmark_name="gsm8k",
    split="test",
    num_samples=100  # or None for full dataset
)

print(f"Accuracy: {result.metrics['accuracy']:.1%}")

# Evaluate on multiple benchmarks
results = BenchmarkRegistry.evaluate_all(
    model=my_model,
    benchmark_names=["gsm8k"],  # or None for all
    num_samples=100
)
```

### Model Interface

Your model must implement a `generate()` method:

```python
class MyModel:
    def generate(self, question: str, **kwargs) -> str:
        """
        Generate an answer for the question.

        Args:
            question: Input question
            **kwargs: Additional generation parameters

        Returns:
            Generated text with <reasoning> and <answer> tags
        """
        # Your generation code here
        return "<reasoning>...</reasoning><answer>42</answer>"
```

### Evaluation Metrics

All benchmarks return standardized metrics:

```python
{
    "accuracy": 0.73,              # Exact match
    "partial_accuracy": 0.81,      # Within 10% tolerance
    "format_accuracy": 0.95,       # Proper tag usage
    "avg_generation_time": 1.2,    # Seconds per sample
    "total_time": 120.0,           # Total evaluation time
    "num_samples": 100,            # Number of samples evaluated
}
```

### Logging Evaluation Results

```python
# After evaluation, log to tracker
tracker.log_evaluation("gsm8k", result.to_dict())
```

### Creating Custom Benchmarks

```python
from src.evaluation.benchmarks.base import BaseBenchmark

class MyCustomBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("my_benchmark")

    def load_dataset(self, split="test", num_samples=None):
        # Load your dataset
        return [
            {"question": "...", "answer": "..."},
            ...
        ]

    def extract_answer(self, text: str):
        # Extract answer from generated text
        # Default implementation handles <answer> tags
        return super().extract_answer(text)

    def check_answer(self, predicted, gold):
        # Check if answer is correct
        return predicted == gold

# Register your benchmark
from src.evaluation.benchmark_registry import BenchmarkRegistry
BenchmarkRegistry.register("my_benchmark", MyCustomBenchmark)
```

## Analysis Tools

### Command-Line Interface

```bash
# List experiments
python src/experiment_cli.py list --limit 20

# Show leaderboard
python src/experiment_cli.py leaderboard \
    --benchmark gsm8k \
    --metric accuracy \
    --top 10

# Show experiment details
python src/experiment_cli.py show exp_20250123_143022_a1b2c3

# Compare experiments
python src/experiment_cli.py compare \
    exp_20250123_143022_a1b2c3 \
    exp_20250122_091544_f9e8d7

# Statistical significance test
python src/experiment_cli.py significance \
    exp_001 exp_002 \
    --benchmark gsm8k \
    --method bootstrap

# Export experiment
python src/experiment_cli.py export exp_001 --output exp_001.json

# Database statistics
python src/experiment_cli.py stats
```

### Programmatic Analysis

```python
from src.analysis import (
    generate_leaderboard,
    compare_experiments,
    compute_significance
)

# Generate leaderboard
leaderboard = generate_leaderboard(
    db_path="experiments.db",
    benchmark="gsm8k",
    metric="accuracy",
    top_k=10
)

for i, entry in enumerate(leaderboard, 1):
    print(f"{i}. {entry['experiment_id']}: {entry['accuracy']:.1%}")

# Compare experiments
comparison = compare_experiments(
    experiment_ids=["exp_001", "exp_002"],
    db_path="experiments.db"
)

# Print formatted comparison
from src.analysis.compare import format_comparison
print(format_comparison(comparison))

# Statistical significance
result = compute_significance(
    "exp_001", "exp_002",
    benchmark="gsm8k",
    method="bootstrap",
    n_bootstrap=10000
)

print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['is_significant']}")
```

### Leaderboard Output

```
================================================================================
GSM8K Leaderboard - Top 10 by accuracy
================================================================================
Rank   Experiment ID                  Accuracy     Partial      Format       Date
------ ------------------------------ ------------ ------------ ------------ ------------
1      exp_20250123_143022_a1b2c3       75.8%        83.2%        96.1%     2025-01-20
2      exp_20250122_091544_d4e5f6       74.2%        81.8%        95.3%     2025-01-19
3      exp_20250121_154833_g7h8i9       73.1%        80.9%        94.7%     2025-01-18
================================================================================
```

### Comparison Output

```
================================================================================
Comparing 3 Experiments
================================================================================

Experiments:
  [1] exp_20250123_143022_a1b2c3           (a1b2c3, 2025-01-20)
  [2] exp_20250122_091544_d4e5f6           (d4e5f6, 2025-01-19)
  [3] exp_20250121_154833_g7h8i9           (g7h8i9, 2025-01-18)

================================================================================
Configuration Differences:
================================================================================

Parameter                      exp_20250123_14 exp_20250122_09 exp_20250121_15
------------------------------ --------------- --------------- ---------------
learning_rate                  5e-5            1e-4            5e-5
lora_rank                      8               8               16
num_generations                4               8               4

================================================================================
Performance Comparison:
================================================================================

gsm8k:
Metric                    exp_20250123_14 exp_20250122_09 exp_20250121_15
------------------------- --------------- --------------- ---------------
Accuracy                       75.8% ⭐        69.8%           73.1%
Partial Accuracy               83.2% ⭐        77.2%           80.9%
Format Accuracy                96.1% ⭐        93.4%           94.7%

================================================================================
Winner: exp_20250123_143022_a1b2c3
Average Accuracy: 75.8%
Improvement: +2.7% vs average of others
================================================================================
```

## Integration with Training

### Modifying `scripts/train_grpo.py`

Add tracking to your existing training script:

```python
# At the top of the file
from src.experiment_tracker import ExperimentTracker, ExperimentConfig

def main():
    # Parse args (existing code)
    args = parse_args()

    # Create config from args
    config = ExperimentConfig(
        base_model=args.model_id,
        learning_rate=args.learning_rate,
        # ... map all args to config
    )

    # Initialize tracker
    tracker = ExperimentTracker(
        backends=["local", "wandb"],
        db_path="experiments.db",
        wandb_project=args.wandb_project
    )

    # Start experiment
    experiment_id = tracker.start_experiment(
        config=config,
        notes=args.experiment_notes  # Add this arg
    )

    # Training loop (existing code with additions)
    for step in range(num_steps):
        # ... existing training code ...

        # Log metrics
        tracker.log_metrics({
            "train/loss": float(loss),
            "train/reward": float(reward),
            "train/learning_rate": float(lr),
        }, step=step)

        # Save checkpoint
        if step % checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(...)
            tracker.log_checkpoint(checkpoint_path, step)

    # After training, run evaluation
    from src.evaluation.benchmark_registry import BenchmarkRegistry
    import src.evaluation.benchmarks

    result = BenchmarkRegistry.evaluate(
        model=model,  # Your trained model
        benchmark_name="gsm8k",
        num_samples=1000
    )

    tracker.log_evaluation("gsm8k", result.to_dict())

    # Finish
    tracker.finish_experiment(status="completed")
```

### Adding Experiment Notes

```python
# Start experiment with notes
experiment_id = tracker.start_experiment(
    config=config,
    notes="Testing impact of higher learning rate on GSM8K accuracy"
)

# Or add via command-line argument
# python scripts/train_grpo.py \
#     --learning_rate 1e-4 \
#     --experiment_notes "Testing higher LR"
```

## Advanced Usage

### Hyperparameter Search with Tracking

```python
from src.experiment_tracker import ExperimentTracker, ExperimentConfig
import itertools

# Define search space
learning_rates = [1e-5, 5e-5, 1e-4]
lora_ranks = [8, 16, 32]

# Initialize tracker
tracker = ExperimentTracker(backends=["local"])

# Grid search
for lr, rank in itertools.product(learning_rates, lora_ranks):
    config = ExperimentConfig(
        learning_rate=lr,
        lora_rank=rank,
        num_steps=500
    )

    exp_id = tracker.start_experiment(
        config=config,
        experiment_name=f"sweep_lr{lr}_rank{rank}",
        notes=f"Grid search: LR={lr}, rank={rank}"
    )

    try:
        # Train and evaluate
        train_model(config, tracker)
        evaluate_model(model, tracker)
        tracker.finish_experiment(status="completed")
    except Exception as e:
        print(f"Failed: {e}")
        tracker.finish_experiment(status="failed")

# Analyze results
from src.analysis import generate_leaderboard
leaderboard = generate_leaderboard(metric="accuracy", top_k=5)
```

### Analyzing Hyperparameter Impact

```python
from src.experiment_tracker import LocalBackend
import json

backend = LocalBackend("experiments.db")
experiments = backend.get_all_experiments()

# Collect data
data = []
for exp in experiments:
    config = json.loads(exp["config_json"])
    eval_results = backend.get_evaluation_results(exp["experiment_id"], "gsm8k")

    if eval_results:
        metrics = eval_results[0]["metrics_json"]
        data.append({
            "learning_rate": config["learning_rate"],
            "lora_rank": config["lora_rank"],
            "accuracy": metrics["accuracy"]
        })

# Analyze correlation
import pandas as pd
df = pd.DataFrame(data)
correlation = df.corr()["accuracy"]
print(correlation.sort_values(ascending=False))
```

### Reproducing an Experiment

```python
from src.experiment_tracker import LocalBackend
import json

# Load experiment configuration
backend = LocalBackend("experiments.db")
exp = backend.get_experiment("exp_20250123_143022_a1b2c3")

if exp:
    # Parse config
    config_dict = json.loads(exp["config_json"])
    config = ExperimentConfig.from_dict(config_dict)

    # Start new experiment with same config
    tracker = ExperimentTracker(backends=["local"])
    new_exp_id = tracker.start_experiment(
        config=config,
        experiment_name="reproduction",
        notes=f"Reproducing {exp['experiment_id']}"
    )

    # Run training with same hyperparameters
    train_model(config, tracker)
```

## Troubleshooting

### Database Locked Error

If you get "database is locked" error:

```python
# Use a different database file for parallel experiments
tracker = ExperimentTracker(
    backends=["local"],
    db_path=f"experiments_{os.getpid()}.db"
)
```

### Missing Per-Sample Results

For statistical tests, you need per-sample results. Ensure evaluation saves them:

```python
# When logging evaluation, include per_sample_results
result = BenchmarkRegistry.evaluate(model, "gsm8k")
tracker.log_evaluation("gsm8k", result.to_dict())  # Includes per-sample data
```

### W&B Not Logging

Make sure wandb is installed and logged in:

```bash
pip install wandb
wandb login
```

Then use the wandb backend:

```python
tracker = ExperimentTracker(
    backends=["local", "wandb"],
    wandb_project="tunix-grpo"
)
```

### Large Database Files

To clean up old experiments:

```bash
# Delete failed experiments
python src/experiment_cli.py list | grep "failed" | \
    awk '{print $1}' | \
    xargs -I {} python src/experiment_cli.py delete {} --confirm

# Or manually with SQL
sqlite3 experiments.db "DELETE FROM experiments WHERE status='failed'"
```

## Best Practices

1. **Always add notes**: Helps you remember why you ran the experiment
2. **Use meaningful experiment names**: Makes it easier to find later
3. **Log checkpoints regularly**: Enables mid-training analysis
4. **Run evaluation immediately after training**: Keeps results together
5. **Compare before claiming improvement**: Use statistical tests
6. **Track failed experiments**: Learn from what doesn't work
7. **Export important results**: Backup before cleanup

## Next Steps

- See [EXPERIMENT_TRACKING_DESIGN.md](EXPERIMENT_TRACKING_DESIGN.md) for architecture details
- Check [README.md](../README.md) for project overview
- Review existing experiments: `python src/experiment_cli.py leaderboard`

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing experiments for examples
- Review the design document for implementation details
