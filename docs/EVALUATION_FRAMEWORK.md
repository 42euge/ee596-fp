## Evaluation Framework

### Overview

The evaluation framework provides a standardized, extensible system for benchmarking reasoning models across multiple datasets. It features:

- **Unified Interface**: All benchmarks return consistent metrics
- **Plugin Architecture**: Easy to add new benchmarks
- **Automatic Registration**: Benchmarks auto-register on import
- **Rich Metrics**: Beyond accuracy - format compliance, partial credit, timing
- **Integration Ready**: Works seamlessly with experiment tracker

### Quick Start

```python
from src.evaluation.benchmark_registry import BenchmarkRegistry
import src.evaluation.benchmarks  # Auto-register benchmarks

# List available benchmarks
print(BenchmarkRegistry.list_benchmarks())

# Evaluate on GSM8K
result = BenchmarkRegistry.evaluate(
    model=my_model,
    benchmark_name="gsm8k",
    num_samples=100
)

print(f"Accuracy: {result.metrics['accuracy']:.1%}")
```

### Supported Benchmarks

| Benchmark | Description | Dataset Size | Metric Focus |
|-----------|-------------|--------------|--------------|
| **GSM8K** | Grade School Math 8K | 1,319 test | Numerical reasoning |
| **MATH** | Coming soon | - | Advanced math |
| **OpenRubrics** | Coming soon | - | Rubric-based evaluation |

### Metrics Explained

#### Core Metrics

1. **accuracy**: Exact match between predicted and gold answer
   - Binary: correct or incorrect
   - Most strict metric
   - Range: [0, 1]

2. **partial_accuracy**: Within tolerance of gold answer
   - For numerical answers: within 10% of gold value
   - More forgiving than exact match
   - Range: [0, 1]

3. **format_accuracy**: Proper use of formatting tags
   - Checks for `<reasoning>` and `<answer>` tags
   - Essential for structured outputs
   - Range: [0, 1]

4. **avg_generation_time**: Average time to generate per sample
   - Measured in seconds
   - Includes model forward pass and decoding
   - Lower is better

#### Extended Metrics (Future)

- **reasoning_length**: Average tokens in reasoning section
- **reasoning_coherence**: LLM-as-judge score for reasoning quality
- **self_consistency**: Agreement across multiple samples
- **error_type_distribution**: Classification of error types

### Adding a New Benchmark

#### 1. Create Benchmark Class

```python
# src/evaluation/benchmarks/my_benchmark.py

from .base import BaseBenchmark
from typing import Any, Dict, List, Optional

class MyBenchmark(BaseBenchmark):
    """My custom benchmark."""

    def __init__(self):
        super().__init__("my_benchmark")

    def load_dataset(
        self,
        split: str = "test",
        num_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load dataset samples."""
        # Option 1: Load from HuggingFace
        from datasets import load_dataset
        dataset = load_dataset("my/dataset", split=split)

        # Convert to standard format
        samples = []
        for i, item in enumerate(dataset):
            if num_samples and i >= num_samples:
                break

            samples.append({
                "id": f"my_benchmark_{i}",
                "question": item["question"],
                "answer": item["answer"]
            })

        return samples

    def extract_answer(self, text: str) -> Any:
        """Extract answer from generated text."""
        # Default implementation handles <answer> tags
        # Override if you need custom extraction
        import re
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            # Custom parsing here
            return self._parse_answer(answer_text)
        return None

    def check_answer(self, predicted: Any, gold: Any) -> bool:
        """Check if predicted answer is correct."""
        # Implement comparison logic
        if predicted is None or gold is None:
            return False

        # Example: case-insensitive string match
        return str(predicted).lower() == str(gold).lower()

    def _parse_answer(self, text: str) -> Any:
        """Custom answer parsing logic."""
        # Your parsing code here
        return text.strip()
```

#### 2. Register Benchmark

```python
# At the end of src/evaluation/benchmarks/my_benchmark.py

from ..benchmark_registry import BenchmarkRegistry
BenchmarkRegistry.register("my_benchmark", MyBenchmark)
```

#### 3. Import in __init__.py

```python
# src/evaluation/benchmarks/__init__.py

from .gsm8k import GSM8KBenchmark
from .my_benchmark import MyBenchmark  # Add this

__all__ = [
    "GSM8KBenchmark",
    "MyBenchmark",  # Add this
]
```

#### 4. Use Your Benchmark

```python
import src.evaluation.benchmarks  # Auto-registers MyBenchmark

result = BenchmarkRegistry.evaluate(
    model=my_model,
    benchmark_name="my_benchmark",
    num_samples=100
)
```

### Model Interface Requirements

Your model must implement a `generate()` method:

```python
class MyModel:
    def generate(self, question: str, **kwargs) -> str:
        """
        Generate answer for a question.

        Args:
            question: Input question
            **kwargs: Generation parameters (temperature, max_length, etc.)

        Returns:
            Generated text with proper formatting
        """
        # Example structure:
        # <reasoning>
        # First, we need to understand...
        # Then, we calculate...
        # </reasoning>
        # <answer>42</answer>

        return self._generate_response(question, **kwargs)
```

### Evaluation Best Practices

#### 1. Consistent Generation Settings

```python
# Use same settings for fair comparison
generation_config = {
    "temperature": 0.7,
    "max_new_tokens": 512,
    "top_p": 0.9,
    "top_k": 50,
}

result = BenchmarkRegistry.evaluate(
    model=model,
    benchmark_name="gsm8k",
    **generation_config
)
```

#### 2. Use Greedy Decoding for Reproducibility

```python
# For deterministic evaluation
result = BenchmarkRegistry.evaluate(
    model=model,
    benchmark_name="gsm8k",
    temperature=0.0,  # Greedy decoding
    do_sample=False
)
```

#### 3. Sample Size Selection

```python
# Quick evaluation during development
result = BenchmarkRegistry.evaluate(
    model=model,
    benchmark_name="gsm8k",
    num_samples=100  # ~10% of dataset
)

# Full evaluation for final results
result = BenchmarkRegistry.evaluate(
    model=model,
    benchmark_name="gsm8k",
    num_samples=None  # All samples
)
```

#### 4. Save Detailed Results

```python
result = BenchmarkRegistry.evaluate(model, "gsm8k")

# Save to JSON for later analysis
import json
with open("evaluation_results.json", "w") as f:
    json.dump(result.to_dict(), f, indent=2)

# Log to experiment tracker
from src.experiment_tracker import ExperimentTracker
tracker = ExperimentTracker(backends=["local"])
tracker.log_evaluation("gsm8k", result.to_dict())
```

### Formatting Evaluation Results

```python
from src.evaluation.metrics import format_evaluation_results

result = BenchmarkRegistry.evaluate(model, "gsm8k")

# Print formatted results
print(format_evaluation_results(result, verbose=True))

# Output:
# ======================================================================
# Benchmark: gsm8k
# Samples: 1000
# ======================================================================
#
# Metrics:
#   Accuracy:         73.2%
#   Partial Accuracy: 81.5%
#   Format Accuracy:  95.1%
#   Avg Gen Time:     1.24s
#
# ======================================================================
# Sample Results (first 10):
# ======================================================================
# ...
```

### Comparing Multiple Benchmarks

```python
from src.evaluation.benchmark_registry import BenchmarkRegistry
from src.evaluation.metrics import compare_results

# Evaluate on multiple benchmarks
results = BenchmarkRegistry.evaluate_all(
    model=model,
    benchmark_names=["gsm8k", "math", "openrubrics"],
    num_samples=100
)

# Print comparison
print(compare_results(results))

# Output:
# ======================================================================
# Multi-Benchmark Comparison
# ======================================================================
#
# Benchmark            Accuracy     Partial      Format
# -------------------- ------------ ------------ ------------
# gsm8k                     73.2%        81.5%        95.1%
# math                      45.3%        52.7%        93.4%
# openrubrics               68.9%        76.2%        97.8%
# -------------------- ------------ ------------ ------------
# Average                   62.5%        70.1%        95.4%
# ======================================================================
```

### Error Analysis

```python
result = BenchmarkRegistry.evaluate(model, "gsm8k", num_samples=1000)

# Analyze incorrect samples
incorrect = [
    sample for sample in result.per_sample_results
    if not sample.is_correct
]

print(f"Total incorrect: {len(incorrect)}")

# Group by error type
format_errors = [s for s in incorrect if not s.format_correct]
calculation_errors = [
    s for s in incorrect
    if s.format_correct and s.predicted_answer != s.gold_answer
]

print(f"Format errors: {len(format_errors)} ({len(format_errors)/len(incorrect)*100:.1f}%)")
print(f"Calculation errors: {len(calculation_errors)} ({len(calculation_errors)/len(incorrect)*100:.1f}%)")

# Examine specific errors
for i, sample in enumerate(incorrect[:5]):
    print(f"\nError #{i+1}:")
    print(f"  Question: {sample.question[:100]}...")
    print(f"  Gold: {sample.gold_answer}")
    print(f"  Predicted: {sample.predicted_answer}")
    print(f"  Reasoning: {sample.reasoning[:100]}...")
```

### Integration with Experiment Tracker

```python
from src.experiment_tracker import ExperimentTracker, ExperimentConfig
from src.evaluation.benchmark_registry import BenchmarkRegistry

# Train model
config = ExperimentConfig(learning_rate=5e-5, num_steps=500)
tracker = ExperimentTracker(backends=["local"])
exp_id = tracker.start_experiment(config)

# ... training code ...

# Evaluate after training
result = BenchmarkRegistry.evaluate(
    model=trained_model,
    benchmark_name="gsm8k"
)

# Log to tracker
tracker.log_evaluation("gsm8k", result.to_dict())
tracker.finish_experiment(status="completed")

# Later: retrieve and analyze
from src.experiment_tracker import LocalBackend
backend = LocalBackend("experiments.db")
eval_results = backend.get_evaluation_results(exp_id, "gsm8k")
print(f"Accuracy: {eval_results[0]['metrics_json']['accuracy']:.1%}")
```

### Custom Metrics

You can add custom metrics by overriding the `evaluate()` method:

```python
class MyBenchmark(BaseBenchmark):
    def evaluate(self, model, split="test", num_samples=None, **generation_kwargs):
        # Call parent evaluation
        result = super().evaluate(model, split, num_samples, **generation_kwargs)

        # Add custom metrics
        avg_reasoning_length = sum(
            len(sample.reasoning.split())
            for sample in result.per_sample_results
        ) / len(result.per_sample_results)

        result.metrics["avg_reasoning_length"] = avg_reasoning_length

        # Add custom analysis
        result.metadata["version"] = "1.0"
        result.metadata["custom_info"] = "..."

        return result
```

### Performance Optimization

#### Batch Evaluation

```python
# TODO: Implement batch evaluation for faster processing
# Currently evaluates one sample at a time
# Future: Batch multiple samples for GPU efficiency
```

#### Caching

```python
# Cache evaluation results to avoid re-running
import hashlib
import json
from pathlib import Path

def cached_evaluate(model, benchmark_name, **kwargs):
    # Create cache key from model and parameters
    cache_key = hashlib.md5(
        json.dumps({"model": model.name, **kwargs}, sort_keys=True).encode()
    ).hexdigest()

    cache_file = Path(f"cache/{benchmark_name}_{cache_key}.json")

    # Check cache
    if cache_file.exists():
        print("Loading from cache...")
        with open(cache_file) as f:
            cached_result = json.load(f)
        return cached_result

    # Run evaluation
    result = BenchmarkRegistry.evaluate(model, benchmark_name, **kwargs)

    # Save to cache
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(result.to_dict(), f)

    return result
```

### Future Enhancements

Planned features:

1. **More Benchmarks**: MATH, HumanEval, MBPP, BigBench subsets
2. **LLM-as-Judge**: Reasoning quality evaluation with GPT-4
3. **Multi-turn Evaluation**: Dialogue and interactive reasoning
4. **Calibration Metrics**: Confidence scoring and calibration
5. **Adversarial Testing**: Robustness evaluation
6. **Batch Processing**: GPU-efficient evaluation
7. **Distributed Evaluation**: Multi-GPU/TPU support

### References

- GSM8K Paper: https://arxiv.org/abs/2110.14168
- MATH Dataset: https://arxiv.org/abs/2103.03874
- Evaluation Best Practices: https://arxiv.org/abs/2401.00001

### Troubleshooting

**Import Error**: Make sure to import benchmarks module
```python
import src.evaluation.benchmarks  # Required for auto-registration
```

**Missing Dependencies**: Install datasets library
```bash
pip install datasets
```

**Slow Evaluation**: Reduce num_samples or use caching
```python
result = BenchmarkRegistry.evaluate(model, "gsm8k", num_samples=100)
```

**Format Errors**: Ensure model outputs proper tags
```python
# Model should output:
# <reasoning>Step-by-step reasoning here</reasoning>
# <answer>Final answer here</answer>
```
