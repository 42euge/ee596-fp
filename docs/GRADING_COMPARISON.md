# Grading Methodology Comparison Tools

Comprehensive tooling for comparing different grading methodologies and understanding their effects on model behavior during GRPO training.

## Table of Contents

1. [Overview](#overview)
2. [Available Grading Methods](#available-grading-methods)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Understanding Results](#understanding-results)
8. [Best Practices](#best-practices)

## Overview

When training models with reinforcement learning (like GRPO), the choice of reward/grading function significantly impacts model behavior. This toolkit provides:

- **Comparison Framework**: Compare multiple grading methods side-by-side
- **Statistical Analysis**: Understand score distributions, correlations, and patterns
- **Disagreement Analysis**: Find where methods differ and why
- **Behavior Analysis**: See how different grading affects model outputs
- **Visualizations**: Plots and reports to communicate findings

## Available Grading Methods

### Format-Based Methods

#### `format_reward`
- **Description**: Rewards proper format usage (reasoning + answer tags)
- **Score Range**: -2.0 to +2.0
- **Use Case**: Encourage structured outputs
- **Strengths**: Simple, interpretable, fast
- **Weaknesses**: Doesn't evaluate content quality

#### `match_format_exactly`
- **Description**: Strict format matching (all or nothing)
- **Score Range**: 0.0 or 3.0
- **Use Case**: Enforce exact formatting requirements
- **Strengths**: Unambiguous, clear signal
- **Weaknesses**: No partial credit

#### `match_format_approximately`
- **Description**: Format matching with partial credit
- **Score Range**: -2.5 to +2.5
- **Use Case**: More lenient format enforcement
- **Strengths**: Gradual feedback
- **Weaknesses**: More complex scoring

### Accuracy-Based Methods

#### `accuracy_reward`
- **Description**: Rewards correct answers (for verifiable tasks)
- **Score Range**: 0.0 or 1.5
- **Use Case**: Math problems, factual questions
- **Strengths**: Objective, ground truth based
- **Weaknesses**: Binary (no partial credit), requires answers

#### `check_answer`
- **Description**: Answer checking with partial credit
- **Score Range**: -1.0 to +3.0
- **Use Case**: Math/numeric problems with tolerance
- **Strengths**: Rewards close answers, more nuanced
- **Weaknesses**: Requires ground truth

#### `check_numbers`
- **Description**: Numerical answer extraction and exact matching
- **Score Range**: 0.0 or 1.5
- **Use Case**: Numeric answer verification
- **Strengths**: Simple, clear
- **Weaknesses**: Binary, no tolerance

### Content-Based Methods

#### `rubric_reward`
- **Description**: Rubric-as-Reward (RaR) scoring
- **Score Range**: 0.0 to 20.0
- **Components**:
  - Rubric overlap (0-10): TF-IDF weighted term matching
  - Reference similarity (0-5): Sequence matching with reference
  - Target score alignment (0-5): Alignment with human ratings
- **Use Case**: Open-ended tasks, essay grading
- **Strengths**: Captures quality without exact answers
- **Weaknesses**: Requires rubrics, more complex

## Installation

The comparison tools are included in the main project. Ensure you have the required dependencies:

```bash
# Install base requirements
pip install -r requirements.txt

# For visualizations (recommended)
pip install matplotlib seaborn scipy
```

## Quick Start

### 1. Command Line Interface

```bash
# Compare all methods on GSM8K dataset
python scripts/compare_grading.py --dataset gsm8k --samples 100 --plot

# Compare specific methods
python scripts/compare_grading.py --dataset gsm8k \
    --methods format_reward accuracy_reward \
    --samples 200 --output ./my_analysis

# List available methods
python scripts/compare_grading.py --list-methods
```

### 2. Python API

```python
from src.grading_registry import create_standard_comparator

# Create comparator with all methods
comparator = create_standard_comparator()

# Run comparison
results = comparator.compare(
    prompts=["What is 2+2?"],
    completions=["<reasoning>2+2=4</reasoning><answer>4</answer>"],
    method_names=['format_reward', 'accuracy_reward'],
    answers=["4"]
)

# Analyze results
print(results.statistics)
comparator.plot_distributions()
```

### 3. Jupyter Notebook

```bash
jupyter notebook notebooks/grading_comparison_analysis.ipynb
```

## Usage Examples

### Example 1: Basic Comparison

```python
from src.grading_registry import compare_methods_quick

prompts = ["What is 5 + 3?", "Calculate 10 - 4"]
completions = [
    "<reasoning>5+3=8</reasoning><answer>8</answer>",
    "The answer is 6"  # Missing format
]
answers = ["8", "6"]

results = compare_methods_quick(
    prompts=prompts,
    completions=completions,
    method_names=['format_reward', 'accuracy_reward'],
    answers=answers
)

# Check statistics
for method, stats in results.statistics.items():
    print(f"{method}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

### Example 2: Find Disagreements

```python
from src.grading_registry import create_standard_comparator

comparator = create_standard_comparator()
results = comparator.compare(prompts, completions, answers=answers)

# Find where methods disagree most
disagreements = comparator.find_disagreements(
    'format_reward', 'accuracy_reward',
    top_k=10,
    normalize=True
)

for ex in disagreements:
    print(f"Question: {ex['prompt']}")
    print(f"Format score: {ex['format_reward_score']:.2f}")
    print(f"Accuracy score: {ex['accuracy_reward_score']:.2f}")
    print(f"Difference: {ex['difference']:.2f}\n")
```

### Example 3: Analyze Full Dataset

```python
from src.grading_registry import analyze_dataset
from src.utils import load_gsm8k_dataset

# Load dataset
dataset = load_gsm8k_dataset(split="train", max_examples=500)

# Add completions (from model inference)
for item in dataset:
    item['completion'] = your_model.generate(item['question'])

# Run full analysis
results = analyze_dataset(
    dataset=dataset,
    method_names=['format_reward', 'accuracy_reward'],
    output_dir='./analysis',
    generate_plots=True
)
```

### Example 4: Custom Grading Method

```python
from src.grading_comparison import GradingComparator

# Define custom grading function
def length_penalty(prompts, completions, **kwargs):
    """Penalize overly long responses."""
    scores = []
    for completion in completions:
        length = len(completion)
        if length < 100:
            score = 1.0
        elif length < 500:
            score = 0.5
        else:
            score = 0.0
        scores.append(score)
    return scores

# Register and compare
comparator = GradingComparator()
comparator.register_method(
    name='length_penalty',
    function=length_penalty,
    description='Penalizes long responses',
    score_range=(0.0, 1.0)
)

results = comparator.compare(prompts, completions)
```

## API Reference

### GradingComparator

Main class for comparing grading methods.

#### Methods

- **`register_method(name, function, description, score_range, ...)`**
  - Register a grading method for comparison
  - Args: name (str), function (callable), description (str), score_range (tuple)

- **`compare(prompts, completions, method_names=None, **kwargs)`**
  - Run comparison across methods
  - Returns: ComparisonResults object

- **`find_disagreements(method1, method2, top_k=10, normalize=True)`**
  - Find examples where methods disagree most
  - Returns: List of disagreement examples

- **`analyze_behavior_effects(method_name, thresholds=None)`**
  - Analyze how grading affects behavior
  - Returns: Dict with behavior analysis by score range

- **`plot_distributions(save_path=None)`**
  - Plot score distributions for all methods

- **`plot_correlation_heatmap(save_path=None)`**
  - Plot correlation heatmap between methods

- **`plot_pairwise_scatter(method1, method2, save_path=None)`**
  - Plot scatter comparing two methods

- **`generate_report(output_path)`**
  - Generate markdown report

### ComparisonResults

Dataclass storing comparison results.

#### Attributes

- **`methods`**: List of method names
- **`scores`**: Dict mapping method names to score lists
- **`statistics`**: Dict of statistical summaries per method
- **`correlations`**: Dict of pairwise correlations
- **`agreement_matrix`**: Matrix of quartile agreement rates
- **`examples`**: List of examples with all scores

#### Methods

- **`save(filepath)`**: Save results to JSON
- **`load(filepath)`**: Load results from JSON

### Convenience Functions

- **`create_standard_comparator()`**: Create comparator with all methods registered
- **`get_all_grading_methods()`**: Get metadata for all available methods
- **`compare_methods_quick(prompts, completions, ...)`**: One-line comparison
- **`analyze_dataset(dataset, method_names=None, ...)`**: Full dataset analysis

## Understanding Results

### Statistical Metrics

- **Mean**: Average score across all examples
- **Median**: Middle score (50th percentile)
- **Std**: Standard deviation (spread of scores)
- **Min/Max**: Score range
- **Q25/Q75**: 25th and 75th percentiles (interquartile range)

### Correlation Metrics

- **Pearson correlation**: Linear relationship between methods (-1 to +1)
  - Near +1: Methods agree strongly
  - Near 0: Methods independent
  - Near -1: Methods inversely related

- **Spearman correlation**: Rank-order relationship
  - More robust to outliers than Pearson
  - Captures monotonic (not just linear) relationships

### Agreement Matrix

Shows proportion of examples where methods assign scores in the same quartile:
- 1.0 (100%): Perfect agreement
- 0.5 (50%): Moderate agreement
- 0.25 (25%): Agreement by chance

### Interpretation Guidelines

1. **High correlation (>0.7)**: Methods largely redundant, pick simpler one
2. **Moderate correlation (0.3-0.7)**: Methods capture different aspects, consider combining
3. **Low correlation (<0.3)**: Methods measure very different things, may conflict in training
4. **Negative correlation**: Methods oppose each other, likely problematic

## Best Practices

### Choosing Methods to Compare

1. **Start with objectives**: What behavior do you want to encourage?
2. **Mix types**: Compare format, accuracy, and content methods
3. **Consider requirements**: Some methods need ground truth or rubrics
4. **Test incrementally**: Add one method at a time

### Interpreting Disagreements

When methods disagree:
1. **Examine examples**: Look at actual prompts/completions
2. **Check edge cases**: Often disagreements reveal edge cases
3. **Consider causation**: Does one method cause the disagreement?
4. **Validate assumptions**: Are your scoring rules correct?

### Using Results for Training

1. **Single method**: Choose highest correlation with human judgment
2. **Weighted combination**: Weight methods by importance
   ```python
   total_reward = 0.4 * format + 0.6 * accuracy
   ```
3. **Multi-objective**: Use separate reward heads
4. **Curriculum**: Start with simple (format), add complex (content) later

### Common Pitfalls

1. **Scale differences**: Normalize before comparing/combining
2. **Overfitting to metrics**: Methods can be gamed by models
3. **Missing ground truth**: Some methods need answers/rubrics
4. **Computational cost**: Complex methods slow down training

### Recommended Workflow

1. **Pilot study**: Compare methods on small dataset (100-500 examples)
2. **Analyze disagreements**: Understand where/why methods differ
3. **Select method(s)**: Choose based on objectives and correlations
4. **Validate**: Test selected method on held-out set
5. **Monitor training**: Track reward distributions during training
6. **Iterate**: Adjust based on model behavior

## Output Files

After running `analyze_dataset()`, you'll get:

- **comparison_results.json**: Full numerical results (for programmatic access)
- **comparison_report.md**: Human-readable summary report
- **distributions.png**: Histogram of scores for each method
- **correlations.png**: Heatmap of method correlations
- **scatter_*.png**: Pairwise scatter plots

## Troubleshooting

### "Method requires ground truth"
- Ensure you pass `answers=...` to `compare()`
- Check that dataset has 'answer' field

### "Method requires rubric"
- Pass `rubrics=...` to `compare()`
- Use OpenRubrics dataset or add rubrics manually

### "No methods available"
- Check that you've registered methods with `register_method()`
- Or use `create_standard_comparator()` for auto-registration

### Plots not showing
- Install matplotlib: `pip install matplotlib seaborn`
- Use `save_path` parameter to save instead of displaying

### Memory issues with large datasets
- Reduce sample size with `max_examples` parameter
- Process in batches
- Disable plots with `generate_plots=False`

## Citation

If you use these tools in your research, please cite:

```bibtex
@software{grading_comparison_tools,
  title={Grading Methodology Comparison Tools for GRPO},
  author={EE596 Final Project},
  year={2024},
  url={https://github.com/yourusername/ee596-fp}
}
```

## Contributing

To add a new grading method:

1. Implement the function with signature: `(prompts, completions, **kwargs) -> List[float]`
2. Add to `get_all_grading_methods()` in `src/grading_registry.py`
3. Document in this file under "Available Grading Methods"
4. Add tests and examples

## License

[Your License Here]
