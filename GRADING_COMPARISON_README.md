# Grading Methodology Comparison Tools

This repository now includes comprehensive tooling for comparing different grading methodologies and understanding their effects on model behavior during GRPO training.

## What's New

### Core Framework (`src/grading_comparison.py`)
- **GradingComparator**: Main class for comparing multiple grading methods
- **ComparisonResults**: Data structure for storing and analyzing results
- Statistical analysis, correlation computation, and agreement matrices
- Visualization tools (distributions, heatmaps, scatter plots)
- Save/load functionality for reproducibility

### Grading Registry (`src/grading_registry.py`)
- Auto-registration of all available grading methods
- Convenience functions for quick comparisons
- Full dataset analysis with one function call
- Support for both format-based, accuracy-based, and rubric-based methods

### Available Grading Methods
1. **format_reward** - Rewards proper formatting with reasoning/answer tags
2. **accuracy_reward** - Rewards correct answers (requires ground truth)
3. **rubric_reward** - Rubric-as-Reward scoring (requires rubrics)
4. **match_format_exactly** - Strict format matching
5. **match_format_approximately** - Approximate format matching with partial credit
6. **check_answer** - Answer checking with tolerance for close answers
7. **check_numbers** - Numerical answer extraction and comparison

## Quick Start

### 1. Command Line

```bash
# List available methods
python scripts/compare_grading.py --list-methods

# Compare all methods on GSM8K
python scripts/compare_grading.py --dataset gsm8k --samples 100 --plot

# Compare specific methods
python scripts/compare_grading.py --dataset gsm8k \
    --methods format_reward accuracy_reward \
    --samples 200 --output ./analysis
```

### 2. Python API

```python
from src.grading_registry import compare_methods_quick

results = compare_methods_quick(
    prompts=["What is 2+2?"],
    completions=["<reasoning>2+2=4</reasoning><answer>4</answer>"],
    method_names=['format_reward', 'accuracy_reward'],
    answers=["4"]
)

print(results.statistics)
```

### 3. Interactive Analysis

```bash
jupyter notebook notebooks/grading_comparison_analysis.ipynb
```

## Files Added

```
src/
├── grading_comparison.py      # Core comparison framework
├── grading_registry.py         # Convenience wrappers and registration

scripts/
└── compare_grading.py          # CLI tool for comparisons

examples/
└── compare_grading_methods.py  # Detailed examples

notebooks/
└── grading_comparison_analysis.ipynb  # Interactive notebook

docs/
└── GRADING_COMPARISON.md       # Comprehensive documentation

tests/
├── test_grading_comparison.py  # Full test suite
└── test_simple.py              # Simple standalone test
```

## Key Features

### 1. Statistical Analysis
- Mean, median, standard deviation, min/max, quartiles
- Distribution plots for each method
- Identify outliers and patterns

### 2. Correlation Analysis
- Pearson and Spearman correlations
- Correlation heatmaps
- Pairwise scatter plots
- Agreement matrices

### 3. Disagreement Analysis
- Find examples where methods disagree most
- Understand why methods differ
- Identify edge cases

### 4. Behavior Analysis
- Group completions by score ranges
- Analyze patterns in model outputs
- See how grading affects generation

### 5. Visualization
- Score distribution histograms
- Correlation heatmaps
- Pairwise scatter plots
- Custom plots for analysis

### 6. Reporting
- JSON results for programmatic access
- Markdown reports for human readers
- Publication-ready plots

## Usage Examples

### Example 1: Basic Comparison

```python
from src.grading_registry import create_standard_comparator

comparator = create_standard_comparator()
results = comparator.compare(
    prompts=prompts,
    completions=completions,
    answers=answers
)

# Print statistics
for method, stats in results.statistics.items():
    print(f"{method}: mean={stats['mean']:.2f}")
```

### Example 2: Find Disagreements

```python
disagreements = comparator.find_disagreements(
    'format_reward', 'accuracy_reward',
    top_k=10,
    normalize=True
)

for ex in disagreements:
    print(f"Difference: {ex['difference']:.2f}")
    print(f"Question: {ex['prompt']}")
```

### Example 3: Full Dataset Analysis

```python
from src.grading_registry import analyze_dataset
from src.utils import load_gsm8k_dataset

dataset = load_gsm8k_dataset(max_examples=500)
results = analyze_dataset(
    dataset=dataset,
    output_dir='./analysis',
    generate_plots=True
)
```

## Documentation

Full documentation available in `docs/GRADING_COMPARISON.md` including:
- Detailed API reference
- Interpretation guidelines
- Best practices
- Troubleshooting

## Testing

Run tests to verify the framework:

```bash
# Simple test
python tests/test_simple.py

# Full test suite (requires dependencies)
python tests/test_grading_comparison.py
```

## Dependencies

Core framework requires:
- numpy
- scipy
- matplotlib
- seaborn

Install with:
```bash
pip install numpy scipy matplotlib seaborn
```

## Use Cases

1. **Research**: Compare grading methods for papers/experiments
2. **Training**: Choose optimal reward function for GRPO
3. **Debugging**: Understand why certain methods fail
4. **Analysis**: Measure agreement with human judgments
5. **Development**: Test new grading methods

## Future Enhancements

Potential additions:
- Support for custom aggregation functions
- Multi-dimensional reward analysis
- Integration with Weights & Biases
- Automated method selection
- Real-time training monitoring

## Contributing

To add a new grading method:
1. Implement function with signature: `(prompts, completions, **kwargs) -> List[float]`
2. Add to `get_all_grading_methods()` in `src/grading_registry.py`
3. Document in `docs/GRADING_COMPARISON.md`
4. Add tests

## License

Same as parent project.

---

For questions or issues, please refer to `docs/GRADING_COMPARISON.md` or open an issue.
