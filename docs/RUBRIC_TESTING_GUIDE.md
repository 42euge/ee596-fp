# Rubric Testing Infrastructure Guide

A comprehensive framework for rapidly testing new rubric designs against small models before scaling up to full training runs.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Creating Custom Rubrics](#creating-custom-rubrics)
5. [Evaluation](#evaluation)
6. [Comparison & Analysis](#comparison--analysis)
7. [CLI Usage](#cli-usage)
8. [API Reference](#api-reference)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

---

## Overview

The Rubric Testing Infrastructure allows researchers to:

- **Rapidly iterate** on rubric designs with small sample sizes
- **Compare** multiple rubric approaches statistically
- **Evaluate** rubrics on small models before expensive training
- **Benchmark** new rubrics against established baselines
- **Visualize** rubric performance with detailed reports

### Key Features

- 🎯 **Modular Design**: Create rubrics from functions, classes, or compositions
- ⚡ **Fast Iteration**: Test on small samples (50-200) in minutes
- 📊 **Statistical Analysis**: T-tests, ANOVA, effect sizes
- 🔬 **Model Agnostic**: Works with any HuggingFace model
- 📈 **Rich Reports**: Markdown, JSON, HTML, and plots
- 🧪 **Comprehensive Testing**: 40+ unit tests included

---

## Quick Start

### Installation

The rubric testing framework is included in the project. Ensure dependencies are installed:

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .

# Optional: For plotting
pip install matplotlib seaborn
```

### 30-Second Example

```python
from src.rubric_testing import (
    KeywordMatchRubric,
    RubricEvaluator,
    EvaluationConfig,
)

# Create a rubric
rubric = KeywordMatchRubric()

# Configure evaluation
config = EvaluationConfig(
    num_samples=50,  # Small sample for rapid testing
    model_name="google/gemma-3-1b-it",
    dataset_name="openrubrics",
)

# Evaluate
evaluator = RubricEvaluator(config)
result = evaluator.evaluate(rubric)

# View results
print(f"Mean Score: {result.mean_score:.2f} ± {result.std_score:.2f}")
```

### CLI Quick Start

```bash
# Test a single rubric
python scripts/test_rubric.py --rubric keyword --samples 50

# Compare multiple rubrics
python scripts/test_rubric.py --compare keyword format length \
    --samples 100 --report comparison.md
```

---

## Core Concepts

### Rubric Score

A `RubricScore` contains:
- **total**: Overall score (typically 0-10 range)
- **components**: Breakdown of score by component
- **metadata**: Additional scoring context

```python
score = RubricScore(
    total=8.5,
    components={"keywords": 3.0, "format": 5.5},
    metadata={"matched_terms": ["solve", "equation"]}
)
```

### Base Rubric

All rubrics inherit from `BaseRubric`:

```python
class BaseRubric(ABC):
    def score(self, prompt, completion, rubric,
              reference_response=None, target_score=None, **kwargs):
        """Score a completion according to rubric criteria"""
        pass
```

### Evaluation Workflow

```
1. Design Rubric → 2. Configure Eval → 3. Run Evaluation → 4. Analyze Results
```

---

## Creating Custom Rubrics

### Method 1: Function-Based (Simplest)

```python
from src.rubric_testing import create_rubric, RubricScore

def my_scorer(prompt, completion, rubric, **kwargs):
    # Your scoring logic
    word_count = len(completion.split())
    score = min(word_count / 10, 10.0)

    return RubricScore(
        total=score,
        components={"word_count": word_count}
    )

rubric = create_rubric("word_counter", score_func=my_scorer)
```

### Method 2: Class-Based (More Flexible)

```python
from src.rubric_testing import BaseRubric, RubricScore

class MyRubric(BaseRubric):
    def __init__(self, threshold=100):
        super().__init__("my_rubric", weight=1.0)
        self.threshold = threshold
        self._score_range = (0.0, 10.0)

    def score(self, prompt, completion, rubric, **kwargs):
        length = len(completion)
        score = 10.0 if length > self.threshold else 5.0

        return RubricScore(
            total=score,
            components={"length": length, "passed": length > self.threshold}
        )
```

### Method 3: Designer Pattern (For Registration)

```python
from src.rubric_testing import RubricDesigner

designer = RubricDesigner()

@designer.register("clarity", weight=1.0)
def clarity_rubric(prompt, completion, rubric, **kwargs):
    # Check for clear structure
    has_intro = completion.startswith(("First", "To solve", "Let's"))
    has_conclusion = any(word in completion.lower()
                         for word in ["therefore", "thus", "finally"])

    score = 0.0
    if has_intro: score += 5.0
    if has_conclusion: score += 5.0

    return RubricScore(total=score)

# Use registered rubric
rubric = designer.get("clarity")
```

### Method 4: Composite Rubrics

Combine multiple rubrics:

```python
from src.rubric_testing import CompositeRubric, WeightedRubric

# Equal weights
composite = CompositeRubric("combined", [
    KeywordMatchRubric(),
    FormatComplianceRubric(),
])

# Custom weights
weighted = WeightedRubric("weighted", {
    KeywordMatchRubric(): 0.7,
    FormatComplianceRubric(): 0.3,
})
```

---

## Evaluation

### Basic Evaluation

```python
from src.rubric_testing import RubricEvaluator, EvaluationConfig

config = EvaluationConfig(
    num_samples=100,
    model_name="google/gemma-3-1b-it",
    temperature=0.9,
    max_length=512,
)

evaluator = RubricEvaluator(config)
result = evaluator.evaluate(my_rubric)

# Access results
print(f"Mean: {result.mean_score}")
print(f"Std: {result.std_score}")
print(f"Components: {result.component_stats}")
```

### Evaluation with Checkpoints

Test rubrics on fine-tuned models:

```python
config = EvaluationConfig(
    num_samples=50,
    use_lora=True,
    lora_checkpoint="./checkpoints/lora_epoch_1",
    quantization="4bit",  # Faster evaluation
)

evaluator = RubricEvaluator(config)
result = evaluator.evaluate(rubric)
```

### Evaluation Without Generation

If you already have completions:

```python
prompts = ["Q1", "Q2", "Q3"]
completions = ["A1", "A2", "A3"]
rubrics_text = ["R1", "R2", "R3"]

result = evaluator.evaluate(
    my_rubric,
    prompts=prompts,
    completions=completions,
    rubrics=rubrics_text,
)
```

### Batch Evaluation

Evaluate multiple rubrics on the same data:

```python
rubrics = [
    KeywordMatchRubric(),
    FormatComplianceRubric(),
    LengthRubric(),
]

results = evaluator.evaluate_multiple(rubrics)
```

---

## Comparison & Analysis

### Basic Comparison

```python
from src.rubric_testing import compare_rubrics

# After evaluating multiple rubrics
comparison = compare_rubrics(results, alpha=0.05)

print(f"Best: {comparison.best_rubric}")
print(f"Rankings: {comparison.rankings}")
print(f"Statistical tests: {comparison.statistical_tests}")
```

### Statistical Analysis

The comparator provides:

1. **Rankings**: Rubrics ordered by performance
2. **Relative Performance**: Normalized to best rubric (0-100%)
3. **Pairwise t-tests**: Statistical significance between pairs
4. **ANOVA**: Overall difference test (3+ rubrics)
5. **Effect Sizes**: Cohen's d for practical significance

```python
# Access statistical tests
ttests = comparison.statistical_tests["pairwise_ttests"]
for pair, test in ttests.items():
    if test["significant"]:
        print(f"{pair}: p={test['p_value']:.4f} ✓")

anova = comparison.statistical_tests["anova"]
print(f"ANOVA: {anova['interpretation']}")
```

### Generating Reports

```python
from src.rubric_testing import generate_report

# Markdown report
report = generate_report(
    results,
    comparison=comparison,
    output_path="rubric_report.md",
    format="markdown"
)

# JSON export
generate_report(results, output_path="results.json", format="json")

# HTML report
generate_report(results, output_path="report.html", format="html")
```

### Visualizations

```python
from src.rubric_testing.reporter import (
    plot_rubric_comparison,
    plot_score_distributions,
    export_to_csv,
)

# Bar chart comparison
plot_rubric_comparison(results, output_path="comparison.png")

# Violin plots of distributions
plot_score_distributions(results, output_path="distributions.png")

# CSV export for further analysis
export_to_csv(results, output_path="results.csv")
```

---

## CLI Usage

### Basic Commands

```bash
# Test single rubric
python scripts/test_rubric.py --rubric keyword --samples 50

# Compare multiple rubrics
python scripts/test_rubric.py --compare keyword format length \
    --samples 100

# With custom model checkpoint
python scripts/test_rubric.py --rubric format \
    --model-checkpoint ./checkpoints/lora \
    --samples 50

# Generate report
python scripts/test_rubric.py --compare keyword format \
    --report comparison.md \
    --plot
```

### Available Rubrics

- `keyword`: Keyword matching with rubric criteria
- `format`: Format compliance (reasoning/answer tags)
- `length`: Response length optimization
- `composite`: Combination of multiple rubrics

### Advanced Options

```bash
python scripts/test_rubric.py --compare keyword format \
    --samples 100 \
    --model google/gemma-3-1b-it \
    --quantization 4bit \
    --temperature 0.7 \
    --max-length 512 \
    --dataset openrubrics \
    --report-format json \
    --output-dir ./results \
    --alpha 0.01 \
    --verbose
```

### Full Options

```
--rubric              Single rubric to evaluate
--compare             Multiple rubrics to compare
--samples             Number of samples (default: 100)
--model               Model name or path
--model-checkpoint    Path to LoRA checkpoint
--quantization        4bit, 8bit, or none
--device              auto, cuda, cpu, mps
--temperature         Sampling temperature
--max-length          Max generation length
--dataset             Dataset name
--report              Output report path
--report-format       markdown, json, or html
--plot                Generate comparison plots
--alpha               Significance level (default: 0.05)
```

---

## API Reference

### Core Classes

#### `BaseRubric`
Abstract base class for all rubrics.

**Methods:**
- `score(prompt, completion, rubric, **kwargs) -> RubricScore`
- `normalize_score(score) -> float`

**Properties:**
- `name: str`
- `weight: float`
- `score_range: Tuple[float, float]`

#### `RubricEvaluator`
Quick evaluation engine for testing rubrics.

**Constructor:**
```python
RubricEvaluator(config: EvaluationConfig)
```

**Methods:**
- `evaluate(rubric, prompts=None, completions=None, ...) -> EvaluationResult`
- `evaluate_multiple(rubrics) -> List[EvaluationResult]`

#### `RubricComparator`
Statistical comparison of multiple rubrics.

**Constructor:**
```python
RubricComparator(alpha: float = 0.05)
```

**Methods:**
- `compare(results, metric="mean_score") -> ComparisonResult`
- `find_best_rubric(results, criteria="mean_score") -> Tuple[str, EvaluationResult]`

#### `RubricReporter`
Report and visualization generation.

**Methods:**
- `generate_report(results, comparison=None, output_path=None, format="markdown") -> str`

### Built-in Rubrics

#### `KeywordMatchRubric`
Scores based on keyword overlap with rubric criteria.

```python
KeywordMatchRubric(
    name="keyword_match",
    weight=1.0,
    case_sensitive=False
)
```

#### `LengthRubric`
Scores based on response length relative to target.

```python
LengthRubric(
    name="length",
    weight=1.0,
    target_length=200,
    tolerance=0.5
)
```

#### `FormatComplianceRubric`
Scores based on format tags (reasoning/answer).

```python
FormatComplianceRubric(
    name="format_compliance",
    weight=1.0
)
```

### Configuration

#### `EvaluationConfig`
```python
@dataclass
class EvaluationConfig:
    num_samples: int = 100
    temperature: float = 0.9
    max_length: int = 512
    model_name: str = "google/gemma-3-1b-it"
    dataset_name: str = "openrubrics"
    quantization: Optional[str] = None
    use_lora: bool = False
    lora_checkpoint: Optional[str] = None
    # ... see source for all options
```

---

## Best Practices

### 1. Start Small

Always test with small samples (50-100) first:

```python
# Good: Quick iteration
config = EvaluationConfig(num_samples=50)

# Not recommended initially: Too slow for iteration
config = EvaluationConfig(num_samples=1000)
```

### 2. Use Quantization for Speed

4-bit quantization significantly speeds up evaluation:

```python
config = EvaluationConfig(
    num_samples=100,
    quantization="4bit",  # 4x faster, minimal quality loss
)
```

### 3. Component Breakdown

Always return component scores for debugging:

```python
def my_rubric(**kwargs):
    # Good: Clear component breakdown
    return RubricScore(
        total=score,
        components={
            "keyword_match": kw_score,
            "length_check": len_score,
        }
    )

    # Not recommended: Just total
    return RubricScore(total=score)
```

### 4. Statistical Significance

Don't rely solely on mean scores:

```python
comparison = compare_rubrics(results, alpha=0.05)

# Check if differences are significant
for pair, test in comparison.statistical_tests["pairwise_ttests"].items():
    if test["significant"]:
        print(f"{pair} shows significant difference (p={test['p_value']:.4f})")
```

### 5. Iterative Refinement

```python
# Workflow for rubric development:
# 1. Create initial rubric
rubric_v1 = MyRubric(threshold=100)

# 2. Test on small sample
result_v1 = evaluator.evaluate(rubric_v1)
print(f"V1 mean: {result_v1.mean_score}")

# 3. Refine based on component stats
print(result_v1.component_stats)

# 4. Create improved version
rubric_v2 = MyRubric(threshold=150)  # Adjusted based on stats

# 5. Compare versions
comparison = compare_rubrics([result_v1, result_v2])
```

### 6. Normalize Scores

When combining rubrics, use normalization:

```python
# Good: Normalized combination
composite = CompositeRubric("combo", rubrics, normalize=True)

# Use WeightedRubric for explicit control
weighted = WeightedRubric("weighted", {
    high_priority_rubric: 0.6,
    low_priority_rubric: 0.4,
})
```

---

## Examples

See `examples/custom_rubrics.py` for comprehensive examples including:

1. Simple function-based rubrics
2. Class-based rubrics with parameters
3. Domain-specific rubrics (math solutions)
4. Composite rubrics
5. Designer pattern usage
6. Full evaluation workflows

### Running Examples

```bash
# Run example file
python examples/custom_rubrics.py

# Run CLI examples
python scripts/test_rubric.py --compare keyword format --samples 50

# Run tests
pytest tests/test_rubric_testing.py -v
```

---

## Troubleshooting

### Model Loading Issues

```python
# If model loading fails, try without generation:
config = EvaluationConfig(
    num_samples=50,
    num_generations_per_prompt=0,  # Skip generation
)

# Then provide pre-generated completions
result = evaluator.evaluate(rubric, prompts=..., completions=...)
```

### Dataset Loading Issues

```python
# If dataset loading fails, use manual data:
prompts = ["Q1", "Q2", "Q3"]
completions = ["A1", "A2", "A3"]
rubrics = ["R1", "R2", "R3"]

result = evaluator.evaluate(
    rubric,
    prompts=prompts,
    completions=completions,
    rubrics=rubrics,
)
```

### Memory Issues

```python
# Use quantization and smaller batches
config = EvaluationConfig(
    quantization="4bit",
    batch_size=1,
    num_samples=50,  # Start small
)
```

---

## Further Reading

- See `src/rubric_testing/` for source code
- See `tests/test_rubric_testing.py` for test examples
- See project README for overall system architecture
- See `src/utils.py` for existing rubric reward functions

---

## Contributing

When creating new rubric types:

1. Inherit from `BaseRubric`
2. Implement `score()` method
3. Set appropriate `_score_range`
4. Return `RubricScore` with components
5. Add tests to `tests/test_rubric_testing.py`
6. Add examples to `examples/custom_rubrics.py`
7. Update this documentation

---

*For questions or issues, please open a GitHub issue or refer to the project documentation.*
