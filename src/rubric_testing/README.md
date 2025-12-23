# Rubric Testing Infrastructure

Infrastructure for rapidly testing new rubric designs against small models before scaling up.

## Quick Start

```python
from src.rubric_testing import (
    KeywordMatchRubric,
    RubricEvaluator,
    EvaluationConfig,
    compare_rubrics,
)

# 1. Create a rubric
rubric = KeywordMatchRubric()

# 2. Configure evaluation
config = EvaluationConfig(num_samples=50)
evaluator = RubricEvaluator(config)

# 3. Evaluate
result = evaluator.evaluate(rubric)
print(f"Score: {result.mean_score:.2f} ± {result.std_score:.2f}")
```

## CLI Usage

```bash
# Test single rubric
python scripts/test_rubric.py --rubric keyword --samples 50

# Compare multiple rubrics
python scripts/test_rubric.py --compare keyword format length \
    --samples 100 --report comparison.md
```

## Features

- **Fast Iteration**: Test rubrics on small samples (50-200) in minutes
- **Statistical Comparison**: T-tests, ANOVA, effect sizes
- **Model Agnostic**: Works with any HuggingFace model
- **Rich Reports**: Markdown, JSON, HTML formats
- **Extensible**: Easy to create custom rubrics

## Components

### Designer (`designer.py`)
- `BaseRubric`: Abstract base class
- `RubricDesigner`: Registration system
- `CompositeRubric`: Combine multiple rubrics
- Built-in rubrics: `KeywordMatchRubric`, `LengthRubric`, `FormatComplianceRubric`

### Evaluator (`evaluator.py`)
- `RubricEvaluator`: Quick evaluation engine
- `EvaluationConfig`: Configuration options
- `EvaluationResult`: Results with statistics

### Comparator (`comparator.py`)
- `RubricComparator`: Statistical comparison
- `ComparisonResult`: Rankings and significance tests
- `compare_rubrics()`: Convenience function

### Reporter (`reporter.py`)
- `RubricReporter`: Generate reports
- `generate_report()`: Markdown/JSON/HTML output
- Visualization utilities (requires matplotlib)

## Documentation

See `/docs/RUBRIC_TESTING_GUIDE.md` for comprehensive documentation.

## Examples

See `/examples/custom_rubrics.py` for custom rubric implementations.

## Tests

```bash
pytest tests/test_rubric_testing.py -v
```

## Architecture

```
RubricDesigner → BaseRubric → RubricEvaluator → EvaluationResult
                                                       ↓
                                              RubricComparator → ComparisonResult
                                                       ↓
                                                RubricReporter → Reports/Plots
```
