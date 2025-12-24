# Rubrics Directory

This directory contains YAML rubric definition files for use with the rubrics module (`src/rubrics/`).

## What is a Rubric?

A rubric defines evaluation criteria for scoring model responses during GRPO training. Each rubric contains:

- **Criteria**: Individual scoring dimensions (e.g., accuracy, format, reasoning)
- **Weights**: Relative importance of each criterion
- **Keywords**: Terms that indicate criterion satisfaction (used for TF-IDF scoring)
- **Reference Response**: Optional example of an ideal response
- **Target Score**: Expected score for the reference response

## File Format

Rubrics are defined in YAML format. You can define a single rubric or a rubric set containing multiple rubrics.

### Single Rubric

```yaml
name: "step_by_step_reasoning"
description: "Evaluates quality of step-by-step reasoning"
question_types:
  - math
  - arithmetic
criteria:
  - name: "logical_flow"
    description: "Reasoning follows logical progression"
    weight: 2.0
    keywords:
      - first
      - then
      - therefore
  - name: "accuracy"
    description: "Correct final answer"
    weight: 2.0
reference_response: |
  <reasoning>Step by step solution</reasoning>
  <answer>42</answer>
target_score: 15.0
```

### Rubric Set

```yaml
name: "Math Reasoning Rubric Set"
description: "Collection of math evaluation rubrics"
metadata:
  version: "1.0"
  author: "your_name"

rubrics:
  - name: "basic_math"
    description: "For simple arithmetic"
    question_types: [math, arithmetic]
    criteria:
      - name: "calculation"
        description: "Shows correct calculation"
        weight: 2.0

  - name: "word_problems"
    description: "For word problems"
    question_types: [math, word_problem]
    criteria:
      - name: "understanding"
        description: "Identifies what the problem asks"
        weight: 1.5
      - name: "solution"
        description: "Provides correct solution"
        weight: 2.0
```

## Usage

### Loading Rubrics

```python
from src.rubrics import load_rubricset_from_yaml, load_rubrics_from_directory

# Load a single file
rubricset = load_rubricset_from_yaml("rubrics/example_math.yaml")

# Load all YAML files from directory
all_rubrics = load_rubrics_from_directory("rubrics/")
```

### Creating Rubrics Programmatically

```python
from src.rubrics import RubricBuilder, save_rubricset_to_yaml, RubricSet

# Simple rubric with RubricBuilder
rubric = (
    RubricBuilder("my_rubric", "Custom evaluation rubric")
    .with_criterion("format", "Uses <reasoning>/<answer> tags", weight=1.0)
    .with_criterion("accuracy", "Correct answer", weight=2.0)
    .with_question_types("math")
    .build()
)

# Save to YAML
rubricset = RubricSet(name="custom", rubrics=[rubric])
save_rubricset_to_yaml(rubricset, "rubrics/custom.yaml")
```

### Complete Example: Math Reasoning Rubric

```python
from src.rubrics import (
    RubricBuilder,
    RubricSet,
    save_rubricset_to_yaml,
    quick_test_rubric,
)

# Build a comprehensive math reasoning rubric
math_rubric = (
    RubricBuilder("math_reasoning", "Evaluates mathematical problem-solving quality")
    .with_criterion(
        name="format_compliance",
        description="Uses required <reasoning> and <answer> tags correctly",
        weight=1.0,
        keywords=["<reasoning>", "</reasoning>", "<answer>", "</answer>"],
    )
    .with_criterion(
        name="step_by_step",
        description="Shows clear step-by-step reasoning process",
        weight=2.0,
        keywords=["first", "then", "next", "step", "calculate", "therefore"],
    )
    .with_criterion(
        name="mathematical_accuracy",
        description="Performs calculations correctly and arrives at right answer",
        weight=2.5,
        keywords=["equals", "result", "total", "sum", "product"],
    )
    .with_criterion(
        name="clarity",
        description="Explanation is clear and easy to follow",
        weight=1.5,
        keywords=["because", "since", "means", "shows"],
    )
    .with_question_types("math", "arithmetic", "algebra", "word_problem")
    .with_reference(
        response="""<reasoning>
Let me solve this step by step.
First, I identify that we need to calculate the total.
The calculation is: 5 + 10 = 15
Therefore, the answer is 15.
</reasoning>
<answer>15</answer>""",
        target_score=18.0,
    )
    .with_metadata(author="training_team", version="1.0")
    .build()
)

# Quick test to validate the rubric
test_responses = [
    # Good response - should score high
    "<reasoning>First, add 2+2=4. Therefore the answer is 4.</reasoning><answer>4</answer>",
    # Medium response - missing some elements
    "<reasoning>2+2=4</reasoning><answer>4</answer>",
    # Poor response - no tags
    "The answer is 4",
]

result = quick_test_rubric(math_rubric, test_responses)
print("Scores:", result["scores"])
print("Mean:", result["mean"])

# Save if satisfied
rubricset = RubricSet(
    name="Math Evaluation Rubrics",
    description="Rubrics for evaluating math problem solutions",
    rubrics=[math_rubric],
)
save_rubricset_to_yaml(rubricset, "rubrics/math_evaluation.yaml")
```

### Modifying Existing Rubrics

```python
from src.rubrics import (
    load_rubricset_from_yaml,
    clone_rubric,
    adjust_weights,
    add_keywords_to_criterion,
    merge_rubrics,
    normalize_weights,
    save_rubric_to_yaml,
)

# Load existing rubric
rubricset = load_rubricset_from_yaml("rubrics/example_math.yaml")
original = rubricset[0]

# Clone and modify
modified = clone_rubric(original, new_name="enhanced_math")

# Adjust criterion weights
modified = adjust_weights(modified, {
    "logical_flow": 3.0,  # Increase importance of logical flow
    "accuracy": 2.5,
})

# Add more keywords to a criterion
modified = add_keywords_to_criterion(
    modified,
    "logical_flow",
    ["consequently", "thus", "hence"],
)

# Normalize weights to sum to 1.0
modified = normalize_weights(modified, target_sum=1.0)

# Save the modified rubric
save_rubric_to_yaml(modified, "rubrics/enhanced_math.yaml")

# Or merge multiple rubrics into one
combined = merge_rubrics(
    [rubricset[0], rubricset[1]],
    name="combined_rubric",
    description="Merged criteria from multiple rubrics",
)
```

### Testing Rubrics

```python
from src.rubrics import test_rubric_with_dataset, quick_test_rubric

# Quick test with sample responses
result = quick_test_rubric(
    rubric,
    test_responses=[
        "<reasoning>2+2=4</reasoning><answer>4</answer>",
        "The answer is 4",
    ]
)
print(f"Scores: {result['scores']}")

# Test against a dataset
results = test_rubric_with_dataset(rubricset, dataset)
print(results.summary())
```

### Using in GRPO Training

```python
from src.rubrics import load_rubricset_from_yaml, create_grpo_reward_function

rubricset = load_rubricset_from_yaml("rubrics/example_math.yaml")
reward_fn = create_grpo_reward_function(rubricset)

# Use reward_fn in your GRPO training loop
```

## Files in This Directory

| File | Description |
|------|-------------|
| `example_math.yaml` | Example rubrics for math reasoning evaluation |

## Best Practices

1. **Use descriptive names**: Rubric and criterion names should clearly indicate what they evaluate
2. **Choose meaningful keywords**: Keywords are used for TF-IDF matching - pick terms that reliably indicate quality
3. **Balance weights**: Higher weights mean more influence on the final score
4. **Include reference responses**: Helps calibrate scoring and provides examples
5. **Test before training**: Use `quick_test_rubric()` to validate your rubric design
