# Reward Comparison Framework

A comprehensive toolkit for comparing different reward methodologies in reinforcement learning from human feedback (RLHF) and reward modeling.

## Overview

This framework allows researchers to easily compare different reward approaches:

- **Programmatic rewards**: Rule-based, pattern matching, format checking
- **Rubric-based evaluation**: Multi-criteria evaluation with configurable rubrics
- **Preference models**: LLM-as-judge, trained reward models, or external APIs

## Features

✨ **Easy to use**: Simple API for defining and comparing reward evaluators
📊 **Comprehensive analysis**: Correlation, agreement, and statistical analysis
📈 **Visualizations**: Score distributions, heatmaps, and scatter plots
🔍 **Disagreement detection**: Find edge cases where methods disagree
💾 **Export capabilities**: Save results to CSV, generate reports

## Quick Start

```python
from tunrex.reward_comparison import (
    ProgrammaticReward,
    RubricReward,
    RubricCriterion,
    RewardComparison,
    RewardAnalyzer,
)

# Define reward evaluators
format_reward = ProgrammaticReward(
    func=check_format,
    name="FormatCheck"
)

rubric = RubricReward(
    criteria=[
        RubricCriterion(
            name="Accuracy",
            description="Is the answer correct?",
            max_score=5.0,
            evaluator=lambda p, c, **kw: 5.0 if correct(c) else 0.0
        ),
        RubricCriterion(
            name="Clarity",
            description="Is the explanation clear?",
            max_score=3.0,
            evaluator=lambda p, c, **kw: rate_clarity(c)
        ),
    ],
    name="QualityRubric"
)

# Compare them
comparison = RewardComparison([format_reward, rubric])
result = comparison.evaluate(prompts, completions)

# Analyze
analyzer = RewardAnalyzer(result)
print(analyzer.generate_report())

# Visualize
analyzer.plot_correlation_heatmap("heatmap.png")
```

## Core Components

### 1. Base Classes (`base.py`)

- `BaseReward`: Abstract base class for all reward evaluators
- `RewardResult`: Contains scores and metadata from evaluation
- `RewardMetadata`: Tracks evaluator info and timing

### 2. Evaluators (`evaluators.py`)

#### ProgrammaticReward
Wraps existing reward functions:

```python
def my_reward_fn(prompts, completions, **kwargs):
    return [score_completion(c) for c in completions]

reward = ProgrammaticReward(my_reward_fn, name="MyReward")
```

#### RubricReward
Multi-criteria evaluation:

```python
rubric = RubricReward(
    criteria=[criterion1, criterion2],
    aggregation="sum"  # or "mean"
)
```

#### PreferenceModelReward
For LLM-as-judge or reward models:

```python
def gpt4_judge(prompts, completions, **kwargs):
    # Call GPT-4 API
    return scores

reward = PreferenceModelReward(
    model_fn=gpt4_judge,
    name="GPT4Judge"
)
```

### 3. Comparison (`comparison.py`)

Run multiple evaluators and compare:

```python
comparison = RewardComparison([reward1, reward2, reward3])
result = comparison.evaluate(prompts, completions, **metadata)

# Get summary
print(comparison.summarize())

# Find disagreements
disagreements = comparison.find_disagreements(threshold=2.0)

# Export to CSV
comparison.export_to_csv("results.csv", include_text=True)
```

### 4. Analysis (`analysis.py`)

Statistical analysis and visualization:

```python
analyzer = RewardAnalyzer(comparison_result)

# Correlation analysis
corr = analyzer.compute_correlations()
print(corr.format_matrix("pearson"))

# Agreement analysis
agreement = analyzer.compute_agreement(threshold=5.0, top_k=[5, 10])

# Generate report
report = analyzer.generate_report()

# Visualizations
analyzer.plot_score_distributions("distributions.png")
analyzer.plot_correlation_heatmap("heatmap.png")
analyzer.plot_score_comparison("reward1", "reward2", "comparison.png")
```

## Examples

See the `examples/` directory:

- `reward_comparison_example.py`: Basic usage with programmatic and rubric rewards
- `reward_comparison_advanced.py`: Custom evaluators and preference models

## Use Cases

### 1. Reward Function Development

Compare candidate reward functions during development:

```python
# Try different format checking approaches
exact_format = create_format_reward(r"<answer>.+</answer>", name="Exact")
flexible_format = create_format_reward(r"answer.*?:", name="Flexible")

comparison = RewardComparison([exact_format, flexible_format])
result = comparison.evaluate(dev_prompts, dev_completions)

# See which works better
analyzer = RewardAnalyzer(result)
print(analyzer.generate_report())
```

### 2. Validate Against Human Preferences

Check if your reward aligns with human judgments:

```python
my_reward = ProgrammaticReward(my_scoring_fn)
human_ratings = PreferenceModelReward(load_human_ratings)

comparison = RewardComparison([my_reward, human_ratings])
result = comparison.evaluate(test_prompts, test_completions)

# Check correlation
analyzer = RewardAnalyzer(result)
corr = analyzer.compute_correlations()
print(f"Alignment: {corr.get_pairwise_correlation('my_reward', 'human_ratings')}")
```

### 3. Ensemble Rewards

Create ensemble rewards based on analysis:

```python
# Compare multiple rewards
comparison = RewardComparison([reward1, reward2, reward3])
result = comparison.evaluate(prompts, completions)

# Analyze correlations
analyzer = RewardAnalyzer(result)
corr = analyzer.compute_correlations()

# Create weighted ensemble of highly-correlated rewards
def ensemble_reward(prompts, completions, **kwargs):
    scores1 = reward1.evaluate(prompts, completions, **kwargs).scores
    scores2 = reward2.evaluate(prompts, completions, **kwargs).scores
    return [0.6 * s1 + 0.4 * s2 for s1, s2 in zip(scores1, scores2)]
```

### 4. Identify Training Issues

Find samples where rewards give conflicting signals:

```python
comparison = RewardComparison([format_reward, accuracy_reward, quality_reward])
result = comparison.evaluate(train_prompts, train_completions)

# Find problematic samples
disagreements = comparison.find_disagreements(threshold=3.0)

# Review and fix
for dis in disagreements[:10]:
    print(f"Sample {dis['sample_idx']}: {dis['completion']}")
    print(f"  Format: {dis['score1']}, Accuracy: {dis['score2']}")
    # Investigate and adjust rewards
```

## Helper Functions

The framework provides convenient helpers in `evaluators.py`:

```python
from tunrex.reward_comparison import (
    create_format_reward,
    create_length_reward,
    create_keyword_reward,
)

# Format checking
format_check = create_format_reward(
    pattern=r"<answer>.+</answer>",
    score_on_match=1.0,
    score_on_miss=0.0
)

# Length constraints
length_check = create_length_reward(
    min_length=50,
    max_length=500,
    max_score=1.0
)

# Keyword requirements
keyword_check = create_keyword_reward(
    required_keywords=["reasoning", "because"],
    forbidden_keywords=["guess", "maybe"],
    score_per_required=0.5,
    penalty_per_forbidden=-1.0
)
```

## Integration with GRPO Training

Use in GRPO training by wrapping evaluators:

```python
from tunrex.reward_comparison import ProgrammaticReward

# Wrap existing reward functions
format_reward = ProgrammaticReward(match_format_exactly)
answer_reward = ProgrammaticReward(check_answer)
quality_reward = ProgrammaticReward(reasoning_quality)

# Use in GRPO trainer
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        lambda p, c, **kw: format_reward.evaluate(p, c, **kw).scores,
        lambda p, c, **kw: answer_reward.evaluate(p, c, **kw).scores,
        lambda p, c, **kw: quality_reward.evaluate(p, c, **kw).scores,
    ],
    algo_config=grpo_config,
)
```

## Best Practices

1. **Start simple**: Begin with programmatic rewards, then add rubrics/preference models
2. **Use small samples**: Test on 50-100 examples before scaling
3. **Check correlations**: Ensure your rewards measure different aspects
4. **Find disagreements**: Edge cases reveal reward design issues
5. **Iterate**: Use comparison insights to improve reward functions
6. **Document**: Add descriptions to evaluators for clarity

## Requirements

Core functionality requires only:
- `numpy`
- `scipy` (for correlation analysis)

Optional visualization features require:
- `matplotlib`
- `seaborn`

## API Reference

See docstrings in source code for detailed API documentation:

```python
help(RewardComparison)
help(RewardAnalyzer)
help(ProgrammaticReward)
```

## Contributing

To add a new reward evaluator:

1. Subclass `BaseReward`
2. Implement `evaluate()` method
3. Implement `get_evaluator_type()` method
4. Return `RewardResult` with scores and metadata

Example:

```python
class MyCustomReward(BaseReward):
    def evaluate(self, prompts, completions, **kwargs):
        import time
        start = time.time()

        scores = [self.score_completion(c) for c in completions]

        elapsed_ms = (time.time() - start) * 1000
        metadata = self._create_metadata(elapsed_ms)

        return RewardResult(scores=scores, metadata=metadata)

    def get_evaluator_type(self):
        return "custom"
```

## License

Part of the TunRex project. See project LICENSE for details.
