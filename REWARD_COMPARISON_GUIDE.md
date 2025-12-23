# Reward Comparison Tools - User Guide

## Overview

This guide explains how to use the new reward comparison framework to compare different reward methodologies (preference models, rubrics, programmatic rewards) and understand their effects on training.

## What is Reward Comparison?

When training models with reinforcement learning (like GRPO), choosing the right reward function is crucial. Different reward methodologies have different strengths:

- **Programmatic rewards**: Fast, deterministic, rule-based (e.g., format checking, keyword matching)
- **Rubric-based rewards**: Multi-criteria evaluation with weights (e.g., accuracy + clarity + completeness)
- **Preference models**: Model-based scoring (e.g., LLM-as-judge, trained reward models)

The reward comparison framework helps you:
1. ✅ Compare multiple reward methods on the same data
2. 📊 Analyze correlations and agreement between methods
3. 🔍 Find edge cases where methods disagree
4. 📈 Visualize score distributions and relationships
5. 🎯 Choose the best reward for your use case

## Quick Start

### Installation

The reward comparison framework is part of TunRex:

```bash
cd TunRex
pip install -e .
```

Dependencies:
- `numpy` (core functionality)
- `scipy` (correlation analysis)
- `matplotlib` (optional, for visualizations)
- `seaborn` (optional, for heatmaps)

### Basic Usage

```python
from tunrex.reward_comparison import (
    ProgrammaticReward,
    RubricReward,
    RubricCriterion,
    RewardComparison,
    RewardAnalyzer,
)

# 1. Define your reward evaluators
format_reward = ProgrammaticReward(
    func=check_format_function,
    name="FormatCheck"
)

accuracy_reward = ProgrammaticReward(
    func=check_accuracy_function,
    name="AccuracyCheck"
)

# 2. Create a comparison
comparison = RewardComparison([format_reward, accuracy_reward])

# 3. Evaluate on your data
result = comparison.evaluate(
    prompts=your_prompts,
    completions=your_completions,
    # Pass any additional metadata your rewards need
    answer=ground_truth_answers
)

# 4. Analyze results
analyzer = RewardAnalyzer(result)
print(analyzer.generate_report())

# 5. Visualize (optional)
analyzer.plot_correlation_heatmap("heatmap.png")
analyzer.plot_score_distributions("distributions.png")
```

## Examples

We provide two comprehensive examples:

### Example 1: Basic Comparison (`examples/reward_comparison_example.py`)

This example demonstrates:
- Using existing programmatic rewards from TunRex
- Creating rubric-based rewards
- Running comparisons and generating reports
- Exporting results to CSV
- Creating visualizations

Run it:
```bash
python examples/reward_comparison_example.py
```

### Example 2: Advanced Usage (`examples/reward_comparison_advanced.py`)

This example shows:
- Creating custom reward evaluators
- Simulating LLM-as-judge evaluators
- Comparing against human preferences
- Finding disagreements between methods
- Analyzing which rewards align best with desired outcomes

Run it:
```bash
python examples/reward_comparison_advanced.py
```

## Creating Custom Reward Evaluators

### Option 1: Programmatic Rewards

Wrap any function that takes `(prompts, completions, **kwargs)` and returns a list of scores:

```python
def my_custom_reward(prompts, completions, **kwargs):
    scores = []
    for completion in completions:
        # Your scoring logic
        score = compute_score(completion)
        scores.append(score)
    return scores

reward = ProgrammaticReward(
    func=my_custom_reward,
    name="MyCustomReward",
    description="What this reward measures"
)
```

### Option 2: Rubric-Based Rewards

Define multiple criteria with weights:

```python
from tunrex.reward_comparison import RubricCriterion, RubricReward

criteria = [
    RubricCriterion(
        name="Accuracy",
        description="Is the answer correct?",
        max_score=5.0,
        evaluator=lambda p, c, **kw: 5.0 if check_correct(c, kw['answer']) else 0.0
    ),
    RubricCriterion(
        name="Clarity",
        description="Is the explanation clear?",
        max_score=3.0,
        evaluator=lambda p, c, **kw: rate_clarity(c)
    ),
    RubricCriterion(
        name="Completeness",
        description="Is the response complete?",
        max_score=2.0,
        evaluator=lambda p, c, **kw: 2.0 if is_complete(c) else 1.0
    ),
]

rubric = RubricReward(
    criteria=criteria,
    name="QualityRubric",
    aggregation="sum"  # or "mean"
)
```

### Option 3: Preference Models (LLM-as-Judge)

Use an LLM or trained model to score completions:

```python
from tunrex.reward_comparison import PreferenceModelReward

def gpt4_judge(prompts, completions, **kwargs):
    """Score completions using GPT-4."""
    scores = []
    for prompt, completion in zip(prompts, completions):
        # Call GPT-4 API with a scoring prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Rate this response from 0-10 for quality."
            }, {
                "role": "user",
                "content": f"Question: {prompt}\nAnswer: {completion}"
            }]
        )
        score = extract_score(response)
        scores.append(score)
    return scores

llm_judge = PreferenceModelReward(
    model_fn=gpt4_judge,
    name="GPT4Judge",
    model_type="llm_judge"
)
```

### Option 4: Custom Evaluator Class

For complex logic, subclass `BaseReward`:

```python
from tunrex.reward_comparison import BaseReward, RewardResult
import time

class MyComplexReward(BaseReward):
    def __init__(self, config, name="ComplexReward"):
        super().__init__(name=name)
        self.config = config

    def evaluate(self, prompts, completions, **kwargs):
        start_time = time.time()

        scores = []
        details = []

        for prompt, completion in zip(prompts, completions):
            # Your complex scoring logic
            score, feedback = self.score_with_details(prompt, completion, kwargs)
            scores.append(score)
            details.append(feedback)

        elapsed_ms = (time.time() - start_time) * 1000
        metadata = self._create_metadata(elapsed_ms)

        return RewardResult(
            scores=scores,
            metadata=metadata,
            details=details
        )

    def get_evaluator_type(self):
        return "custom"

    def score_with_details(self, prompt, completion, metadata):
        # Your implementation
        return score, feedback_dict
```

## Analysis Features

### Correlation Analysis

Understand how different rewards relate:

```python
analyzer = RewardAnalyzer(comparison_result)
corr = analyzer.compute_correlations()

# Get specific correlation
pearson_r = corr.get_pairwise_correlation("Reward1", "Reward2", method="pearson")
print(f"Correlation: {pearson_r:.3f}")

# Print matrix
print(corr.format_matrix("pearson"))
```

### Agreement Analysis

Measure how often rewards agree:

```python
# Binary agreement (above threshold)
agreement = analyzer.compute_agreement(threshold=5.0)

# Top-k agreement (which samples rank highest)
agreement = analyzer.compute_agreement(top_k=[5, 10, 20])

print(agreement.format_summary())
```

### Finding Disagreements

Identify edge cases:

```python
disagreements = comparison.find_disagreements(threshold=2.0)

for dis in disagreements[:5]:
    print(f"Sample {dis['sample_idx']}:")
    print(f"  {dis['evaluator1']}: {dis['score1']:.2f}")
    print(f"  {dis['evaluator2']}: {dis['score2']:.2f}")
    print(f"  Difference: {dis['difference']:.2f}")
    print(f"  Text: {dis['completion'][:100]}...")
```

### Visualization

Generate plots to understand reward behavior:

```python
# Score distributions
analyzer.plot_score_distributions("distributions.png")

# Correlation heatmap
analyzer.plot_correlation_heatmap("heatmap.png", method="pearson")

# Pairwise comparison
analyzer.plot_score_comparison("Reward1", "Reward2", "scatter.png")
```

### Export Results

Save results for further analysis:

```python
# Export to CSV
comparison.export_to_csv(
    "results.csv",
    include_text=True  # Include prompts and completions
)

# Generate text report
report = analyzer.generate_report(
    include_correlations=True,
    include_agreement=True,
    agreement_threshold=5.0,
    top_k=[5, 10]
)

with open("report.txt", "w") as f:
    f.write(report)
```

## Integration with GRPO Training

Use reward comparison to select and validate rewards for training:

```python
# Step 1: Compare candidate rewards
comparison = RewardComparison([reward1, reward2, reward3])
result = comparison.evaluate(dev_prompts, dev_completions)

# Step 2: Analyze which rewards are most suitable
analyzer = RewardAnalyzer(result)
report = analyzer.generate_report()

# Step 3: Choose rewards based on analysis
# e.g., rewards with high correlation to human preferences
# and good coverage of different aspects

# Step 4: Use selected rewards in GRPO
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        lambda p, c, **kw: reward1.evaluate(p, c, **kw).scores,
        lambda p, c, **kw: reward3.evaluate(p, c, **kw).scores,
    ],
    algo_config=grpo_config,
)
```

## Best Practices

1. **Start with a small sample**: Test on 50-100 examples before scaling to full dataset

2. **Use diverse data**: Include easy, medium, and hard examples to test reward robustness

3. **Check correlations**: High correlation might mean redundant rewards; low correlation might mean complementary aspects

4. **Examine disagreements**: Edge cases reveal reward design issues

5. **Validate against goals**: Compare programmatic rewards to human preferences or desired outcomes

6. **Iterate**: Use insights from comparison to improve reward design

7. **Document**: Add clear names and descriptions to make results interpretable

## Common Use Cases

### Use Case 1: Choosing Between Format Checkers

```python
exact_format = create_format_reward(r"<answer>.+</answer>", name="Exact")
flexible_format = create_format_reward(r"answer:", name="Flexible")

comparison = RewardComparison([exact_format, flexible_format])
result = comparison.evaluate(samples)

# Which catches more valid responses?
print(comparison.summarize())
```

### Use Case 2: Validating Reward Against Human Ratings

```python
my_reward = ProgrammaticReward(my_scoring_fn)
human_ratings = load_human_ratings()  # Load actual human scores

comparison = RewardComparison([my_reward, human_ratings])
result = comparison.evaluate(samples)

# How well does my reward align?
analyzer = RewardAnalyzer(result)
corr = analyzer.compute_correlations()
print(f"Alignment: {corr.get_pairwise_correlation('my_reward', 'human'):.3f}")
```

### Use Case 3: Designing Ensemble Rewards

```python
# Compare multiple aspects
format_reward = ProgrammaticReward(check_format)
accuracy_reward = ProgrammaticReward(check_accuracy)
quality_reward = ProgrammaticReward(check_quality)

comparison = RewardComparison([format_reward, accuracy_reward, quality_reward])
result = comparison.evaluate(samples)

# Check correlations to avoid redundancy
analyzer = RewardAnalyzer(result)
corr = analyzer.compute_correlations()

# Create weighted ensemble based on analysis
def ensemble(prompts, completions, **kwargs):
    r1 = format_reward.evaluate(prompts, completions, **kwargs).scores
    r2 = accuracy_reward.evaluate(prompts, completions, **kwargs).scores
    r3 = quality_reward.evaluate(prompts, completions, **kwargs).scores
    return [0.3*s1 + 0.4*s2 + 0.3*s3 for s1, s2, s3 in zip(r1, r2, r3)]
```

## Troubleshooting

### Issue: Import errors

If you see `ModuleNotFoundError`, ensure TunRex is installed:
```bash
cd TunRex
pip install -e .
```

### Issue: Visualization errors

If plots fail, install optional dependencies:
```bash
pip install matplotlib seaborn
```

### Issue: Memory issues with large datasets

Use batching for large comparisons:
```python
# Process in batches
batch_size = 100
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    batch_completions = completions[i:i+batch_size]
    result = comparison.evaluate(batch_prompts, batch_completions)
    # Process results...
```

## API Documentation

Full API documentation is available in docstrings:

```python
from tunrex.reward_comparison import RewardComparison
help(RewardComparison)
```

See also:
- `TunRex/src/tunrex/reward_comparison/README.md` - Module README
- Module source code for detailed implementation

## Contributing

To contribute new reward evaluators or analysis methods:

1. Subclass `BaseReward` for new evaluator types
2. Add helper functions to `evaluators.py`
3. Add new analysis methods to `RewardAnalyzer`
4. Include examples in `examples/`
5. Update documentation

## Citation

If you use this framework in research, please cite:

```bibtex
@software{tunrex_reward_comparison,
  title={TunRex Reward Comparison Framework},
  author={TunRex Contributors},
  year={2025},
  url={https://github.com/yourusername/ee596-fp}
}
```

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check existing examples in `examples/`
- Review documentation in `TunRex/src/tunrex/reward_comparison/`
