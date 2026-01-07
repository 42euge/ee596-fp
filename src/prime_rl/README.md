# PRIME RL Module

**Process-based Reinforcement with Intermediate Model Evaluation**

This module implements PRIME RL, a reinforcement learning approach that assigns rewards to intermediate reasoning steps rather than just final answers.

## Module Structure

```
src/prime_rl/
├── __init__.py              # Public API exports
├── config.py                # Configuration dataclasses
├── step_parser.py           # Parse reasoning into steps
├── step_evaluator.py        # Evaluate individual steps
├── process_reward.py        # Calculate trajectory rewards
├── rewards.py               # GRPO-compatible reward functions
├── utils.py                 # Utility functions
└── README.md                # This file
```

## Quick Start

```python
from src.prime_rl import PRIMEConfig, prime_rl_reward

# Configure
config = PRIMEConfig(gamma=0.95)

# Calculate rewards
rewards = prime_rl_reward(
    prompts=["What is 2+2?"],
    completions=["Step 1: Add 2+2=4\nAnswer: 4"],
    answer=["4"],
    config=config
)
```

## Key Components

### 1. Configuration (`config.py`)

- `PRIMEConfig`: Main configuration class
- `StepEvaluationMethod`: Enum for evaluation methods
- `RewardAggregation`: Enum for aggregation strategies
- `StepParsingStrategy`: Enum for parsing strategies

### 2. Step Parser (`step_parser.py`)

Extracts intermediate reasoning steps from completions.

**Strategies:**
- `NUMBERED`: "Step 1:", "Step 2:", etc.
- `LINE_BASED`: Each line is a step
- `SENTENCE_BASED`: Each sentence is a step
- `SEMANTIC`: Paragraph/idea-based chunking

### 3. Step Evaluator (`step_evaluator.py`)

Evaluates individual reasoning steps.

**Methods:**
- `RULE_BASED`: Pattern matching, format checks
- `SYMBOLIC`: Mathematical verification (sympy)
- `LLM_JUDGE`: AI-based quality assessment
- `HYBRID`: Combination of all methods

### 4. Process Reward Calculator (`process_reward.py`)

Calculates trajectory-based rewards with credit assignment.

**Features:**
- Multiple aggregation strategies
- Discount factor (gamma) for credit assignment
- Combination with outcome-based rewards
- Reward normalization

### 5. Reward Functions (`rewards.py`)

GRPO-compatible reward functions.

**Functions:**
- `prime_rl_reward()`: Main reward function
- `prime_rl_with_accuracy()`: Combined with accuracy checking
- `prime_rl_with_format()`: Combined with format checking
- `create_prime_rl_reward_suite()`: Create suite of rewards

## Usage Examples

### Basic Usage

```python
from src.prime_rl import PRIMEConfig, prime_rl_reward

config = PRIMEConfig(
    step_evaluation_method="hybrid",
    reward_aggregation="discounted_sum",
    gamma=0.95
)

rewards = prime_rl_reward(
    prompts=["Solve: x + 5 = 10"],
    completions=["Step 1: x = 10 - 5\nStep 2: x = 5\nAnswer: 5"],
    answer=["5"],
    config=config
)
```

### Training Integration

```python
from src.prime_rl import create_prime_rl_reward_suite

# Create reward functions
config = PRIMEConfig()
reward_fns = create_prime_rl_reward_suite(config)

# Use in GRPO training
trainer = GRPOLearner(
    model=model,
    reward_fns=reward_fns,
    ...
)
```

### Trajectory Analysis

```python
from src.prime_rl import ProcessRewardCalculator

calculator = ProcessRewardCalculator(config)
trajectory = calculator.calculate_trajectory_reward(
    prompt="What is 2+2?",
    completion="Step 1: Add 2+2\nStep 2: Result is 4",
    final_answer_reward=1.0
)

print(f"Steps: {trajectory.num_steps}")
print(f"Mean reward: {trajectory.mean_step_reward:.3f}")
print(f"Total: {trajectory.total_reward:.3f}")
```

## Configuration Options

### Essential Parameters

```python
PRIMEConfig(
    # Discount factor for credit assignment
    gamma=0.95,

    # How to evaluate steps
    step_evaluation_method="hybrid",

    # How to aggregate step rewards
    reward_aggregation="discounted_sum",

    # Enable/disable LLM judge
    llm_judge_enabled=False,

    # Weight for outcome vs process
    outcome_reward_weight=0.3,
)
```

### Advanced Parameters

```python
PRIMEConfig(
    # Step parsing
    step_parsing_strategy="numbered",
    min_step_length=10,
    max_steps=20,

    # Evaluation
    enable_symbolic_solver=True,
    llm_judge_model="gpt-4o-mini",

    # Reward processing
    normalize_rewards=True,
    baseline_subtraction=True,

    # Process supervision
    penalize_incorrect_steps=True,
    incorrect_step_penalty=-0.5,
)
```

## Testing

Run tests:

```bash
python -m pytest tests/test_prime_rl.py -v
```

## Examples

See `examples/prime_rl_demo.py` for comprehensive demonstrations.

## Documentation

Full documentation: `docs/PRIME_RL.md`

## Integration with Existing Code

PRIME RL integrates seamlessly with existing GRPO infrastructure:

```python
# Standard GRPO rewards
from TunRex.src.tunrex.datasets.rewards import (
    check_answer,
    match_format_exactly
)

# PRIME RL rewards
from src.prime_rl import create_prime_rl_reward_suite

# Combine
standard_rewards = [check_answer, match_format_exactly]
prime_rewards = create_prime_rl_reward_suite()
all_rewards = standard_rewards + prime_rewards

# Use in training
trainer = GRPOLearner(reward_fns=all_rewards, ...)
```

## Performance Considerations

### Computational Cost

- **Rule-based evaluation**: Very fast (~0.1ms/step)
- **Symbolic evaluation**: Fast (~1ms/step)
- **LLM judge**: Slow (~100-500ms/step), expensive

**Recommendation**: Use `hybrid` without LLM judge for most cases.

### Optimization Tips

1. **Disable LLM judge during development**:
   ```python
   config = PRIMEConfig(llm_judge_enabled=False)
   ```

2. **Limit max steps**:
   ```python
   config = PRIMEConfig(max_steps=10)
   ```

3. **Use caching for LLM judge**:
   ```python
   from src.llm_judge.cache import RubricCache
   cache = RubricCache(cache_dir=".cache/rubrics")
   ```

## Contributing

When adding new features:

1. Add configuration parameters to `config.py`
2. Implement logic in appropriate module
3. Export in `__init__.py`
4. Add tests to `tests/test_prime_rl.py`
5. Update documentation

## License

MIT License - see LICENSE file for details.
