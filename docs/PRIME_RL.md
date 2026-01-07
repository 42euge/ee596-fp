# PRIME RL: Process-based Reinforcement with Intermediate Model Evaluation

## Table of Contents

- [What is PRIME RL?](#what-is-prime-rl)
- [Why PRIME RL?](#why-prime-rl)
- [How PRIME RL Works](#how-prime-rl-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Comparison with Standard RL](#comparison-with-standard-rl)

---

## What is PRIME RL?

**PRIME RL** (Process-based Reinforcement with Intermediate Model Evaluation) is a reinforcement learning paradigm for training reasoning models where **rewards are assigned to intermediate reasoning steps**, not just the final answer.

### Core Idea (One Sentence)

> PRIME RL decomposes reasoning into explicit intermediate steps and applies step-level rewards, enabling the model to learn **how to reason**, not just **what to answer**.

### Key Innovation

Instead of treating reasoning as an opaque sequence judged only at the end, PRIME RL:
- ✅ Evaluates **each reasoning step** independently
- ✅ Provides **fine-grained feedback** on the reasoning process
- ✅ Enables **better credit assignment** for long-horizon reasoning
- ✅ Encourages **stable, generalizable reasoning patterns**

---

## Why PRIME RL?

### Problems with Standard RL for Reasoning

Standard RL approaches (RLHF, GRPO, PPO) for language models:

| Issue | Description | Impact |
|-------|-------------|--------|
| **Opaque Reasoning** | Rewards only the final output | Cannot distinguish correct vs. flawed reasoning |
| **Poor Credit Assignment** | No signal for intermediate steps | Models learn shortcuts instead of reasoning |
| **Reward Hacking** | Correct answer via wrong reasoning gets rewarded | Brittle, non-generalizable solutions |
| **Long Horizon Problem** | Sparse rewards for multi-step reasoning | Slow learning, unstable training |

### How PRIME RL Solves These

```
Standard RL:          PRIME RL:

Question → [Black Box Reasoning] → Answer        Question → Step 1 → Step 2 → ... → Step T → Answer
            ↓                                              ↓      ↓             ↓       ↓
          r_final                                        r_1    r_2          r_T-1   r_T
                                                            ↓
                                                    Better Credit Assignment
```

**PRIME RL provides:**
1. **Reduced Reward Hacking**: Incorrect logic is penalized even if final answer is correct
2. **Stable Multi-Step Reasoning**: Models learn reusable reasoning primitives
3. **Improved Generalization**: Process optimization leads to better extrapolation
4. **Interpretability**: Step-level rewards reveal what the model learned

---

## How PRIME RL Works

### 1. Structured Reasoning Output

The model emits reasoning in structured steps:

```
<reasoning>
Step 1: Identify known quantities: x = 5, y = 3
Step 2: Apply formula: z = x + y
Step 3: Calculate: z = 5 + 3 = 8
</reasoning>
<answer>8</answer>
```

### 2. Step-Level Evaluation

Each step is evaluated using one or more methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| **Rule-Based** | Pattern matching, format checks | Fast, deterministic validation |
| **Symbolic** | Mathematical verification (sympy) | Verify equations and calculations |
| **LLM Judge** | AI-based quality assessment | Semantic correctness, clarity |
| **Hybrid** | Combination of all methods | Comprehensive evaluation |

Example evaluation:
```python
Step 1: "Identify x = 5, y = 3"
  ✓ Rule-based: Has variable definitions → +0.3
  ✓ Format: Complete sentence → +0.1
  ✓ LLM Judge: Clear and correct → +0.5
  → Total: r_1 = 0.9
```

### 3. Credit Assignment Across Reasoning

Instead of a single reward `r_final`, PRIME RL uses trajectory rewards:

```
∇θ E[∑(t=1 to T) γ^t r_t]
```

Where:
- `r_t`: Reward for step `t`
- `γ`: Discount factor (typically 0.95)
- `T`: Total number of steps

**Aggregation strategies:**
- **Discounted Sum**: `∑ γ^t r_t` (recommended)
- **Mean**: `(∑ r_t) / T`
- **Weighted Mean**: Later steps weighted more
- **Min**: Strictest step determines reward

### 4. Integration with RL Algorithms

PRIME RL is **algorithm-agnostic** and works with:
- ✅ PPO (Proximal Policy Optimization)
- ✅ GRPO (Group Relative Policy Optimization)
- ✅ Actor-Critic methods
- ✅ Any policy gradient algorithm

**The key difference is WHERE rewards come from, not the optimizer.**

---

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy torch transformers

# Optional: Symbolic evaluation
pip install sympy

# Optional: LLM Judge
pip install openai anthropic
```

### Install PRIME RL

PRIME RL is included in this repository:

```bash
cd ee596-fp
pip install -e .
```

---

## Quick Start

### Basic Usage

```python
from src.prime_rl import PRIMEConfig, prime_rl_reward

# Configure PRIME RL
config = PRIMEConfig(
    step_parsing_strategy="numbered",      # How to parse steps
    step_evaluation_method="hybrid",       # How to evaluate steps
    reward_aggregation="discounted_sum",   # How to aggregate rewards
    gamma=0.95,                           # Discount factor
)

# Calculate rewards (GRPO-compatible)
prompts = ["What is 2 + 2?"]
completions = ["""<reasoning>
Step 1: Add 2 + 2
Step 2: Result is 4
</reasoning>
<answer>4</answer>"""]

rewards = prime_rl_reward(
    prompts=prompts,
    completions=completions,
    answer=["4"],
    config=config
)

print(rewards)  # [0.87]  (example)
```

### Training with PRIME RL

```python
from src.prime_rl import create_prime_rl_reward_suite
from TunRex.src.tunrex.datasets.rewards import check_answer

# Create PRIME RL reward functions
config = PRIMEConfig(gamma=0.95)
prime_rewards = create_prime_rl_reward_suite(config)

# Combine with existing rewards
all_rewards = [check_answer] + prime_rewards

# Use in GRPO training
from tunix import GRPOLearner

trainer = GRPOLearner(
    model=model,
    reward_fns=all_rewards,  # ← PRIME RL rewards
    ...
)

trainer.train(train_ds, val_ds)
```

### Command-Line Training

```bash
# Train with default PRIME RL configuration
python scripts/train_prime_rl.py

# Train with custom settings
python scripts/train_prime_rl.py \
    --gamma 0.95 \
    --step_evaluation_method hybrid \
    --reward_aggregation discounted_sum \
    --llm_judge_enabled \
    --llm_judge_model gpt-4o-mini
```

---

## Configuration

### PRIMEConfig Parameters

```python
from src.prime_rl import PRIMEConfig

config = PRIMEConfig(
    # === Step Parsing ===
    step_parsing_strategy="numbered",     # How to split reasoning into steps
    # Options: "numbered", "line_based", "sentence_based", "semantic"

    min_step_length=10,                   # Minimum characters per step
    max_steps=20,                         # Maximum steps to extract

    # === Step Evaluation ===
    step_evaluation_method="hybrid",      # How to evaluate steps
    # Options: "rule_based", "symbolic", "llm_judge", "hybrid"

    enable_symbolic_solver=True,          # Use sympy for math verification
    llm_judge_enabled=False,              # Use LLM for step evaluation (costly)
    llm_judge_model="gpt-4o-mini",       # LLM model for evaluation

    # === Reward Aggregation ===
    reward_aggregation="discounted_sum",  # How to combine step rewards
    # Options: "sum", "discounted_sum", "mean", "weighted_mean", "min"

    gamma=0.95,                           # Discount factor (0-1)
    step_reward_scale=1.0,                # Scale factor for step rewards
    final_answer_weight=2.0,              # Extra weight for correct final answer

    # === Credit Assignment ===
    normalize_rewards=True,               # Normalize to [-1, 1]
    baseline_subtraction=True,            # Subtract baseline for variance reduction

    # === Outcome Combination ===
    combine_with_outcome_rewards=True,    # Combine step + outcome rewards
    outcome_reward_weight=0.3,            # Weight for outcome (0-1)
    # Process weight = 1 - outcome_weight

    # === Process Supervision ===
    penalize_incorrect_steps=True,        # Penalize wrong intermediate steps
    incorrect_step_penalty=-0.5,          # Penalty magnitude
    reward_correct_process=True,          # Reward good process even if answer wrong
)
```

### Recommended Configurations

#### For Mathematical Reasoning (GSM8K, MATH)
```python
config = PRIMEConfig(
    step_parsing_strategy="numbered",
    step_evaluation_method="hybrid",
    reward_aggregation="discounted_sum",
    gamma=0.95,
    enable_symbolic_solver=True,
    combine_with_outcome_rewards=True,
    outcome_reward_weight=0.3,
)
```

#### For Pure Process Optimization
```python
config = PRIMEConfig(
    step_evaluation_method="hybrid",
    use_step_rewards_only=True,           # Ignore final answer
    reward_correct_process=True,
    penalize_incorrect_steps=True,
)
```

#### For Fast Iteration (No LLM Judge)
```python
config = PRIMEConfig(
    step_evaluation_method="rule_based",  # Fast, no API calls
    llm_judge_enabled=False,
    enable_symbolic_solver=True,          # Still verify math
)
```

---

## Usage Examples

### Example 1: Analyze Trajectory

```python
from src.prime_rl import ProcessRewardCalculator, PRIMEConfig

config = PRIMEConfig()
calculator = ProcessRewardCalculator(config)

# Calculate trajectory reward
trajectory = calculator.calculate_trajectory_reward(
    prompt="What is 15 + 27?",
    completion="""<reasoning>
Step 1: Identify the numbers: 15 and 27
Step 2: Add them: 15 + 27 = 42
Step 3: Verify: 42 is correct
</reasoning>
<answer>42</answer>""",
    final_answer_reward=1.0,
    answer="42"
)

# Analyze
print(f"Number of steps: {trajectory.num_steps}")
print(f"Mean step reward: {trajectory.mean_step_reward:.3f}")
print(f"Total reward: {trajectory.total_reward:.3f}")

# Inspect individual steps
for i, step_reward in enumerate(trajectory.step_rewards):
    print(f"Step {i+1}: {step_reward.reward:.3f}")
    print(f"  Text: {step_reward.step_text}")
    print(f"  Method: {step_reward.evaluation_method}")
```

### Example 2: Batch Processing

```python
from src.prime_rl import prime_rl_reward

prompts = [
    "Calculate 5 × 3",
    "What is 100 - 37?",
    "Solve 2^3"
]

completions = [
    "Step 1: 5 × 3 = 15\nAnswer: 15",
    "Step 1: 100 - 37 = 63\nAnswer: 63",
    "Step 1: 2^3 = 2×2×2 = 8\nAnswer: 8"
]

answers = ["15", "63", "8"]

rewards = prime_rl_reward(prompts, completions, answer=answers)
print(rewards)  # [0.92, 0.88, 0.95]
```

### Example 3: Custom Step Evaluation

```python
from src.prime_rl import StepParser, StepEvaluator, PRIMEConfig
from src.prime_rl.step_parser import ParsedStep

config = PRIMEConfig(step_evaluation_method="hybrid")
evaluator = StepEvaluator(config)

# Custom step
step = ParsedStep(
    index=0,
    text="Using the Pythagorean theorem: a² + b² = c²"
)

# Evaluate with context
reward = evaluator.evaluate_step(
    step=step,
    context={
        "question": "Find the hypotenuse of a right triangle with sides 3 and 4",
        "answer": "5"
    }
)

print(f"Reward: {reward.reward:.3f}")
print(f"Details: {reward.details}")
```

---

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    PRIME RL System                       │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌─────────────┐  ┌──────────────────┐  ┌──────────────┐
│ StepParser  │  │ StepEvaluator    │  │ ProcessReward│
│             │  │                  │  │ Calculator   │
│ • Numbered  │  │ • Rule-Based     │  │              │
│ • Line-Based│  │ • Symbolic       │  │ • Discounted │
│ • Semantic  │  │ • LLM Judge      │  │ • Mean       │
│ • Sentence  │  │ • Hybrid         │  │ • Weighted   │
└─────────────┘  └──────────────────┘  └──────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                  ┌─────────────────┐
                  │  GRPO-Compatible│
                  │  Reward Function│
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ GRPO/PPO Trainer│
                  └─────────────────┘
```

### Data Flow

1. **Input**: Prompt + Completion
2. **Step Parsing**: Extract intermediate steps
3. **Step Evaluation**: Score each step
4. **Aggregation**: Combine step rewards with credit assignment
5. **Combination**: Mix with outcome-based rewards
6. **Output**: Single reward value for RL

---

## API Reference

### Core Classes

#### `PRIMEConfig`

Configuration for PRIME RL.

```python
config = PRIMEConfig(
    gamma=0.95,
    step_evaluation_method="hybrid",
    reward_aggregation="discounted_sum"
)
```

#### `StepParser`

Parses completions into intermediate steps.

```python
parser = StepParser(config)
steps = parser.parse(completion)
```

#### `StepEvaluator`

Evaluates individual reasoning steps.

```python
evaluator = StepEvaluator(config)
reward = evaluator.evaluate_step(step, context)
```

#### `ProcessRewardCalculator`

Calculates trajectory-based rewards.

```python
calculator = ProcessRewardCalculator(config)
trajectory = calculator.calculate_trajectory_reward(
    prompt, completion, final_answer_reward
)
```

### Reward Functions

#### `prime_rl_reward()`

Main GRPO-compatible reward function.

```python
rewards = prime_rl_reward(
    prompts,
    completions,
    answer=answers,
    config=config
)
```

#### `create_prime_rl_reward_suite()`

Creates suite of PRIME RL rewards.

```python
reward_fns = create_prime_rl_reward_suite(
    config,
    include_format=True,
    include_accuracy=True,
    include_pure_prime=True
)
```

---

## Comparison with Standard RL

| Aspect | Standard RL (RLHF/GRPO) | PRIME RL |
|--------|------------------------|----------|
| **Reward Granularity** | Final output only | Intermediate steps |
| **Credit Assignment** | Sparse (one reward) | Dense (reward per step) |
| **Reward Hacking** | Easy (shortcut to answer) | Reduced (process matters) |
| **Generalization** | Often brittle | More robust |
| **Training Stability** | Can be unstable | Improved stability |
| **Interpretability** | Opaque | Step-level insights |
| **Computational Cost** | Lower | Moderate (step evaluation) |
| **Use Case** | General chat/completion | Reasoning tasks |

### When to Use PRIME RL

✅ **Use PRIME RL when:**
- Training reasoning models (math, logic, code)
- Multi-step problem solving is required
- Process quality matters as much as final answer
- You want interpretable learning signals
- Generalization to harder problems is important

❌ **Don't use PRIME RL when:**
- Simple completion tasks (translation, summarization)
- Process doesn't matter (only outcome matters)
- Computational budget is very limited
- Reasoning steps aren't well-defined

---

## Advanced Topics

### Custom Step Evaluation

Implement your own step evaluator:

```python
from src.prime_rl.step_evaluator import StepEvaluation
from src.prime_rl.config import StepEvaluationMethod

def my_custom_evaluator(step, context):
    # Your custom logic
    score = your_scoring_function(step.text)

    return StepEvaluation(
        step_index=step.index,
        method=StepEvaluationMethod.RULE_BASED,
        score=score,
        passed=score > 0.5,
        details={"custom_metric": score}
    )
```

### Integration with LLM Judge

Use LLM Judge for step-aware rubrics:

```python
from src.llm_judge.step_rubrics import StepRubricGenerator

generator = StepRubricGenerator()
rubrics = generator.generate_trajectory_rubrics(
    question="What is 2+2?",
    steps=["Add 2+2", "Result is 4"]
)

for rubric in rubrics:
    print(f"Step {rubric.step_index}: {rubric.step_type}")
    print(f"Criteria: {rubric.criteria}")
```

---

## Citation

If you use PRIME RL in your research, please cite:

```bibtex
@software{prime_rl_2025,
  title={PRIME RL: Process-based Reinforcement with Intermediate Model Evaluation},
  author={EE596 Final Project Team},
  year={2025},
  url={https://github.com/42euge/ee596-fp}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check documentation in `docs/`
- Review examples in `tests/test_prime_rl.py`
