# LLM Judge Module

LLM-based rubric generation and response scoring for GRPO training.

## Overview

This module provides:
- **RubricGenerator**: Generate evaluation rubrics dynamically using LLMs
- **ResponseScorer**: Score responses using LLM-as-judge techniques
- **GRPO-compatible reward functions**: Drop-in replacements for training pipelines

## Installation

The module is part of the `src` package. Required dependencies depend on which backend you use:

```bash
# For local HuggingFace models (default)
pip install transformers torch

# For OpenAI backend
pip install openai

# For Anthropic backend
pip install anthropic

# For vLLM backend (high-throughput inference)
pip install vllm
```

## Quick Start

```python
from src.llm_judge import (
    RubricGenerator,
    ResponseScorer,
    RubricGeneratorConfig,
    ResponseScorerConfig,
    configure,
    llm_judge_reward,
)

# Option 1: Configure globally for training
configure(
    generator_config=RubricGeneratorConfig.for_local(device="cuda"),
    scorer_config=ResponseScorerConfig.for_local(device="cuda"),
)

# Generate a rubric
generator = RubricGenerator()
rubric = generator.generate("What is the capital of France?")
print(rubric.to_text())

# Score a response
scorer = ResponseScorer()
result = scorer.score(
    question="What is the capital of France?",
    response="The capital of France is Paris.",
    rubric=rubric,
)
print(f"Score: {result.score}/{result.max_score}")

# Option 2: Use directly as GRPO reward function
scores = llm_judge_reward(
    prompts=["What is 2 + 2?"],
    completions=["2 + 2 equals 4."],
)
```

## Components

### RubricGenerator

Generates structured evaluation rubrics for questions/tasks.

```python
from src.llm_judge import RubricGenerator, RubricGeneratorConfig

# Using default config (local HuggingFace model)
generator = RubricGenerator()

# Custom config for OpenAI
config = RubricGeneratorConfig.for_openai(model="gpt-4o-mini")
generator = RubricGenerator(config)

# Generate rubric
rubric = generator.generate(
    question="Explain photosynthesis",
    question_type="science",  # Optional: auto-detected if not provided
)

# Access rubric properties
print(f"Type: {rubric.question_type}")
print(f"Criteria: {len(rubric.criteria)}")
print(f"Score range: {rubric.min_score}-{rubric.max_score}")
```

**Supported question types**: `math`, `creative`, `science`, `summarization`, `coding`, `reasoning`, `default`

### ResponseScorer

Scores responses using LLM-as-judge with multiple modes.

```python
from src.llm_judge import ResponseScorer, ResponseScorerConfig, JudgeMode

# Rubric-based scoring (default)
scorer = ResponseScorer()
result = scorer.score(
    question="What is 2 + 2?",
    response="The answer is 4.",
    rubric=rubric,  # Rubric object from RubricGenerator
)

# Reference-based scoring
config = ResponseScorerConfig(judge_mode=JudgeMode.REFERENCE_BASED)
scorer = ResponseScorer(config)
result = scorer.score(
    question="What is 2 + 2?",
    response="2 + 2 = 4",
    reference_answer="The sum of 2 and 2 is 4.",
)

# Access score details
print(f"Score: {result.score}")
print(f"Normalized: {result.normalized_score}")  # 0.0 to 1.0
print(f"Reasoning: {result.reasoning}")
```

### GRPO Reward Functions

Drop-in reward functions compatible with GRPO training pipelines.

```python
from src.llm_judge import (
    configure,
    llm_judge_reward,
    generate_and_score_reward,
    create_rubric_reward_fn,
)

# Configure once at startup
configure(
    generator_config=RubricGeneratorConfig.for_local(device="cuda"),
    scorer_config=ResponseScorerConfig.for_local(device="cuda"),
)

# Use in training loop
scores = llm_judge_reward(
    prompts=batch["prompt"],
    completions=batch["completion"],
)

# Combined reward with accuracy check
scores = generate_and_score_reward(
    prompts=batch["prompt"],
    completions=batch["completion"],
    answers=batch["answer"],  # Optional ground truth
)

# Create custom reward function
custom_reward = create_rubric_reward_fn(
    generator_config=RubricGeneratorConfig.for_openai(),
    scorer_config=ResponseScorerConfig.for_openai(),
    weight=0.5,
)
```

## Backends

### HuggingFace (default)

Local inference using transformers.

```python
config = RubricGeneratorConfig.for_local(
    model_id="google/gemma-3-1b-it",
    device="cuda",  # or "cpu", "auto"
)
```

### OpenAI

```python
# Requires OPENAI_API_KEY environment variable
config = RubricGeneratorConfig.for_openai(
    model="gpt-4o-mini",  # or "gpt-4o", "gpt-3.5-turbo"
)
```

### Anthropic

```python
# Requires ANTHROPIC_API_KEY environment variable
config = RubricGeneratorConfig.for_anthropic(
    model="claude-3-5-sonnet-20241022",
)
```

### vLLM

High-throughput local inference.

```python
config = RubricGeneratorConfig.for_vllm(
    model_id="google/gemma-3-1b-it",
    tensor_parallel_size=1,
)
```

### Custom Backend

```python
from src.llm_judge.backends import LLMBackend, register_backend

@register_backend("my_backend")
class MyBackend(LLMBackend):
    def generate(self, prompt, config=None):
        # Implementation
        pass

    def generate_batch(self, prompts, config=None):
        # Implementation
        pass

    def is_available(self):
        return True

    @property
    def backend_name(self):
        return "my_backend"
```

## Caching

Rubrics are cached to avoid regenerating for the same questions.

```python
generator = RubricGenerator(
    RubricGeneratorConfig(
        cache_enabled=True,
        cache_dir="./.rubric_cache",
    )
)

# Check cache stats
stats = generator.cache_stats()
print(f"Cached rubrics: {stats['num_items']}")

# Clear cache
generator.clear_cache()
```

## Configuration Reference

### RubricGeneratorConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"huggingface"` | Backend to use |
| `model_id` | `"google/gemma-3-1b-it"` | Model identifier |
| `max_new_tokens` | `1024` | Max tokens to generate |
| `temperature` | `0.3` | Sampling temperature |
| `cache_enabled` | `True` | Enable rubric caching |
| `cache_dir` | `"./.rubric_cache"` | Cache directory |
| `auto_detect_question_type` | `True` | Auto-detect question type |
| `num_criteria` | `5` | Number of rubric criteria |
| `score_range` | `(0, 10)` | Min/max score range |

### ResponseScorerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `"huggingface"` | Backend to use |
| `model_id` | `"google/gemma-3-1b-it"` | Model identifier |
| `judge_mode` | `RUBRIC_BASED` | Scoring mode |
| `max_new_tokens` | `512` | Max tokens to generate |
| `temperature` | `0.1` | Sampling temperature |
| `output_format` | `"json"` | Output format (`json`, `xml`, `plain`) |
| `include_reasoning` | `True` | Include reasoning in output |
| `max_retries` | `3` | Retry count for API backends |

## Module Structure

```
src/llm_judge/
├── __init__.py           # Public API exports
├── config.py             # Configuration dataclasses
├── rubric_generator.py   # RubricGenerator class
├── response_scorer.py    # ResponseScorer class
├── rewards.py            # GRPO reward functions
├── cache.py              # Rubric caching
├── parsing.py            # Output parsing (JSON, XML, plain)
├── backends/
│   ├── __init__.py       # Backend registry
│   ├── base.py           # Abstract LLMBackend
│   ├── huggingface.py    # HuggingFace backend
│   ├── openai.py         # OpenAI backend
│   ├── anthropic.py      # Anthropic backend
│   └── vllm.py           # vLLM backend
└── prompts/
    ├── __init__.py
    └── templates.py      # Prompt templates
```

## Version

Current version: `0.1.0`
