# Training Pipeline Module

This module provides a light orchestration API for GRPO training in Google Colab notebooks.

## Philosophy

The notebook should be a **thin orchestration layer** that only:
- Installs dependencies
- Sets configuration values
- Calls high-level functions
- Displays outputs

All heavy logic lives in **reusable Python modules** that are:
- Version controlled
- Testable in isolation
- Easy to maintain and debug
- Reusable across projects

## Architecture

```
demo/train_colab.ipynb          # Light orchestration notebook
    ↓
src/training/colab_pipeline.py  # Heavy training logic
    ↓
TunRex/Tunix libraries          # Model and dataset utilities
```

## Module Structure

### Configuration (`ColabTrainingConfig`)

A dataclass that encapsulates all training hyperparameters:

```python
config = ColabTrainingConfig(
    num_batches=500,
    learning_rate=3e-6,
    lora_rank=64,
    use_openrubrics=True,
    save_to_drive=False,
)
```

**Features:**
- Type hints for all parameters
- Sensible defaults
- Automatic credential setup from multiple sources
- Computed fields (e.g., checkpoint directory)

### Session Preparation (`prepare_colab_session`)

Initializes the training environment:

```python
session = prepare_colab_session(config)
```

**What it does:**
1. Sets up credentials (Colab secrets, Kaggle secrets, or literals)
2. Mounts Google Drive (if requested)
3. Creates checkpoint directory
4. Loads base model and tokenizer
5. Creates LoRA model
6. Sets up sampler
7. Prepares datasets

**Returns:** `ColabSession` object with all initialized components

### Training (`train_grpo`)

Runs the GRPO training loop:

```python
trainer_state = train_grpo(config, session)
```

**What it does:**
1. Creates optimizer with warmup-cosine schedule
2. Configures GRPO algorithm
3. Sets up RL cluster
4. Creates checkpoint manager
5. Runs training loop
6. Saves checkpoints periodically

**Returns:** `TrainerState` object with trained model

### Export (`export_checkpoint`)

Exports checkpoints for local use:

```python
checkpoint_path = export_checkpoint(config, trainer_state)
```

**What it does:**
1. Finds latest checkpoint
2. Extracts LoRA parameters
3. Creates zip file (if saving to Drive)
4. Prints usage instructions

**Returns:** Path to latest checkpoint

### Quick Test (`quick_test`)

Tests the trained model:

```python
response = quick_test(config, session, test_question="...")
```

**What it does:**
1. Formats the test question
2. Generates response using trained model
3. Displays output

**Returns:** Model response string

## Usage in Notebooks

### Basic Usage

```python
# 1. Import
from src.training.colab_pipeline import (
    ColabTrainingConfig,
    prepare_colab_session,
    train_grpo,
    export_checkpoint,
    quick_test,
)

# 2. Configure
config = ColabTrainingConfig(
    num_batches=500,
    lora_rank=64,
)

# 3. Prepare
session = prepare_colab_session(config)

# 4. Train
trainer_state = train_grpo(config, session)

# 5. Export
checkpoint_path = export_checkpoint(config, trainer_state)

# 6. Test
quick_test(config, session)
```

### Full Pipeline

For convenience, use the `run_full_pipeline` function:

```python
from src.training.colab_pipeline import ColabTrainingConfig, run_full_pipeline

config = ColabTrainingConfig(num_batches=500)
session, trainer_state, checkpoint_path = run_full_pipeline(config)
```

## Development Workflow

### Editing the Pipeline

1. **Edit locally**:
   ```bash
   vim src/training/colab_pipeline.py
   ```

2. **Test locally** (optional):
   ```python
   python -m pytest tests/test_colab_pipeline.py
   ```

3. **Commit**:
   ```bash
   git add src/training/colab_pipeline.py
   git commit -m "Update training pipeline"
   ```

4. **Push**:
   ```bash
   git push origin your-branch
   ```

5. **Sync in Colab**:
   ```python
   !cd /content/ee596-fp && git pull origin your-branch
   ```

6. **Auto-reload** (already set up in notebook):
   ```python
   %load_ext autoreload
   %autoreload 2
   ```
   The module will automatically reload on next import!

### Adding New Features

#### Add a new dataset

1. Edit `prepare_datasets()` function:
   ```python
   def prepare_datasets(config: ColabTrainingConfig) -> Tuple[Any, Any]:
       if config.use_openrubrics:
           # ... existing code ...
       elif config.use_my_dataset:  # NEW
           from my_library import load_my_dataset
           train_data = load_my_dataset(max_examples=config.my_dataset_max)
       # ...
   ```

2. Add config parameters:
   ```python
   @dataclass
   class ColabTrainingConfig:
       # ... existing fields ...
       use_my_dataset: bool = False
       my_dataset_max: int = 1000
   ```

#### Add a new reward function

1. Edit `create_reward_functions()`:
   ```python
   def create_reward_functions(config: ColabTrainingConfig) -> List[Callable]:
       # ... existing reward functions ...

       def my_reward(prompts, completions, **kwargs):
           """My custom reward."""
           scores = []
           for completion in completions:
               score = compute_my_score(completion)
               scores.append(score)
           return scores

       return [match_format_reward, rar_reward, my_reward]  # Add it
   ```

#### Add a new training parameter

1. Add to config:
   ```python
   @dataclass
   class ColabTrainingConfig:
       # ... existing fields ...
       my_parameter: float = 0.5
   ```

2. Use in `train_grpo()`:
   ```python
   def train_grpo(config: ColabTrainingConfig, session: ColabSession):
       # ... existing code ...
       grpo_config = GRPOConfig(
           num_generations=config.num_generations,
           beta=config.beta,
           my_param=config.my_parameter,  # Use it
       )
   ```

## Benefits of This Architecture

### For Notebooks
- **Clean and readable** - Only ~15 cells instead of 30+
- **Less error-prone** - Less code to maintain
- **Self-documenting** - Function names explain what they do
- **Easy to modify** - Just change config values

### For Code
- **Reusable** - Use the same pipeline in multiple notebooks
- **Testable** - Can write unit tests for each function
- **Version controlled** - Track changes over time
- **Type safe** - Type hints catch errors early

### For Collaboration
- **Easy to review** - Logic in Python files, not notebooks
- **Easier to debug** - Can add print statements, debug locally
- **Better diffs** - Git diffs work properly on .py files
- **Documentation** - Docstrings explain each function

## File Organization

```
src/training/
├── __init__.py           # Package initialization, exports
├── colab_pipeline.py     # Main pipeline implementation
└── README.md            # This file

demo/
└── train_colab.ipynb    # Light orchestration notebook
```

## API Reference

### Data Classes

- `ColabTrainingConfig` - Training configuration
- `ColabSession` - Session state container
- `TrainerState` - Post-training state container

### Functions

- `prepare_colab_session(config)` → `ColabSession`
- `train_grpo(config, session)` → `TrainerState`
- `export_checkpoint(config, trainer_state)` → `str`
- `quick_test(config, session, test_question=None)` → `str`
- `run_full_pipeline(config)` → `Tuple[ColabSession, TrainerState, str]`

### Helper Functions

- `format_prompt(question, config, rubric=None)` → `str`
- `prepare_datasets(config)` → `Tuple[Dataset, Dataset]`
- `create_reward_functions(config)` → `List[Callable]`

## Examples

See `demo/train_colab.ipynb` for a complete working example.

## Future Enhancements

Potential additions:
- Support for other models (Llama, Mistral, etc.)
- Support for other datasets (GSM8K, MATH, etc.)
- Checkpoint resumption
- Distributed training configuration
- W&B experiment tracking integration
- Evaluation metrics computation
- Hyperparameter search integration

## Questions?

For questions or issues, please:
1. Check the notebook documentation
2. Read the function docstrings
3. Open an issue on GitHub
