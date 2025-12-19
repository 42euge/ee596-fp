"""
Colab Training Pipeline Module

This module encapsulates the heavy lifting for GRPO training in Google Colab,
exposing a simple API for orchestration notebooks.

Main Components:
1. ColabTrainingConfig - Configuration dataclass for all hyperparameters
2. prepare_colab_session - Environment setup and initialization
3. train_grpo - Main training loop
4. export_checkpoint - Convert checkpoints to HuggingFace format
5. quick_test - Test the trained model with a sample question

Usage in notebook:
    from src.training.colab_pipeline import (
        ColabTrainingConfig, prepare_colab_session, train_grpo,
        export_checkpoint, quick_test
    )

    config = ColabTrainingConfig(num_batches=500, lora_rank=64)
    session = prepare_colab_session(config)
    trainer_state = train_grpo(config, session)
    export_checkpoint(config, trainer_state)
    quick_test(config, session)
"""

import os
import re
import gc
import glob
import shutil
import string
import difflib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable
from collections import Counter

import jax
import jax.numpy as jnp
import grain
import optax
from flax import nnx
from orbax import checkpoint as ocp
from tqdm.auto import tqdm

from tunix.generate import sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import params, model
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class ColabTrainingConfig:
    """Complete configuration for GRPO training in Colab.

    Attributes:
        Training settings:
            num_batches: Number of training batches (500 = ~30 min on TPU)
            learning_rate: Learning rate for optimizer
            lora_rank: LoRA rank for efficient fine-tuning
            lora_alpha: LoRA alpha parameter

        Dataset settings:
            use_openrubrics: Use OpenRubrics dataset
            openrubrics_max: Max examples from OpenRubrics

        Checkpoint settings:
            save_to_drive: Save checkpoints to Google Drive
            experiment_name: Name for this experiment
            checkpoint_dir: Directory for saving checkpoints

        GRPO settings:
            num_generations: Number of generations per prompt
            num_iterations: Number of GRPO iterations
            beta: KL penalty coefficient
            epsilon: PPO clipping parameter

        Data settings:
            batch_size: Batch size for training
            max_prompt_length: Maximum prompt length
            total_generation_steps: Maximum generation length
            train_fraction: Fraction of data for training vs validation

        Prompt settings:
            reasoning_start: Start token for reasoning section
            reasoning_end: End token for reasoning section
            solution_start: Start token for solution section
            solution_end: End token for solution section
            system_prompt: System prompt template
    """

    # Training settings
    num_batches: int = 500
    learning_rate: float = 3e-6
    lora_rank: int = 64
    lora_alpha: float = 64.0

    # Dataset settings
    use_openrubrics: bool = True
    openrubrics_max: int = 2000

    # Checkpoint settings
    save_to_drive: bool = False
    experiment_name: str = "gemma3_grpo_reasoning"
    checkpoint_dir: Optional[str] = None

    # GRPO settings
    num_generations: int = 2
    num_iterations: int = 1
    beta: float = 0.08
    epsilon: float = 0.2

    # Data settings
    batch_size: int = 2
    max_prompt_length: int = 256
    total_generation_steps: int = 512
    train_fraction: float = 0.9

    # Prompt settings
    reasoning_start: str = "<reasoning>"
    reasoning_end: str = "</reasoning>"
    solution_start: str = "<answer>"
    solution_end: str = "</answer>"
    system_prompt: str = ""  # Will be set in __post_init__

    # Credentials (optional - can use secrets instead)
    wandb_api_key: Optional[str] = None
    kaggle_username: Optional[str] = None
    kaggle_key: Optional[str] = None

    def __post_init__(self):
        """Initialize derived configuration."""
        # Set checkpoint directory
        if self.checkpoint_dir is None:
            if self.save_to_drive:
                self.checkpoint_dir = f"/content/drive/MyDrive/{self.experiment_name}/checkpoints"
            else:
                self.checkpoint_dir = "/content/checkpoints"

        # Set system prompt if not already set
        if not self.system_prompt:
            self.system_prompt = (
                f"You are given a problem. Think carefully and show your detailed reasoning "
                f"step-by-step. Place your reasoning between {self.reasoning_start} and "
                f"{self.reasoning_end}. After completing your reasoning, provide the final "
                f"answer between {self.solution_start} and {self.solution_end}."
            )

    def setup_credentials(self):
        """Set up credentials from config or secrets providers.

        Priority:
        1. Config values (if provided)
        2. Google Colab secrets
        3. Kaggle secrets
        """
        # Set from config if provided
        if self.wandb_api_key:
            os.environ['WANDB_API_KEY'] = self.wandb_api_key
        if self.kaggle_username:
            os.environ['KAGGLE_USERNAME'] = self.kaggle_username
        if self.kaggle_key:
            os.environ['KAGGLE_KEY'] = self.kaggle_key

        # Try secrets providers if not set
        if not os.environ.get('KAGGLE_USERNAME'):
            # Try Google Colab secrets first
            try:
                from google.colab import userdata
                if not os.environ.get('WANDB_API_KEY'):
                    os.environ['WANDB_API_KEY'] = userdata.get('WANDB_API_KEY')
                if not os.environ.get('KAGGLE_USERNAME'):
                    os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
                if not os.environ.get('KAGGLE_KEY'):
                    os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
                print("✓ Using Google Colab secrets")
            except (ImportError, ModuleNotFoundError, Exception):
                # Fall back to Kaggle secrets
                try:
                    from kaggle_secrets import UserSecretsClient
                    secrets = UserSecretsClient()
                    if not os.environ.get('WANDB_API_KEY'):
                        os.environ['WANDB_API_KEY'] = secrets.get_secret('WANDB_API_KEY')
                    if not os.environ.get('KAGGLE_USERNAME'):
                        os.environ['KAGGLE_USERNAME'] = secrets.get_secret('KAGGLE_USERNAME')
                    if not os.environ.get('KAGGLE_KEY'):
                        os.environ['KAGGLE_KEY'] = secrets.get_secret('KAGGLE_KEY')
                    print("✓ Using Kaggle secrets")
                except (ImportError, ModuleNotFoundError, Exception):
                    print("⚠ WARNING: No credentials found")
        else:
            print("✓ Using credentials from environment")

    def get_prompt_template(self) -> str:
        """Get the prompt template string."""
        return (
            "<start_of_turn>user\n"
            "{system_prompt}\n\n"
            "{question}<end_of_turn>\n"
            "<start_of_turn>model"
        )


# =============================================================================
# 2. Session State Container
# =============================================================================

@dataclass
class ColabSession:
    """Container for session state and initialized objects."""
    config: ColabTrainingConfig
    tokenizer: Any
    ref_model: Any
    lora_policy: Any
    mesh: Any
    sampler: Any
    train_dataset: Any
    test_dataset: Any
    model_config: Any
    ckpt_path: str
    model_checkpoint_path: str


# =============================================================================
# 3. Environment Setup
# =============================================================================

def prepare_colab_session(config: ColabTrainingConfig) -> ColabSession:
    """Prepare the Colab session with all necessary setup.

    This function:
    1. Sets up credentials
    2. Mounts Google Drive (if requested)
    3. Creates checkpoint directory
    4. Loads the base model and tokenizer
    5. Creates the LoRA model
    6. Sets up the sampler
    7. Prepares datasets

    Args:
        config: Training configuration

    Returns:
        ColabSession with all initialized objects
    """
    print("=" * 60)
    print("PREPARING COLAB SESSION")
    print("=" * 60)

    # 1. Set up credentials
    config.setup_credentials()

    # 2. Mount Google Drive if requested
    if config.save_to_drive:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print(f"✓ Google Drive mounted")
        except ImportError:
            print("⚠ Not in Colab - skipping Drive mount")

    # 3. Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"✓ Checkpoint directory: {config.checkpoint_dir}")

    # 4. Load base model and tokenizer
    print("\n📦 Loading base model...")
    from tunrex import prepare_gemma_checkpoint, get_gemma_ref_model

    intermediate_ckpt_dir = "/tmp/intermediate_ckpt"
    ckpt_path, model_cp_path, tokenizer = prepare_gemma_checkpoint(
        ckpt_dir=intermediate_ckpt_dir,
    )
    print("✓ Base model checkpoint prepared")

    # 5. Load reference model
    print("📦 Loading reference model...")
    ref_model, mesh, model_config = get_gemma_ref_model(
        ckpt_path=ckpt_path,
        model_checkpoint_path=model_cp_path,
    )
    print("✓ Reference model loaded")
    print(f"  Devices: {jax.devices()}")

    # 6. Create LoRA model
    print(f"\n🔧 Creating LoRA model (rank={config.lora_rank}, alpha={config.lora_alpha})...")
    from tunrex import get_lora_model

    lora_policy = get_lora_model(
        base_model=ref_model,
        mesh=mesh,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
    )
    print("✓ LoRA model created")

    # 7. Create sampler
    print("\n🎲 Creating sampler...")
    sampler = sampler_lib.Sampler(
        model=lora_policy,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
        mesh=mesh,
    )
    print("✓ Sampler created")

    # 8. Prepare datasets
    print("\n📊 Loading datasets...")
    train_dataset, test_dataset = prepare_datasets(config)
    print(f"✓ Datasets prepared")

    print("\n" + "=" * 60)
    print("SESSION READY")
    print("=" * 60)

    return ColabSession(
        config=config,
        tokenizer=tokenizer,
        ref_model=ref_model,
        lora_policy=lora_policy,
        mesh=mesh,
        sampler=sampler,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_config=model_config,
        ckpt_path=ckpt_path,
        model_checkpoint_path=model_cp_path,
    )


# =============================================================================
# 4. Dataset Preparation
# =============================================================================

def format_prompt(question: str, config: ColabTrainingConfig, rubric: Optional[str] = None) -> str:
    """Format a question into the training prompt format.

    Args:
        question: The question to format
        config: Training configuration
        rubric: Optional rubric text

    Returns:
        Formatted prompt string
    """
    rubric_block = f"\nRubric:\n{rubric}\n\n" if rubric else ""
    template = config.get_prompt_template()
    return template.format(
        system_prompt=config.system_prompt,
        question=f"{rubric_block}{question}",
    )


def prepare_datasets(config: ColabTrainingConfig) -> Tuple[Any, Any]:
    """Load and prepare training and test datasets.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load data
    if config.use_openrubrics:
        from tunrex import load_openrubrics
        train_data = load_openrubrics(max_examples=config.openrubrics_max)
        print(f"  Loaded {len(train_data)} examples from OpenRubrics")
    else:
        train_data = []  # Add other dataset loading if needed
        print("  No dataset loaded")

    # Split into train/test
    split_idx = int(len(train_data) * config.train_fraction)
    train_split = train_data[:split_idx]
    test_split = train_data[split_idx:]

    print(f"  Train: {len(train_split)}, Test: {len(test_split)}")

    # Create grain datasets
    def create_dataset(data):
        return (
            grain.MapDataset.source(data)
            .shuffle(seed=42)
            .map(
                lambda x: {
                    "prompts": format_prompt(x["question"], config, x.get("rubric")),
                    "question": x["question"],
                    "rubric": x.get("rubric", ""),
                    "reference_response": x.get("reference_response", ""),
                }
            )
        )

    train_dataset = create_dataset(train_split)
    test_dataset = create_dataset(test_split)

    return train_dataset, test_dataset


# =============================================================================
# 5. Reward Functions
# =============================================================================

def create_reward_functions(config: ColabTrainingConfig) -> List[Callable]:
    """Create reward functions for GRPO training.

    Args:
        config: Training configuration

    Returns:
        List of reward functions
    """
    # Format matching regex
    match_format = re.compile(
        rf"{config.reasoning_start}.*?{config.reasoning_end}.*?"
        rf"{config.solution_start}.*?{config.solution_end}",
        re.DOTALL
    )

    def match_format_reward(prompts, completions, **kwargs):
        """Reward for proper format usage."""
        scores = []
        for completion in completions:
            if match_format.search(completion):
                scores.append(2.0)
            elif config.reasoning_start in completion or config.solution_start in completion:
                scores.append(0.5)
            else:
                scores.append(-1.0)
        return scores

    def rubric_overlap_score(response: str, rubric_text: str) -> float:
        """Calculate rubric overlap with TF-IDF weighting."""
        def tokenize(text):
            text = text.lower()
            for ch in string.punctuation:
                text = text.replace(ch, " ")
            return [t for t in text.split() if len(t) > 2]

        rubric_tokens = tokenize(rubric_text)
        response_tokens = set(tokenize(response))

        if not rubric_tokens:
            return 0.0

        token_counts = Counter(rubric_tokens)
        weighted_matches = sum(
            1.0 / token_counts[t] for t in response_tokens if t in token_counts
        )
        max_score = sum(1.0 / c for c in token_counts.values())

        coverage = weighted_matches / max_score if max_score > 0 else 0.0
        return coverage * 10.0

    def rar_reward(prompts, completions, rubric=None, reference_response=None, **kwargs):
        """Rubric-as-Reward scoring."""
        rubrics = rubric or [""] * len(completions)
        references = reference_response or [""] * len(completions)

        rewards = []
        for response, rub, ref in zip(completions, rubrics, references):
            # Rubric overlap (0-10)
            r_score = rubric_overlap_score(response, rub) if rub else 0.0

            # Reference similarity (0-5)
            f_score = difflib.SequenceMatcher(None, ref, response).ratio() * 5.0 if ref else 0.0

            rewards.append(r_score + f_score)

        return rewards

    return [match_format_reward, rar_reward]


# =============================================================================
# 6. Training
# =============================================================================

@dataclass
class TrainerState:
    """Container for trainer state after training."""
    grpo_trainer: GRPOLearner
    final_policy: Any
    checkpoint_dir: str


def train_grpo(config: ColabTrainingConfig, session: ColabSession) -> TrainerState:
    """Run GRPO training.

    Args:
        config: Training configuration
        session: Initialized Colab session

    Returns:
        TrainerState with trained model and trainer
    """
    print("\n" + "=" * 60)
    print("TRAINING SETUP")
    print("=" * 60)

    # 1. Create optimizer
    max_steps = int(config.num_batches * config.train_fraction * 0.94)
    warmup_steps = int(0.1 * max_steps)

    optimizer = optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            end_value=0.0,
        ),
        b1=0.9,
        b2=0.99,
        weight_decay=0.1,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=0.1),
        optimizer,
    )

    print(f"✓ Optimizer: AdamW with warmup-cosine schedule")
    print(f"  Max steps: {max_steps}, Warmup: {warmup_steps}")

    # 2. Create GRPO config
    grpo_config = GRPOConfig(
        num_generations=config.num_generations,
        num_iterations=config.num_iterations,
        beta=config.beta,
        epsilon=config.epsilon,
    )
    print(f"✓ GRPO config: generations={config.num_generations}, beta={config.beta}")

    # 3. Create cluster config
    cluster_config = rl_cluster_lib.ClusterConfig(
        max_prompt_length=config.max_prompt_length,
        total_generation_steps=config.total_generation_steps,
    )

    # 4. Create data iterator config
    data_iter_config = base_rollout.DataIteratorConfig(
        batch_size=config.batch_size,
        num_batches=config.num_batches,
    )
    print(f"✓ Data config: batch_size={config.batch_size}, num_batches={config.num_batches}")

    # 5. Create RL cluster
    rl_cluster = rl_cluster_lib.RLCluster(
        config=cluster_config,
        reference=session.ref_model,
        tokenizer=session.tokenizer,
        mesh=session.mesh,
        sampler=session.sampler,
    )
    print("✓ RL cluster created")

    # 6. Create checkpoint manager options
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=100,
        max_to_keep=3,
    )

    # 7. Create metrics logger options
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir="/tmp/tensorboard/grpo",
        flush_every_n_steps=20,
    )

    # 8. Create reward functions
    reward_fns = create_reward_functions(config)
    print(f"✓ Reward functions: {len(reward_fns)} functions")

    # 9. Create GRPO trainer
    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=grpo_config,
        optimizer=optimizer,
        ckpt_dir=config.checkpoint_dir,
        ckpt_options=checkpointing_options,
        metrics_logger_options=metrics_logging_options,
    )
    print(f"✓ GRPO trainer created")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")

    # 10. Create data iterator
    train_iter = session.train_dataset.batch(data_iter_config.batch_size)

    # 11. Run training
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    grpo_trainer.train(
        policy=session.lora_policy,
        data_iterator=train_iter,
        data_iterator_config=data_iter_config,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    return TrainerState(
        grpo_trainer=grpo_trainer,
        final_policy=session.lora_policy,
        checkpoint_dir=config.checkpoint_dir,
    )


# =============================================================================
# 7. Export Utilities
# =============================================================================

def export_checkpoint(config: ColabTrainingConfig, trainer_state: TrainerState) -> str:
    """Export checkpoint to HuggingFace format.

    Args:
        config: Training configuration
        trainer_state: Trainer state after training

    Returns:
        Path to exported checkpoint
    """
    print("\n" + "=" * 60)
    print("EXPORTING CHECKPOINT")
    print("=" * 60)

    # 1. Find latest checkpoint
    ckpt_dirs = sorted(glob.glob(f"{config.checkpoint_dir}/actor/*/"))
    if not ckpt_dirs:
        print("⚠ No checkpoints found!")
        return ""

    latest_ckpt = ckpt_dirs[-1]
    print(f"✓ Latest checkpoint: {latest_ckpt}")

    # 2. Create export directory
    export_dir = f"{config.checkpoint_dir}/hf_lora"
    os.makedirs(export_dir, exist_ok=True)

    # 3. Extract LoRA parameters
    lora_state = nnx.state(trainer_state.final_policy)
    lora_params = {}

    def extract_lora(path, value):
        path_str = ".".join(str(p) for p in path)
        if "lora" in path_str.lower():
            lora_params[path_str] = value

    jax.tree_util.tree_map_with_path(extract_lora, lora_state)
    print(f"✓ Found {len(lora_params)} LoRA parameters")

    # 4. Create zip file if saving to Drive
    if config.save_to_drive:
        import subprocess
        zip_path = f"{config.checkpoint_dir}/checkpoint_export.zip"

        # Change to checkpoint dir and zip
        subprocess.run(
            ["zip", "-r", "checkpoint_export.zip", "actor/"],
            cwd=config.checkpoint_dir,
            check=True,
        )
        print(f"✓ Zipped checkpoint: {zip_path}")
        print("  Download from Google Drive and extract to checkpoints/")

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nCheckpoint location: {config.checkpoint_dir}")
    print("\nTo use locally:")
    print("1. Download the checkpoint folder from Google Drive")
    print("2. Place in your local checkpoints/ directory")
    print("3. Run: python demo/demo.py --checkpoint ./checkpoints/actor/<step>/model_params")

    return latest_ckpt


# =============================================================================
# 8. Quick Test
# =============================================================================

def quick_test(config: ColabTrainingConfig, session: ColabSession,
               test_question: Optional[str] = None) -> str:
    """Run a quick test of the trained model.

    Args:
        config: Training configuration
        session: Colab session with trained model
        test_question: Optional custom test question

    Returns:
        Model response string
    """
    print("\n" + "=" * 60)
    print("QUICK TEST")
    print("=" * 60)

    # Default test question
    if test_question is None:
        test_question = "A store sells apples for $2 each. If I buy 5 apples, how much do I spend?"

    test_prompt = format_prompt(test_question, config)

    print(f"\nQuestion: {test_question}\n")
    print("Generating response...\n")

    response = session.sampler(
        [test_prompt],
        total_generation_steps=256,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )[0]

    print("Response:")
    print("-" * 60)
    print(response)
    print("-" * 60)

    return response


# =============================================================================
# 9. Convenience Functions
# =============================================================================

def run_full_pipeline(config: ColabTrainingConfig) -> Tuple[ColabSession, TrainerState, str]:
    """Run the complete training pipeline from start to finish.

    This is a convenience function that runs:
    1. Session preparation
    2. Training
    3. Checkpoint export
    4. Quick test

    Args:
        config: Training configuration

    Returns:
        Tuple of (session, trainer_state, checkpoint_path)
    """
    # Prepare session
    session = prepare_colab_session(config)

    # Train
    trainer_state = train_grpo(config, session)

    # Export
    checkpoint_path = export_checkpoint(config, trainer_state)

    # Test
    quick_test(config, session)

    return session, trainer_state, checkpoint_path
