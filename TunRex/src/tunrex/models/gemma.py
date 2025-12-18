"""Gemma model loading utilities."""

from __future__ import annotations

import gc
import os
import shutil
from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from tunix.models.gemma3.model import GemmaModel, ModelConfig


# Default mesh configuration for sharding
DEFAULT_MESH = [(1, 1), ("fsdp", "tp")]


def prepare_gemma_checkpoint(
    ckpt_dir: str = "/tmp/intermediate_ckpt",
    clean_dirs: list[str] | None = None,
) -> Tuple[str, Any, Any]:
    """Download Gemma model and save initial checkpoint for training.

    This is a workaround for memory-efficient model loading. It:
    1. Cleans up any existing checkpoint directories
    2. Downloads and loads the Gemma 3 1B IT model
    3. Saves the model state to a checkpoint
    4. Cleans up to free memory

    Args:
        ckpt_dir: Directory to save the intermediate checkpoint.
        clean_dirs: Additional directories to clean before saving.
            Defaults to [ckpt_dir, "/tmp/content/ckpts"].

    Returns:
        Tuple of (ckpt_path, model_checkpoint_path, tokenizer).

    Example:
        >>> from tunrex import prepare_gemma_checkpoint, get_gemma_ref_model
        >>> ckpt_path, model_cp_path, tokenizer = prepare_gemma_checkpoint()
        >>> ref_model, mesh, config = get_gemma_ref_model(ckpt_path, model_cp_path)
    """
    from flax import nnx
    from orbax import checkpoint as ocp
    from tunix.models.gemma3 import model, params

    # Default directories to clean
    if clean_dirs is None:
        clean_dirs = [ckpt_dir, "/tmp/content/ckpts"]

    # Clean up existing directories
    for dir_path in clean_dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path, ignore_errors=True)

    os.makedirs(ckpt_dir, exist_ok=True)

    # Get model checkpoint path and create model
    model_cp_path = params.GEMMA3_1B_IT
    config = model.ModelConfig.gemma3_1b()
    gemma = params.create_model_from_checkpoint(model_cp_path, config)
    tokenizer = params.create_tokenizer()

    # Save state checkpoint
    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(gemma)
    state_path = os.path.join(ckpt_dir, "state")
    checkpointer.save(state_path, state)
    checkpointer.wait_until_finished()

    # Clean up to free memory
    del gemma
    del state
    gc.collect()

    print(f"Gemma checkpoint saved to: {state_path}")
    return state_path, model_cp_path, tokenizer


def get_gemma_ref_model(
    ckpt_path: str,
    model_checkpoint_path: str,
    mesh_config: list | None = None,
) -> Tuple[Any, Any, Any]:
    """Load a Gemma reference model from checkpoint.

    Args:
        ckpt_path: Path to the checkpoint state directory.
        model_checkpoint_path: Path to the original model checkpoint (for config).
        mesh_config: Mesh configuration for sharding. Defaults to DEFAULT_MESH.

    Returns:
        Tuple of (gemma_model, mesh, model_config).

    Example:
        >>> from tunrex.models import get_gemma_ref_model
        >>> ref_model, mesh, config = get_gemma_ref_model(
        ...     ckpt_path="/tmp/intermediate_ckpt/state",
        ...     model_checkpoint_path=MODEL_CP_PATH,
        ... )
    """
    # Lazy imports to avoid import errors when tunix isn't installed
    from flax import nnx
    from orbax import checkpoint as ocp
    import qwix
    from tunix.models.gemma3 import model, params

    if mesh_config is None:
        mesh_config = DEFAULT_MESH

    model_config = model.ModelConfig.gemma3_1b()

    # Create abstract model shape
    abs_gemma = nnx.eval_shape(
        lambda: params.create_model_from_checkpoint(model_checkpoint_path, model_config)
    )

    # Setup sharding
    abs_state = nnx.state(abs_gemma)
    mesh = qwix.create_mesh(mesh_config)
    sharding_rules = model.GemmaModel.default_sharding_config(model_config, mesh)
    shardings = qwix.make_shardings(abs_state, sharding_rules)

    # Restore checkpoint
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(ckpt_path, target=abs_state)

    # Merge graph definition with restored parameters
    graph_def, _ = nnx.split(abs_gemma)
    gemma_model = nnx.merge(graph_def, restored_params)

    return gemma_model, mesh, model_config


def save_model_state(model: Any, ckpt_dir: str) -> str:
    """Save model state to checkpoint directory.

    Args:
        model: The Gemma model to save.
        ckpt_dir: Directory to save the checkpoint.

    Returns:
        Path to the saved state.
    """
    from flax import nnx
    from orbax import checkpoint as ocp

    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    state_path = os.path.join(ckpt_dir, "state")
    checkpointer.save(state_path, state)
    return state_path


def get_lora_model(
    base_model: Any,
    mesh: Any,
    rank: int = 64,
    alpha: float = 64.0,
    include: list | None = None,
    exclude: list | None = None,
) -> Any:
    """Create a LoRA model from a base model.

    Args:
        base_model: The base Gemma model to apply LoRA to.
        mesh: The mesh for sharding.
        rank: LoRA rank (default: 64).
        alpha: LoRA alpha scaling factor (default: 64.0).
        include: Regex patterns for layers to include (default: [".*attn.*"]).
        exclude: Regex patterns for layers to exclude (default: [".*embed.*"]).

    Returns:
        The LoRA-wrapped model.

    Example:
        >>> from tunrex import get_gemma_ref_model, get_lora_model
        >>> ref_model, mesh, config = get_gemma_ref_model(ckpt_path, model_cp_path)
        >>> lora_policy = get_lora_model(ref_model, mesh, rank=64, alpha=64.0)
    """
    from flax import nnx
    import jax
    import qwix

    if include is None:
        include = [r".*attn.*"]
    if exclude is None:
        exclude = [r".*embed.*"]

    lora_provider = qwix.LoRAProvider(
        rank=rank,
        alpha=alpha,
        include=include,
        exclude=exclude,
    )

    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input
    )

    with mesh:
        state = nnx.state(lora_model)
        sharded_state = jax.tree.map(jax.lax.with_sharding_constraint, state, state)
        nnx.update(lora_model, sharded_state)

    return lora_model
