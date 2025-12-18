"""Gemma model loading utilities."""

import os
from typing import Tuple

from flax import nnx
import jax
from orbax import checkpoint as ocp
import qwix
from tunix.models.gemma3 import model, params
from tunix.models.gemma3.model import GemmaModel, ModelConfig


# Default mesh configuration for sharding
DEFAULT_MESH = [(1, 1), ("fsdp", "tp")]


def get_gemma_ref_model(
    ckpt_path: str,
    model_checkpoint_path: str,
    mesh_config: list = None,
) -> Tuple[GemmaModel, any, ModelConfig]:
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


def save_model_state(model: GemmaModel, ckpt_dir: str) -> str:
    """Save model state to checkpoint directory.

    Args:
        model: The Gemma model to save.
        ckpt_dir: Directory to save the checkpoint.

    Returns:
        Path to the saved state.
    """
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    state_path = os.path.join(ckpt_dir, "state")
    checkpointer.save(state_path, state)
    return state_path


def get_lora_model(
    base_model: GemmaModel,
    mesh,
    rank: int = 64,
    alpha: float = 64.0,
    include: list = None,
    exclude: list = None,
) -> GemmaModel:
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
