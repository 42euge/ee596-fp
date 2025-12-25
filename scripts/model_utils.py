"""Model download and loading utilities for GRPO training."""

import json
import os


def download_model(model_id: str):
    """Download model from HuggingFace and return path and EOS tokens.

    Args:
        model_id: HuggingFace model ID (e.g., "google/gemma-3-1b-it")

    Returns:
        Tuple of (local_model_path, eos_tokens)
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading model: {model_id}...")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("  Using HF_TOKEN for authentication")
    else:
        print("  WARNING: HF_TOKEN not set, gated models may fail")

    local_model_path = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.pth"],
        token=hf_token
    )
    print(f"  Downloaded to: {local_model_path}")

    # Get EOS tokens from generation config
    eos_tokens = []
    gen_config_path = os.path.join(local_model_path, "generation_config.json")
    if os.path.exists(gen_config_path):
        with open(gen_config_path, "r") as f:
            gen_config = json.load(f)
        eos_tokens = gen_config.get("eos_token_id", [])
        print(f"  EOS tokens: {eos_tokens}")

    return local_model_path, eos_tokens


def get_model_config(model_id: str):
    """Get tunix model config for a given model ID."""
    from tunix.models.gemma3 import model as gemma_lib

    if "gemma-3-270m" in model_id:
        return gemma_lib.ModelConfig.gemma3_270m()
    elif "gemma-3-1b" in model_id:
        return gemma_lib.ModelConfig.gemma3_1b_it()
    else:
        print(f"WARNING: Unknown model {model_id}, defaulting to gemma3_1b_it")
        return gemma_lib.ModelConfig.gemma3_1b_it()


def load_models(model_path: str, mesh, args, eos_tokens: list):
    """Load reference model, policy model (with optional LoRA), and tokenizer.

    Args:
        model_path: Local path to downloaded model
        mesh: JAX mesh for sharding
        args: Training arguments with use_lora, lora_rank, lora_alpha
        eos_tokens: List of EOS token IDs to append tokenizer EOS

    Returns:
        Tuple of (policy_model, reference_model, tokenizer, eos_tokens)
    """
    from flax import nnx
    import jax
    import qwix
    from tunix.generate import tokenizer_adapter as tokenizer_lib
    from tunix.models.gemma3 import params_safetensors as params_safetensors_lib

    model_config = get_model_config(args.model_id)

    print("Loading reference model...")
    reference_model = params_safetensors_lib.create_model_from_safe_tensors(
        model_path, model_config, mesh
    )

    if args.use_lora:
        print(f"Creating LoRA model (rank={args.lora_rank}, alpha={args.lora_alpha})...")

        lora_provider = qwix.LoraProvider(
            module_path=(
                ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
                ".*attn_vec_einsum"
            ),
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )

        base_for_lora = params_safetensors_lib.create_model_from_safe_tensors(
            model_path, model_config, mesh
        )
        model_input = base_for_lora.get_model_input()
        policy_model = qwix.apply_lora_to_model(
            base_for_lora, lora_provider, **model_input
        )

        # Shard LoRA model
        state = nnx.state(policy_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(policy_model, sharded_state)
    else:
        print("Using full model (no LoRA)...")
        policy_model = params_safetensors_lib.create_model_from_safe_tensors(
            model_path, model_config, mesh
        )

    # Load tokenizer
    GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)

    if tokenizer.eos_id() not in eos_tokens:
        eos_tokens = list(eos_tokens) + [tokenizer.eos_id()]

    return policy_model, reference_model, tokenizer, eos_tokens
