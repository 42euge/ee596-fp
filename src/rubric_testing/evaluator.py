"""
Rubric Evaluator - Quick evaluation engine for testing rubric designs on small models

This module provides tools for rapidly evaluating rubric performance without full training runs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import numpy as np
from pathlib import Path
import json

from .designer import BaseRubric, RubricScore


@dataclass
class EvaluationConfig:
    """Configuration for rubric evaluation"""

    # Sample settings
    num_samples: int = 100
    max_samples: Optional[int] = None
    sample_seed: int = 42

    # Generation settings
    max_length: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    num_generations_per_prompt: int = 1

    # Model settings
    model_name: str = "google/gemma-3-1b-it"
    device: str = "auto"
    quantization: Optional[str] = None  # "4bit", "8bit", or None
    use_lora: bool = False
    lora_checkpoint: Optional[str] = None

    # Dataset settings
    dataset_name: str = "openrubrics"
    dataset_split: str = "train"
    dataset_source: str = "huggingface"

    # Evaluation settings
    compute_metrics: bool = True
    save_outputs: bool = True
    output_dir: Optional[str] = None

    # Performance settings
    batch_size: int = 1
    use_cache: bool = True


@dataclass
class EvaluationResult:
    """Results from rubric evaluation"""

    rubric_name: str
    config: EvaluationConfig

    # Scores
    scores: List[RubricScore] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    median_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0

    # Component statistics
    component_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Timing
    total_time: float = 0.0
    time_per_sample: float = 0.0

    # Metadata
    num_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Raw outputs
    prompts: List[str] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    rubrics: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary"""
        return {
            "rubric_name": self.rubric_name,
            "num_samples": self.num_samples,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "median_score": self.median_score,
            "score_range": (self.min_score, self.max_score),
            "total_time": self.total_time,
            "time_per_sample": self.time_per_sample,
            "component_stats": self.component_stats,
        }

    def save(self, path: str):
        """Save results to JSON file"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "rubric_name": self.rubric_name,
            "config": self.config.__dict__,
            "summary": self.summary(),
            "scores": [
                {
                    "total": s.total,
                    "components": s.components,
                    "metadata": s.metadata
                }
                for s in self.scores
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


class RubricEvaluator:
    """
    Quick evaluation engine for testing rubric designs.

    This class allows rapid iteration on rubric designs by:
    1. Loading a small model (optionally with LoRA checkpoint)
    2. Generating completions on a small sample of prompts
    3. Scoring completions with the rubric
    4. Computing statistics and metrics

    Example:
        config = EvaluationConfig(num_samples=50, model_name="google/gemma-3-1b-it")
        evaluator = RubricEvaluator(config)

        rubric = KeywordMatchRubric()
        result = evaluator.evaluate(rubric)

        print(f"Mean score: {result.mean_score:.2f} ± {result.std_score:.2f}")
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of model and dataset"""
        if self._initialized:
            return

        print(f"Initializing RubricEvaluator...")
        print(f"  Model: {self.config.model_name}")
        print(f"  Dataset: {self.config.dataset_name}")
        print(f"  Samples: {self.config.num_samples}")

        # Load dataset
        self._load_dataset()

        # Load model (only if we need to generate)
        if self.config.num_generations_per_prompt > 0:
            self._load_model()

        self._initialized = True

    def _load_dataset(self):
        """Load dataset for evaluation"""
        try:
            from ..utils import load_openrubrics_dataset, load_gsm8k_dataset
        except ImportError:
            # Fallback to TunRex if available
            try:
                from TunRex.src.tunrex.datasets import TunRexDataset, TunRexConfig
                if self.config.dataset_name == "openrubrics":
                    cfg = TunRexConfig.openrubrics(max_examples=self.config.max_samples)
                else:
                    cfg = TunRexConfig.gsm8k(max_examples=self.config.max_samples)
                self.dataset = TunRexDataset(cfg)
                return
            except ImportError:
                pass

        # Manual loading as fallback
        if self.config.dataset_name == "openrubrics":
            try:
                from datasets import load_dataset
                ds = load_dataset("allenai/open_rubrics", split=self.config.dataset_split)
                if self.config.max_samples:
                    ds = ds.select(range(min(len(ds), self.config.max_samples)))
                self.dataset = ds
            except Exception as e:
                print(f"Warning: Could not load dataset: {e}")
                self.dataset = None
        else:
            print(f"Warning: Dataset {self.config.dataset_name} not supported in fallback mode")
            self.dataset = None

    def _load_model(self):
        """Load model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"Loading model {self.config.model_name}...")

            # Determine device
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            else:
                device = self.config.device

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Load model with quantization if specified
            model_kwargs = {"device_map": device}

            if self.config.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif self.config.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )

            # Load LoRA checkpoint if specified
            if self.config.use_lora and self.config.lora_checkpoint:
                from peft import PeftModel
                print(f"Loading LoRA checkpoint from {self.config.lora_checkpoint}")
                self.model = PeftModel.from_pretrained(self.model, self.config.lora_checkpoint)

            self.model.eval()
            print(f"Model loaded on {device}")

        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
            self.tokenizer = None

    def evaluate(
        self,
        rubric: BaseRubric,
        prompts: Optional[List[str]] = None,
        completions: Optional[List[str]] = None,
        rubrics: Optional[List[str]] = None,
        reference_responses: Optional[List[str]] = None,
        target_scores: Optional[List[float]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a rubric design.

        Args:
            rubric: The rubric to evaluate
            prompts: Optional pre-provided prompts (if None, loads from dataset)
            completions: Optional pre-provided completions (if None, generates)
            rubrics: Optional rubric texts
            reference_responses: Optional reference answers
            target_scores: Optional target quality scores

        Returns:
            EvaluationResult with statistics and metrics
        """
        start_time = time.time()

        # Initialize if needed
        if not self._initialized:
            self.initialize()

        # Prepare data
        if prompts is None or rubrics is None:
            prompts, rubrics, reference_responses, target_scores = self._prepare_data_from_dataset()

        # Generate completions if not provided
        if completions is None:
            completions = self._generate_completions(prompts)

        # Score all completions
        scores = []
        for i in range(len(prompts)):
            score = rubric.score(
                prompt=prompts[i],
                completion=completions[i],
                rubric=rubrics[i] if rubrics else "",
                reference_response=reference_responses[i] if reference_responses else None,
                target_score=target_scores[i] if target_scores else None,
            )
            scores.append(score)

        # Compute statistics
        total_scores = [s.total for s in scores]
        result = EvaluationResult(
            rubric_name=rubric.name,
            config=self.config,
            scores=scores,
            mean_score=float(np.mean(total_scores)),
            std_score=float(np.std(total_scores)),
            median_score=float(np.median(total_scores)),
            min_score=float(np.min(total_scores)),
            max_score=float(np.max(total_scores)),
            num_samples=len(scores),
            prompts=prompts,
            completions=completions,
            rubrics=rubrics if rubrics else [],
            total_time=time.time() - start_time,
        )

        result.time_per_sample = result.total_time / max(result.num_samples, 1)

        # Compute component statistics
        component_stats = {}
        for score in scores:
            for comp_name, comp_value in score.components.items():
                if comp_name not in component_stats:
                    component_stats[comp_name] = []
                component_stats[comp_name].append(comp_value)

        result.component_stats = {
            name: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in component_stats.items()
        }

        # Save outputs if requested
        if self.config.save_outputs and self.config.output_dir:
            output_path = Path(self.config.output_dir) / f"{rubric.name}_evaluation.json"
            result.save(str(output_path))
            print(f"Saved results to {output_path}")

        return result

    def _prepare_data_from_dataset(self) -> Tuple[List[str], List[str], List[str], List[float]]:
        """Extract data from loaded dataset"""
        if self.dataset is None:
            raise ValueError("No dataset loaded")

        prompts = []
        rubrics = []
        reference_responses = []
        target_scores = []

        num_samples = min(self.config.num_samples, len(self.dataset))

        for i in range(num_samples):
            item = self.dataset[i]

            # Handle different dataset formats
            if isinstance(item, dict):
                prompts.append(item.get("question", item.get("prompt", "")))
                rubrics.append(item.get("rubric", ""))
                reference_responses.append(item.get("reference_response", item.get("answer", "")))
                target_scores.append(float(item.get("target_score", 0.0)))
            else:
                # Assume it's a tuple or list
                prompts.append(str(item[0]))
                rubrics.append(str(item[1]) if len(item) > 1 else "")
                reference_responses.append(str(item[2]) if len(item) > 2 else "")
                target_scores.append(float(item[3]) if len(item) > 3 else 0.0)

        return prompts, rubrics, reference_responses, target_scores

    def _generate_completions(self, prompts: List[str]) -> List[str]:
        """Generate completions for prompts"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Set num_generations_per_prompt > 0 and initialize.")

        import torch

        completions = []
        print(f"Generating {len(prompts)} completions...")

        for i, prompt in enumerate(prompts):
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(prompts)}")

            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(self.model, "device"):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_length,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            # Decode
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove prompt from completion
            if completion.startswith(prompt):
                completion = completion[len(prompt):]

            completions.append(completion.strip())

        return completions

    def evaluate_multiple(
        self,
        rubrics: List[BaseRubric],
        **kwargs
    ) -> List[EvaluationResult]:
        """Evaluate multiple rubrics on the same data"""
        # Generate data once
        prompts, rubric_texts, references, targets = self._prepare_data_from_dataset()
        completions = self._generate_completions(prompts)

        # Evaluate each rubric
        results = []
        for rubric in rubrics:
            print(f"\nEvaluating rubric: {rubric.name}")
            result = self.evaluate(
                rubric,
                prompts=prompts,
                completions=completions,
                rubrics=rubric_texts,
                reference_responses=references,
                target_scores=targets,
            )
            results.append(result)

        return results
