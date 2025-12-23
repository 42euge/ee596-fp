"""Base classes for benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SampleResult:
    """Result for a single sample."""

    sample_id: str
    question: str
    gold_answer: Any
    predicted_answer: Any
    reasoning: str
    is_correct: bool
    format_correct: bool
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a benchmark."""

    benchmark_name: str
    num_samples: int
    metrics: Dict[str, float]
    per_sample_results: List[SampleResult]
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
            "per_sample_results": [
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "gold_answer": r.gold_answer,
                    "predicted_answer": r.predicted_answer,
                    "reasoning": r.reasoning,
                    "is_correct": r.is_correct,
                    "format_correct": r.format_correct,
                    "generation_time": r.generation_time,
                    "metadata": r.metadata,
                }
                for r in self.per_sample_results
            ],
            "config": self.config,
            "metadata": self.metadata,
        }


class BaseBenchmark(ABC):
    """Base class for evaluation benchmarks."""

    def __init__(self, name: str):
        """Initialize benchmark.

        Args:
            name: Benchmark name
        """
        self.name = name

    @abstractmethod
    def load_dataset(self, split: str = "test", num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load dataset samples.

        Args:
            split: Dataset split (train/val/test)
            num_samples: Number of samples to load (None = all)

        Returns:
            List of sample dictionaries with 'question' and 'answer' keys
        """
        pass

    @abstractmethod
    def extract_answer(self, text: str) -> Any:
        """Extract answer from generated text.

        Args:
            text: Generated text

        Returns:
            Extracted answer
        """
        pass

    @abstractmethod
    def check_answer(self, predicted: Any, gold: Any) -> bool:
        """Check if predicted answer matches gold answer.

        Args:
            predicted: Predicted answer
            gold: Gold answer

        Returns:
            True if correct, False otherwise
        """
        pass

    def check_format(self, text: str) -> bool:
        """Check if text has proper format.

        Args:
            text: Generated text

        Returns:
            True if format is correct, False otherwise
        """
        # Default: check for <reasoning> and <answer> tags
        return "<reasoning>" in text and "</reasoning>" in text and "<answer>" in text and "</answer>" in text

    def extract_reasoning(self, text: str) -> str:
        """Extract reasoning from generated text.

        Args:
            text: Generated text

        Returns:
            Extracted reasoning text
        """
        # Default: extract text between <reasoning> tags
        import re
        match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def evaluate(
        self,
        model,
        split: str = "test",
        num_samples: Optional[int] = None,
        **generation_kwargs
    ) -> EvaluationResult:
        """Evaluate model on this benchmark.

        Args:
            model: Model to evaluate (should have generate() method)
            split: Dataset split
            num_samples: Number of samples to evaluate
            **generation_kwargs: Additional generation arguments

        Returns:
            EvaluationResult object
        """
        import time

        # Load dataset
        samples = self.load_dataset(split, num_samples)

        # Evaluate each sample
        per_sample_results = []
        correct = 0
        partial_correct = 0
        format_correct = 0
        total_time = 0

        for i, sample in enumerate(samples):
            question = sample["question"]
            gold_answer = sample["answer"]

            # Generate prediction
            start_time = time.time()
            generated_text = model.generate(question, **generation_kwargs)
            generation_time = time.time() - start_time
            total_time += generation_time

            # Extract components
            reasoning = self.extract_reasoning(generated_text)
            predicted_answer = self.extract_answer(generated_text)
            is_format_correct = self.check_format(generated_text)
            is_correct = self.check_answer(predicted_answer, gold_answer)

            # Check partial correctness (within 10% for numerical answers)
            is_partial_correct = is_correct
            if not is_correct and isinstance(predicted_answer, (int, float)) and isinstance(gold_answer, (int, float)):
                if gold_answer != 0:
                    ratio = predicted_answer / gold_answer
                    is_partial_correct = 0.9 <= ratio <= 1.1

            # Record result
            sample_result = SampleResult(
                sample_id=sample.get("id", f"{self.name}_{i}"),
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer,
                reasoning=reasoning,
                is_correct=is_correct,
                format_correct=is_format_correct,
                generation_time=generation_time,
                metadata={"generated_text": generated_text}
            )
            per_sample_results.append(sample_result)

            # Update counters
            if is_correct:
                correct += 1
            if is_partial_correct:
                partial_correct += 1
            if is_format_correct:
                format_correct += 1

        # Compute metrics
        num_samples_total = len(samples)
        metrics = {
            "accuracy": correct / num_samples_total if num_samples_total > 0 else 0,
            "partial_accuracy": partial_correct / num_samples_total if num_samples_total > 0 else 0,
            "format_accuracy": format_correct / num_samples_total if num_samples_total > 0 else 0,
            "avg_generation_time": total_time / num_samples_total if num_samples_total > 0 else 0,
            "total_time": total_time,
        }

        return EvaluationResult(
            benchmark_name=self.name,
            num_samples=num_samples_total,
            metrics=metrics,
            per_sample_results=per_sample_results,
            config=generation_kwargs,
        )
