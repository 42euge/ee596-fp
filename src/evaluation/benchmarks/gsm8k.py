"""GSM8K benchmark implementation."""

import re
from typing import Any, Dict, List, Optional

from .base import BaseBenchmark


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K (Grade School Math 8K) benchmark."""

    def __init__(self):
        """Initialize GSM8K benchmark."""
        super().__init__("gsm8k")

    def load_dataset(self, split: str = "test", num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load GSM8K dataset.

        Args:
            split: Dataset split (train/test)
            num_samples: Number of samples to load

        Returns:
            List of sample dictionaries
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")

        # Load dataset
        dataset = load_dataset("gsm8k", "main", split=split)

        # Convert to our format
        samples = []
        for i, item in enumerate(dataset):
            if num_samples and i >= num_samples:
                break

            # Extract answer from solution
            question = item["question"]
            solution = item["answer"]

            # GSM8K answers are in format "#### 42"
            answer_match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", solution)
            if answer_match:
                answer_str = answer_match.group(1).replace(",", "")
                try:
                    answer = float(answer_str)
                    if answer.is_integer():
                        answer = int(answer)
                except ValueError:
                    answer = answer_str
            else:
                answer = None

            samples.append({
                "id": f"gsm8k_{i}",
                "question": question,
                "answer": answer,
                "solution": solution
            })

        return samples

    def extract_answer(self, text: str) -> Any:
        """Extract answer from generated text.

        Args:
            text: Generated text with <answer> tags

        Returns:
            Extracted numerical answer
        """
        # Try to extract from <answer> tags first
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
        else:
            # Fallback: use entire text
            answer_text = text

        # Extract numerical value
        return self._extract_number(answer_text)

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text.

        Args:
            text: Text containing a number

        Returns:
            Extracted number or None
        """
        # Remove common non-numeric characters
        text = text.strip()

        # Try multiple patterns
        patterns = [
            r"(?:answer is:?\s*)?(-?\d+(?:,\d+)*(?:\.\d+)?)",  # Standard number
            r"(?:=\s*)?(-?\d+(?:,\d+)*(?:\.\d+)?)",  # After equals sign
            r"\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)",  # With dollar sign
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                num_str = match.group(1).replace(",", "")
                try:
                    num = float(num_str)
                    if num.is_integer():
                        return int(num)
                    return num
                except ValueError:
                    continue

        return None

    def check_answer(self, predicted: Any, gold: Any) -> bool:
        """Check if predicted answer matches gold answer.

        Args:
            predicted: Predicted answer
            gold: Gold answer

        Returns:
            True if correct
        """
        if predicted is None or gold is None:
            return False

        # Convert both to float for comparison
        try:
            pred_float = float(predicted)
            gold_float = float(gold)
            return abs(pred_float - gold_float) < 1e-6
        except (ValueError, TypeError):
            return False


# Register benchmark
from ..benchmark_registry import BenchmarkRegistry
BenchmarkRegistry.register("gsm8k", GSM8KBenchmark)
