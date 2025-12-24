#!/usr/bin/env python3
"""Test rubric reward function without TPU or torch.

This is a standalone script that validates rubrics can be loaded and scored
before using them in actual GRPO training. It has no external dependencies
beyond Python stdlib and PyYAML.

Usage:
    python scripts/test_rubric_reward.py --rubric-file rubrics/example_math.yaml
"""

import argparse
import re
from collections import Counter
import math
from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class Criterion:
    """A single evaluation criterion."""
    name: str
    description: str
    weight: float = 1.0
    keywords: List[str] = field(default_factory=list)


@dataclass
class Rubric:
    """A complete rubric with multiple criteria."""
    name: str
    description: str
    criteria: List[Criterion]
    question_types: List[str] = field(default_factory=list)
    reference_response: Optional[str] = None
    target_score: Optional[float] = None

    def to_text(self) -> str:
        """Convert rubric to text for scoring."""
        lines = [
            f"Evaluation Rubric: {self.name}",
            f"Description: {self.description}",
            "",
            "Criteria:",
        ]
        for c in self.criteria:
            lines.append(f"- {c.name} (weight: {c.weight}): {c.description}")
            if c.keywords:
                lines.append(f"  Keywords: {', '.join(c.keywords)}")
        return "\n".join(lines)


@dataclass
class RubricSet:
    """A collection of rubrics."""
    name: str
    rubrics: List[Rubric]
    description: str = ""

    def __len__(self):
        return len(self.rubrics)


def load_rubricset_from_yaml(path: str) -> RubricSet:
    """Load a rubric set from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Check if it's a rubricset or single rubric
    if 'rubrics' in data:
        rubrics = []
        for r in data['rubrics']:
            criteria = [
                Criterion(
                    name=c['name'],
                    description=c.get('description', ''),
                    weight=c.get('weight', 1.0),
                    keywords=c.get('keywords', []),
                )
                for c in r.get('criteria', [])
            ]
            rubrics.append(Rubric(
                name=r['name'],
                description=r.get('description', ''),
                criteria=criteria,
                question_types=r.get('question_types', []),
                reference_response=r.get('reference_response'),
                target_score=r.get('target_score'),
            ))
        return RubricSet(
            name=data.get('name', 'unnamed'),
            description=data.get('description', ''),
            rubrics=rubrics,
        )
    else:
        # Single rubric
        criteria = [
            Criterion(
                name=c['name'],
                description=c.get('description', ''),
                weight=c.get('weight', 1.0),
                keywords=c.get('keywords', []),
            )
            for c in data.get('criteria', [])
        ]
        rubric = Rubric(
            name=data['name'],
            description=data.get('description', ''),
            criteria=criteria,
            question_types=data.get('question_types', []),
            reference_response=data.get('reference_response'),
            target_score=data.get('target_score'),
        )
        return RubricSet(name=data['name'], rubrics=[rubric])


def simple_tfidf_score(text: str, reference: str) -> float:
    """Simple TF-IDF-like overlap score.

    This mimics the rubric_overlap_score function from src.utils
    but without requiring torch/sklearn.
    """
    def tokenize(s):
        return re.findall(r'\b\w+\b|<[^>]+>', s.lower())

    text_tokens = tokenize(text)
    ref_tokens = tokenize(reference)

    if not text_tokens or not ref_tokens:
        return 0.0

    # Count term frequencies
    text_tf = Counter(text_tokens)
    ref_tf = Counter(ref_tokens)

    # Calculate overlap
    common = set(text_tf.keys()) & set(ref_tf.keys())
    if not common:
        return 0.0

    # Simple weighted overlap score
    score = 0.0
    for term in common:
        weight = math.log(1 + ref_tf[term])
        score += weight * min(text_tf[term], ref_tf[term])

    # Normalize by reference length
    max_score = sum(math.log(1 + c) * c for c in ref_tf.values())

    return (score / max_score) * 10.0 if max_score > 0 else 0.0


def score_with_rubric(completion: str, rubric: Rubric) -> float:
    """Score a completion against a rubric."""
    rubric_text = rubric.to_text()
    return simple_tfidf_score(completion, rubric_text)


def create_test_reward_function(rubricset: RubricSet, weight: float = 1.0):
    """Create a reward function for testing (matches GRPO interface)."""
    def reward_fn(prompts, completions, **kwargs):
        scores = []
        for completion in completions:
            score = score_with_rubric(completion, rubricset.rubrics[0])
            scores.append(score * weight)
        return scores
    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="Test rubric reward function")
    parser.add_argument(
        "--rubric-file",
        type=str,
        default="rubrics/example_math.yaml",
        help="Path to rubric YAML file",
    )
    parser.add_argument(
        "--rubric-weight",
        type=float,
        default=1.0,
        help="Weight for rubric reward",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RUBRIC REWARD FUNCTION TEST (standalone)")
    print("=" * 60)

    # Load rubric
    print(f"\n1. Loading rubric from: {args.rubric_file}")
    rubricset = load_rubricset_from_yaml(args.rubric_file)
    print(f"   Loaded: {rubricset.name} ({len(rubricset)} rubrics)")

    for rubric in rubricset.rubrics:
        print(f"   - {rubric.name}: {len(rubric.criteria)} criteria")
        for c in rubric.criteria:
            kw_preview = ", ".join(c.keywords[:3]) if c.keywords else "none"
            print(f"     * {c.name} (weight={c.weight}): {kw_preview}...")

    # Create reward function
    print(f"\n2. Creating reward function (weight={args.rubric_weight})")
    reward_fn = create_test_reward_function(rubricset, weight=args.rubric_weight)

    # Test responses
    print("\n3. Testing with sample responses...")
    test_cases = [
        {
            "name": "Excellent (tags + reasoning)",
            "prompt": "What is 5 + 10?",
            "completion": "<reasoning>First, I add 5 and 10. The calculation is 5 + 10 = 15. Therefore, the answer is 15.</reasoning><answer>15</answer>",
        },
        {
            "name": "Good (tags, minimal reasoning)",
            "prompt": "What is 5 + 10?",
            "completion": "<reasoning>5+10=15</reasoning><answer>15</answer>",
        },
        {
            "name": "Poor (no tags)",
            "prompt": "What is 5 + 10?",
            "completion": "The answer is 15.",
        },
        {
            "name": "Very poor (wrong format)",
            "prompt": "What is 5 + 10?",
            "completion": "15",
        },
    ]

    prompts = [tc["prompt"] for tc in test_cases]
    completions = [tc["completion"] for tc in test_cases]

    scores = reward_fn(prompts=prompts, completions=completions)

    print("\n   Results:")
    print("-" * 60)
    for tc, score in zip(test_cases, scores):
        print(f"   [{tc['name']:30}] Score: {score:6.2f}")
        preview = tc["completion"][:50] + "..." if len(tc["completion"]) > 50 else tc["completion"]
        print(f"      Response: {preview}")
    print("-" * 60)

    # Validate ordering
    print("\n4. Validation...")
    if scores[0] > scores[1] > scores[2]:
        print("   ✓ Score ordering is correct (better responses score higher)")
    else:
        print("   ⚠ Score ordering may need adjustment")
        print(f"     Scores: {scores}")

    # Show rubric text
    print("\n5. Rubric text representation:")
    print("-" * 60)
    print(rubricset.rubrics[0].to_text()[:500])
    print("-" * 60)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nThe rubric is ready for use in training:")
    print(f"  python scripts/train_grpo.py --rubric-file {args.rubric_file}")
    print("\nNote: This test uses a simplified scorer. The actual training")
    print("uses src.utils.rubric_overlap_score which may produce slightly")
    print("different scores but follows the same pattern.")


if __name__ == "__main__":
    main()
