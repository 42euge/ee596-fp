#!/usr/bin/env python3
"""Advanced example: Comparing rewards with preference models and custom evaluators.

This example demonstrates:
1. Creating custom reward evaluators
2. Using preference models (simulated LLM-as-judge)
3. Comparing reward effects on training data
4. Analyzing which rewards best align with human preferences
"""

import sys
from pathlib import Path
from typing import List, Any
import random

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "TunRex" / "src"))

from tunrex.datasets.config import (
    reasoning_start,
    reasoning_end,
    solution_start,
    solution_end,
)
from tunrex.reward_comparison import (
    BaseReward,
    RewardResult,
    ProgrammaticReward,
    PreferenceModelReward,
    RubricReward,
    RubricCriterion,
    RewardComparison,
    RewardAnalyzer,
)


# ============================================================================
# Custom Reward Evaluators
# ============================================================================

class ReasoningQualityReward(BaseReward):
    """Custom reward that evaluates reasoning quality.

    This reward checks for:
    - Presence of step-by-step reasoning
    - Use of mathematical operators
    - Logical flow indicators
    """

    def __init__(self, name: str = "ReasoningQuality"):
        super().__init__(name=name)
        self.step_keywords = ["first", "then", "next", "finally", "step"]
        self.math_operators = ["+", "-", "*", "/", "="]
        self.logic_keywords = ["because", "therefore", "thus", "so"]

    def evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> RewardResult:
        """Evaluate reasoning quality."""
        import time
        start_time = time.time()

        scores = []
        details = []

        for completion in completions:
            score = 0.0
            feedback = {}

            # Extract reasoning section
            if reasoning_start in completion and reasoning_end in completion:
                start_idx = completion.find(reasoning_start) + len(reasoning_start)
                end_idx = completion.find(reasoning_end)
                reasoning = completion[start_idx:end_idx].lower()

                # Check for step indicators
                step_count = sum(1 for kw in self.step_keywords if kw in reasoning)
                feedback["steps"] = step_count
                score += min(1.0, step_count * 0.3)

                # Check for math operators
                operator_count = sum(1 for op in self.math_operators if op in reasoning)
                feedback["operators"] = operator_count
                score += min(1.0, operator_count * 0.2)

                # Check for logical connectives
                logic_count = sum(1 for kw in self.logic_keywords if kw in reasoning)
                feedback["logic"] = logic_count
                score += min(1.0, logic_count * 0.5)

                # Bonus for length (indicates detail)
                if len(reasoning) > 50:
                    score += 0.5
                    feedback["detailed"] = True
            else:
                feedback["error"] = "No reasoning section found"

            scores.append(score)
            details.append(feedback)

        elapsed_ms = (time.time() - start_time) * 1000
        metadata = self._create_metadata(elapsed_ms)

        return RewardResult(scores=scores, metadata=metadata, details=details)

    def get_evaluator_type(self) -> str:
        return "programmatic"


def simulated_llm_judge(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Simulated LLM-as-judge that scores completions.

    In a real scenario, this would call an API (e.g., GPT-4) to score
    completions based on helpfulness, correctness, and clarity.

    This simulation uses heuristics to approximate such scoring.
    """
    scores = []

    for prompt, completion in zip(prompts, completions):
        score = 0.0

        # Check format (30% weight)
        if all(tag in completion for tag in [reasoning_start, reasoning_end,
                                              solution_start, solution_end]):
            score += 3.0

        # Check length and detail (30% weight)
        if len(completion) > 100:
            score += 3.0
        elif len(completion) > 50:
            score += 1.5

        # Check for reasoning quality (40% weight)
        if reasoning_start in completion:
            reasoning_section = completion[
                completion.find(reasoning_start):
                completion.find(reasoning_end) if reasoning_end in completion else len(completion)
            ]

            # Reward detailed explanations
            if any(word in reasoning_section.lower() for word in
                   ["because", "therefore", "since", "thus"]):
                score += 2.0

            # Reward step-by-step
            if any(word in reasoning_section.lower() for word in
                   ["first", "then", "next", "step"]):
                score += 2.0

        # Normalize to [0, 10]
        scores.append(min(10.0, score))

    return scores


def human_preference_simulator(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Simulated human preference scores.

    In practice, this would be actual human ratings. Here we simulate
    preferences that value:
    - Correctness (40%)
    - Clarity (30%)
    - Conciseness (30%)
    """
    answers = kwargs.get("answer", [None] * len(completions))

    scores = []
    for prompt, completion, answer in zip(prompts, completions, answers):
        score = 0.0

        # Correctness (check if answer matches)
        if answer and solution_start in completion and solution_end in completion:
            start = completion.find(solution_start) + len(solution_start)
            end = completion.find(solution_end)
            extracted = completion[start:end].strip()

            if answer.strip() == extracted:
                score += 4.0  # 40% weight
            elif answer.replace(" ", "") in extracted.replace(" ", ""):
                score += 2.0  # Partial credit

        # Clarity (has clear structure)
        if reasoning_start in completion and solution_start in completion:
            score += 3.0  # 30% weight

        # Conciseness (not too long)
        length = len(completion)
        if 50 <= length <= 200:
            score += 3.0  # 30% weight
        elif 30 <= length < 50 or 200 < length <= 300:
            score += 1.5

        scores.append(score)

    return scores


def main():
    print("=" * 80)
    print("Advanced Reward Comparison: Custom Evaluators & Preference Models")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Create realistic sample data
    # ========================================================================
    print("[1/4] Creating sample data...")

    # Create diverse examples with varying quality
    examples = [
        {
            "prompt": "What is 12 + 8?",
            "completion": f"{reasoning_start}First, I'll add 12 and 8. "
                         f"12 + 8 = 20{reasoning_end} {solution_start}20{solution_end}",
            "answer": "20"
        },
        {
            "prompt": "Calculate 7 * 6",
            "completion": f"{reasoning_start}7 times 6 equals 42{reasoning_end} "
                         f"{solution_start}42{solution_end}",
            "answer": "42"
        },
        {
            "prompt": "What is 100 - 37?",
            "completion": "The answer is 63",  # Poor format
            "answer": "63"
        },
        {
            "prompt": "Solve: 18 / 3",
            "completion": f"{reasoning_start}To divide 18 by 3, I need to find how "
                         f"many times 3 goes into 18. Since 3 * 6 = 18, therefore "
                         f"18 / 3 = 6{reasoning_end} {solution_start}6{solution_end}",
            "answer": "6"
        },
        {
            "prompt": "What is 25 + 13?",
            "completion": f"{reasoning_start}Adding 25 and 13 step by step: "
                         f"First take 25, then add 10 to get 35, then add 3 more "
                         f"to get 38. Thus 25 + 13 = 38{reasoning_end} "
                         f"{solution_start}38{solution_end}",
            "answer": "38"
        },
        {
            "prompt": "Calculate 9 * 5",
            "completion": f"{reasoning_start}9 * 5{reasoning_end} "
                         f"{solution_start}45{solution_end}",  # Minimal reasoning
            "answer": "45"
        },
    ]

    prompts = [ex["prompt"] for ex in examples]
    completions = [ex["completion"] for ex in examples]
    answers = [ex["answer"] for ex in examples]

    print(f"  Created {len(examples)} examples with varying quality")
    print()

    # ========================================================================
    # Step 2: Define comprehensive set of evaluators
    # ========================================================================
    print("[2/4] Defining diverse reward evaluators...")

    evaluators = []

    # 1. Custom reasoning quality evaluator
    evaluators.append(ReasoningQualityReward(name="ReasoningQuality"))

    # 2. Programmatic format check
    def simple_format_check(prompts, completions, **kwargs):
        return [
            3.0 if all(tag in c for tag in [reasoning_start, reasoning_end,
                                             solution_start, solution_end])
            else 0.0
            for c in completions
        ]

    evaluators.append(ProgrammaticReward(
        simple_format_check,
        name="SimpleFormat",
        description="Binary format check"
    ))

    # 3. Rubric-based comprehensive evaluator
    rubric_criteria = [
        RubricCriterion(
            name="Structure",
            description="Has proper reasoning and answer structure",
            max_score=2.0,
            evaluator=lambda p, c, **kw: (
                2.0 if reasoning_start in c and solution_start in c else
                1.0 if solution_start in c else 0.0
            )
        ),
        RubricCriterion(
            name="Detail",
            description="Provides detailed reasoning",
            max_score=2.0,
            evaluator=lambda p, c, **kw: (
                2.0 if len(c) > 150 else
                1.0 if len(c) > 80 else 0.0
            )
        ),
        RubricCriterion(
            name="Clarity",
            description="Uses clear language and connectives",
            max_score=1.0,
            evaluator=lambda p, c, **kw: (
                1.0 if any(word in c.lower() for word in
                          ["first", "then", "therefore", "because"]) else 0.0
            )
        ),
    ]

    evaluators.append(RubricReward(
        criteria=rubric_criteria,
        name="ComprehensiveRubric",
        aggregation="sum"
    ))

    # 4. Simulated LLM judge
    evaluators.append(PreferenceModelReward(
        model_fn=simulated_llm_judge,
        name="LLMJudge",
        model_type="llm_judge",
        description="Simulated GPT-4 style scoring"
    ))

    # 5. Simulated human preferences
    evaluators.append(PreferenceModelReward(
        model_fn=human_preference_simulator,
        name="HumanPreference",
        model_type="human_ratings",
        description="Simulated human preference scores"
    ))

    print(f"  Created {len(evaluators)} evaluators:")
    for evaluator in evaluators:
        print(f"    - {evaluator.name} ({evaluator.get_evaluator_type()})")
    print()

    # ========================================================================
    # Step 3: Run comparison and analyze
    # ========================================================================
    print("[3/4] Running comparison and analysis...")
    print()

    comparison = RewardComparison(
        evaluators=evaluators,
        name="Advanced Reward Methodology Comparison"
    )

    result = comparison.evaluate(
        prompts=prompts,
        completions=completions,
        answer=answers,
        question=prompts
    )

    print()
    print(comparison.summarize(result))
    print()

    # Detailed analysis
    analyzer = RewardAnalyzer(result)

    # Generate comprehensive report
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print()

    report = analyzer.generate_report(
        include_correlations=True,
        include_agreement=True,
        agreement_threshold=5.0,
        top_k=[2, 3, 4]
    )
    print(report)

    # ========================================================================
    # Step 4: Identify insights
    # ========================================================================
    print("[4/4] Identifying insights...")
    print()

    # Find which rewards correlate most with human preferences
    corr_analysis = analyzer.compute_correlations()
    human_pref_idx = result.evaluators.index("HumanPreference")

    print("Correlation with Human Preferences:")
    print("-" * 50)
    for i, name in enumerate(result.evaluators):
        if name != "HumanPreference":
            pearson = corr_analysis.pearson_correlation[i, human_pref_idx]
            spearman = corr_analysis.spearman_correlation[i, human_pref_idx]
            print(f"  {name:25} Pearson: {pearson:6.3f}  Spearman: {spearman:6.3f}")
    print()

    # Find disagreements
    print("Largest Disagreements (between LLMJudge and HumanPreference):")
    print("-" * 80)
    disagreements = comparison.find_disagreements(
        threshold=2.0,
        evaluator_pairs=[("LLMJudge", "HumanPreference")]
    )

    for i, dis in enumerate(disagreements[:3], 1):
        print(f"\nDisagreement #{i}:")
        print(f"  Prompt: {dis['prompt']}")
        print(f"  Completion: {dis['completion'][:80]}...")
        print(f"  LLM Judge: {dis['score1']:.2f}")
        print(f"  Human Pref: {dis['score2']:.2f}")
        print(f"  Difference: {dis['difference']:.2f}")

    print()
    print("=" * 80)
    print("✓ Advanced analysis complete!")
    print()
    print("Key Takeaways:")
    print("  - Compare multiple reward types to understand their effects")
    print("  - Identify which rewards best align with desired outcomes")
    print("  - Use disagreement analysis to find edge cases")
    print("  - Iterate on reward design based on correlation analysis")
    print("=" * 80)


if __name__ == "__main__":
    main()
