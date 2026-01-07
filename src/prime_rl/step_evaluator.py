"""
Step Evaluator for PRIME RL

Evaluates individual reasoning steps using multiple methods:
- Rule-based checks (pattern matching, format validation)
- Symbolic solvers (for mathematical correctness)
- LLM judges (for semantic quality)
- Verifier models (learned step verification)
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

from .config import StepEvaluationMethod, PRIMEConfig, StepReward
from .step_parser import ParsedStep


@dataclass
class StepEvaluation:
    """Evaluation result for a single step."""

    step_index: int
    method: StepEvaluationMethod
    score: float  # Typically in [0, 1] or [-1, 1]
    passed: bool
    details: Dict[str, Any]

    def to_step_reward(self, step_text: str) -> StepReward:
        """Convert to StepReward object."""
        return StepReward(
            step_index=self.step_index,
            step_text=step_text,
            reward=self.score,
            evaluation_method=self.method.value,
            details=self.details
        )


class RuleBasedEvaluator:
    """
    Rule-based step evaluator using pattern matching and heuristics.

    Checks for:
    - Presence of calculations
    - Logical connectives
    - Mathematical operations
    - Variable usage
    - Conclusion indicators
    """

    def __init__(self, config: PRIMEConfig):
        self.config = config
        self.patterns = config.rule_based_patterns

        # Compile patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }

        # Additional quality patterns
        self.quality_patterns = {
            "has_numbers": re.compile(r"\d+\.?\d*"),
            "has_math_ops": re.compile(r"[\+\-\*/=]"),
            "has_variables": re.compile(r"\b[a-zA-Z]\b\s*="),
            "has_units": re.compile(r"\b(cm|m|km|kg|g|s|min|hr|°C|°F)\b"),
            "complete_sentence": re.compile(r"[.!?]$"),
        }

    def evaluate(self, step: ParsedStep, context: Optional[Dict] = None) -> StepEvaluation:
        """
        Evaluate a step using rule-based checks.

        Args:
            step: Parsed reasoning step
            context: Optional context (question, previous steps, etc.)

        Returns:
            StepEvaluation with score and details
        """
        details = {}
        score_components = []

        # Check each pattern
        for name, pattern in self.compiled_patterns.items():
            matches = pattern.search(step.text)
            details[f"has_{name}"] = bool(matches)
            if matches:
                score_components.append(0.2)  # +0.2 for each matched pattern

        # Check quality patterns
        for name, pattern in self.quality_patterns.items():
            matches = pattern.search(step.text)
            details[name] = bool(matches)
            if matches:
                score_components.append(0.1)  # +0.1 for quality indicators

        # Length check (too short steps are likely incomplete)
        length_score = min(len(step.text) / 100.0, 1.0)  # Normalize to [0, 1]
        details["length_score"] = length_score
        score_components.append(length_score * 0.2)

        # Calculate final score
        base_score = sum(score_components)
        final_score = min(base_score, 1.0)  # Cap at 1.0

        # Determine if step passes
        passed = final_score >= 0.3  # Threshold for passing

        details["score_components"] = score_components
        details["base_score"] = base_score

        return StepEvaluation(
            step_index=step.index,
            method=StepEvaluationMethod.RULE_BASED,
            score=final_score,
            passed=passed,
            details=details
        )


class SymbolicEvaluator:
    """
    Symbolic evaluator for mathematical reasoning steps.

    Uses symbolic computation to verify mathematical correctness.
    """

    def __init__(self, config: PRIMEConfig):
        self.config = config
        self.enabled = config.enable_symbolic_solver

        # Try to import sympy for symbolic computation
        try:
            import sympy
            self.sympy = sympy
            self.sympy_available = True
        except ImportError:
            self.sympy_available = False
            if self.enabled:
                warnings.warn(
                    "sympy not available. Symbolic evaluation will be disabled. "
                    "Install with: pip install sympy"
                )

    def evaluate(self, step: ParsedStep, context: Optional[Dict] = None) -> StepEvaluation:
        """
        Evaluate mathematical correctness of a step.

        Args:
            step: Parsed reasoning step
            context: Optional context with problem variables

        Returns:
            StepEvaluation with mathematical correctness score
        """
        if not self.enabled or not self.sympy_available:
            return StepEvaluation(
                step_index=step.index,
                method=StepEvaluationMethod.SYMBOLIC,
                score=0.5,  # Neutral score when disabled
                passed=True,
                details={"enabled": False}
            )

        details = {}

        # Extract equations from step
        equations = self._extract_equations(step.text)
        details["equations_found"] = len(equations)

        if not equations:
            # No equations to verify
            return StepEvaluation(
                step_index=step.index,
                method=StepEvaluationMethod.SYMBOLIC,
                score=0.5,
                passed=True,
                details=details
            )

        # Verify each equation
        correct_count = 0
        total_count = 0

        for eq in equations:
            try:
                is_correct = self._verify_equation(eq, context)
                details[f"eq_{total_count}"] = {"equation": eq, "correct": is_correct}
                if is_correct:
                    correct_count += 1
                total_count += 1
            except Exception as e:
                details[f"eq_{total_count}_error"] = str(e)
                total_count += 1

        # Calculate score
        if total_count > 0:
            score = correct_count / total_count
            passed = score >= 0.5
        else:
            score = 0.5
            passed = True

        details["correct_equations"] = correct_count
        details["total_equations"] = total_count

        return StepEvaluation(
            step_index=step.index,
            method=StepEvaluationMethod.SYMBOLIC,
            score=score,
            passed=passed,
            details=details
        )

    def _extract_equations(self, text: str) -> List[str]:
        """Extract equations from text."""
        # Look for patterns like "x = y" or "expression = expression"
        equation_pattern = re.compile(r"([^=]+)=([^=\n]+)")
        matches = equation_pattern.findall(text)

        equations = []
        for left, right in matches:
            # Clean up whitespace
            left = left.strip()
            right = right.strip()

            # Skip if either side is empty or too short
            if len(left) < 1 or len(right) < 1:
                continue

            equations.append(f"{left}={right}")

        return equations

    def _verify_equation(self, equation: str, context: Optional[Dict] = None) -> bool:
        """
        Verify if an equation is mathematically correct.

        Args:
            equation: Equation string (e.g., "2 + 2 = 4")
            context: Optional context with variable values

        Returns:
            True if equation is correct
        """
        try:
            # Split equation
            parts = equation.split('=')
            if len(parts) != 2:
                return False

            left, right = parts

            # Try to parse and evaluate with sympy
            left_expr = self.sympy.sympify(left)
            right_expr = self.sympy.sympify(right)

            # Simplify both sides
            left_simplified = self.sympy.simplify(left_expr)
            right_simplified = self.sympy.simplify(right_expr)

            # Check if equal
            return self.sympy.simplify(left_simplified - right_simplified) == 0

        except Exception:
            # If sympy can't parse or evaluate, assume neutral
            return False


class LLMJudgeStepEvaluator:
    """
    LLM-based step evaluator.

    Uses an LLM to judge the quality of individual reasoning steps.
    """

    def __init__(self, config: PRIMEConfig):
        self.config = config

        # Import LLM judge components
        try:
            from ..llm_judge.response_scorer import ResponseScorer
            from ..llm_judge.backends.factory import create_backend

            backend = create_backend(
                backend_type="openai",
                model=config.llm_judge_model,
                temperature=config.llm_judge_temperature,
                max_tokens=config.llm_judge_max_tokens
            )

            self.scorer = ResponseScorer(backend=backend)
            self.available = True

        except Exception as e:
            warnings.warn(f"LLM Judge not available for step evaluation: {e}")
            self.available = False

    def evaluate(self, step: ParsedStep, context: Optional[Dict] = None) -> StepEvaluation:
        """
        Evaluate step quality using LLM judge.

        Args:
            step: Parsed reasoning step
            context: Context with question, previous steps, etc.

        Returns:
            StepEvaluation with LLM-judged score
        """
        if not self.available:
            return StepEvaluation(
                step_index=step.index,
                method=StepEvaluationMethod.LLM_JUDGE,
                score=0.5,
                passed=True,
                details={"available": False}
            )

        # Build evaluation prompt
        prompt = self._build_step_evaluation_prompt(step, context)

        try:
            # Score using LLM judge
            # This is a simplified version - in practice, you'd use a proper rubric
            result = self.scorer.backend.generate(prompt)

            # Parse score from result (expecting format like "Score: 8/10")
            score = self._parse_score(result)

            details = {
                "llm_response": result,
                "parsed_score": score
            }

            passed = score >= 0.6

            return StepEvaluation(
                step_index=step.index,
                method=StepEvaluationMethod.LLM_JUDGE,
                score=score,
                passed=passed,
                details=details
            )

        except Exception as e:
            return StepEvaluation(
                step_index=step.index,
                method=StepEvaluationMethod.LLM_JUDGE,
                score=0.5,
                passed=True,
                details={"error": str(e)}
            )

    def _build_step_evaluation_prompt(
        self,
        step: ParsedStep,
        context: Optional[Dict] = None
    ) -> str:
        """Build prompt for LLM step evaluation."""
        context = context or {}

        prompt_parts = [
            "Evaluate the quality of this reasoning step.",
            "",
            f"Question: {context.get('question', 'N/A')}",
            "",
            f"Step {step.index + 1}: {step.text}",
            "",
            "Evaluate this step on:",
            "1. Logical coherence (does it make sense?)",
            "2. Mathematical correctness (if applicable)",
            "3. Relevance to the question",
            "4. Clarity of explanation",
            "",
            "Provide a score from 0-10 and brief justification.",
            "Format: Score: X/10\nJustification: ..."
        ]

        return "\n".join(prompt_parts)

    def _parse_score(self, response: str) -> float:
        """Parse score from LLM response."""
        # Look for "Score: X/10" pattern
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return score / 10.0  # Normalize to [0, 1]

        # Look for "X/10" pattern
        match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", response)
        if match:
            score = float(match.group(1))
            return score / 10.0

        # Default to neutral
        return 0.5


class StepEvaluator:
    """
    Main step evaluator that orchestrates multiple evaluation methods.

    Combines rule-based, symbolic, and LLM-based evaluation.
    """

    def __init__(self, config: PRIMEConfig):
        self.config = config
        self.method = config.step_evaluation_method

        # Initialize evaluators
        self.rule_based = RuleBasedEvaluator(config)
        self.symbolic = SymbolicEvaluator(config)

        if config.llm_judge_enabled:
            self.llm_judge = LLMJudgeStepEvaluator(config)
        else:
            self.llm_judge = None

    def evaluate_step(
        self,
        step: ParsedStep,
        context: Optional[Dict] = None
    ) -> StepReward:
        """
        Evaluate a single step using configured method(s).

        Args:
            step: Parsed reasoning step
            context: Optional context (question, previous steps, etc.)

        Returns:
            StepReward with aggregated evaluation
        """
        evaluations = []

        # Apply configured evaluation methods
        if self.method == StepEvaluationMethod.RULE_BASED:
            eval_result = self.rule_based.evaluate(step, context)
            evaluations.append(eval_result)

        elif self.method == StepEvaluationMethod.SYMBOLIC:
            eval_result = self.symbolic.evaluate(step, context)
            evaluations.append(eval_result)

        elif self.method == StepEvaluationMethod.LLM_JUDGE:
            if self.llm_judge:
                eval_result = self.llm_judge.evaluate(step, context)
                evaluations.append(eval_result)

        elif self.method == StepEvaluationMethod.HYBRID:
            # Use all available methods
            evaluations.append(self.rule_based.evaluate(step, context))
            evaluations.append(self.symbolic.evaluate(step, context))
            if self.llm_judge:
                evaluations.append(self.llm_judge.evaluate(step, context))

        # Aggregate evaluations
        if not evaluations:
            # No evaluations available, return neutral score
            return StepReward(
                step_index=step.index,
                step_text=step.text,
                reward=0.5,
                evaluation_method="none",
                details={}
            )

        # Average scores from all evaluations
        avg_score = sum(e.score for e in evaluations) / len(evaluations)

        # Combine details
        combined_details = {
            "evaluations": [
                {
                    "method": e.method.value,
                    "score": e.score,
                    "passed": e.passed,
                    "details": e.details
                }
                for e in evaluations
            ],
            "num_evaluations": len(evaluations)
        }

        return StepReward(
            step_index=step.index,
            step_text=step.text,
            reward=avg_score,
            evaluation_method=self.method.value,
            details=combined_details
        )

    def evaluate_steps(
        self,
        steps: List[ParsedStep],
        context: Optional[Dict] = None
    ) -> List[StepReward]:
        """
        Evaluate multiple steps.

        Args:
            steps: List of parsed steps
            context: Optional context

        Returns:
            List of StepReward objects
        """
        return [self.evaluate_step(step, context) for step in steps]
