"""
Verification-Based Reward Functions

External verification through code execution and symbolic math.
Based on research: RLVR (Reinforcement Learning with Verifiable Rewards) 2025

Usage:
    from enhancements.verification_rewards import verify_symbolic_math

    scores = verify_symbolic_math(prompts, completions, answers)
"""

import re
import sys
import io
import ast
import sympy as sp
from typing import List, Dict, Any, Optional, Callable
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Symbolic Math Verification
# ============================================================================

def verify_symbolic_math(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    **kwargs
) -> List[float]:
    """
    Verify mathematical expressions symbolically using SymPy.

    This reward function checks algebraic equivalence, not just string matching.
    Handles expressions like "2x + 4" vs "4 + 2x" as equivalent.

    Research basis: RLVR 2025 - verifiable rewards for factual grounding

    Args:
        prompts: Input prompts
        completions: Model completions
        answer: Ground truth answers
        **kwargs: Additional arguments (e.g., answer_pattern for extraction)

    Returns:
        List of reward scores (3.0 for exact, 1.5 for approximate, 0.0 for wrong)
    """
    scores = []
    answer_pattern = kwargs.get('answer_pattern', r'<answer>(.*?)</answer>')

    for completion, true_answer in zip(completions, answer):
        # Extract answer from completion
        match = re.search(answer_pattern, completion, re.DOTALL)
        if match is None:
            scores.append(0.0)
            continue

        extracted = match.group(1).strip()

        try:
            # Parse both as symbolic expressions
            student_expr = _safe_parse_expr(extracted)
            correct_expr = _safe_parse_expr(true_answer)

            if student_expr is None or correct_expr is None:
                # Fallback to string matching
                score = 1.0 if extracted == true_answer else 0.0
                scores.append(score)
                continue

            # Check symbolic equality
            if sp.simplify(student_expr - correct_expr) == 0:
                scores.append(3.0)  # Perfect symbolic match
                logger.debug(f"Symbolic match: {extracted} == {true_answer}")
            else:
                # Try numerical evaluation for approximate match
                try:
                    student_val = float(student_expr)
                    correct_val = float(correct_expr)
                    diff = abs(student_val - correct_val)

                    if diff < 0.01:
                        scores.append(1.5)  # Close enough
                        logger.debug(f"Numerical match: {student_val} ≈ {correct_val}")
                    else:
                        scores.append(0.0)
                        logger.debug(f"Wrong: {student_val} != {correct_val}")
                except:
                    scores.append(0.0)

        except Exception as e:
            logger.warning(f"Error in symbolic verification: {e}")
            # Fallback to string matching
            score = 1.0 if extracted.strip() == true_answer.strip() else 0.0
            scores.append(score)

    return scores


def _safe_parse_expr(expr_str: str) -> Optional[sp.Expr]:
    """
    Safely parse a string into a SymPy expression.

    Handles common edge cases and formats.
    """
    try:
        # Clean up common formatting
        expr_str = expr_str.strip()
        expr_str = expr_str.replace('×', '*').replace('÷', '/')

        # Try direct parsing
        expr = sp.sympify(expr_str)
        return expr
    except:
        # Try some common transformations
        try:
            # Remove units (e.g., "42 kg" -> "42")
            expr_str = re.sub(r'[a-zA-Z]+$', '', expr_str).strip()
            expr = sp.sympify(expr_str)
            return expr
        except:
            return None


# ============================================================================
# Code Execution Verification
# ============================================================================

@contextmanager
def safe_execution_environment(timeout: int = 5):
    """
    Context manager for safe code execution.

    Note: This is a basic sandbox. For production, use proper sandboxing
    (Docker, subprocess with resource limits, etc.)
    """
    # Capture stdout/stderr
    stdout = io.StringIO()
    stderr = io.StringIO()

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield stdout, stderr
    finally:
        pass


def verify_code_execution(
    prompts: List[str],
    completions: List[str],
    test_cases: List[List[Dict[str, Any]]],
    **kwargs
) -> List[float]:
    """
    Verify code correctness by executing against test cases.

    WARNING: This executes generated code. Use proper sandboxing in production!

    Research basis: RLVR 2025 - executable verification for code generation

    Args:
        prompts: Input prompts
        completions: Model completions containing code
        test_cases: List of test cases for each completion
            Format: [{"input": {...}, "expected": ...}, ...]

    Returns:
        List of reward scores (0.0 to 3.0 based on test pass rate)
    """
    scores = []
    code_pattern = kwargs.get('code_pattern', r'```python\n(.*?)```')

    for completion, tests in zip(completions, test_cases):
        # Extract code from completion
        match = re.search(code_pattern, completion, re.DOTALL)
        if match is None:
            scores.append(0.0)
            logger.warning("No code block found in completion")
            continue

        code = match.group(1)

        # Run test cases
        passed = 0
        total = len(tests)

        for test in tests:
            try:
                # Create safe execution environment
                with safe_execution_environment() as (stdout, stderr):
                    # Prepare namespace
                    namespace = {}

                    # Execute the code
                    exec(code, namespace)

                    # Find the main function (assume first defined function)
                    func_name = _extract_function_name(code)
                    if func_name is None:
                        logger.warning("Could not find function in code")
                        break

                    func = namespace.get(func_name)
                    if func is None:
                        logger.warning(f"Function {func_name} not found")
                        break

                    # Call with test inputs
                    result = func(**test['input'])

                    # Check result
                    if result == test['expected']:
                        passed += 1
                    else:
                        logger.debug(f"Test failed: {result} != {test['expected']}")

            except Exception as e:
                logger.debug(f"Test execution error: {e}")
                continue

        # Compute score based on pass rate
        if total > 0:
            pass_rate = passed / total
            score = 3.0 * pass_rate
            scores.append(score)
            logger.info(f"Code verification: {passed}/{total} tests passed (score: {score:.2f})")
        else:
            scores.append(0.0)

    return scores


def _extract_function_name(code: str) -> Optional[str]:
    """Extract the first function name from Python code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        pass
    return None


# ============================================================================
# External API Verification
# ============================================================================

def verify_with_external_api(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    api_function: Callable,
    **kwargs
) -> List[float]:
    """
    Verify answers using an external API (e.g., Wolfram Alpha, fact-checking API).

    Args:
        prompts: Input prompts
        completions: Model completions
        answer: Ground truth answers
        api_function: Function that takes (extracted_answer, true_answer) and returns bool

    Returns:
        List of reward scores
    """
    scores = []
    answer_pattern = kwargs.get('answer_pattern', r'<answer>(.*?)</answer>')

    for completion, true_answer in zip(completions, answer):
        # Extract answer from completion
        match = re.search(answer_pattern, completion, re.DOTALL)
        if match is None:
            scores.append(0.0)
            continue

        extracted = match.group(1).strip()

        try:
            # Call external API for verification
            is_correct = api_function(extracted, true_answer)
            score = 3.0 if is_correct else 0.0
            scores.append(score)
        except Exception as e:
            logger.error(f"External API error: {e}")
            # Fallback to string matching
            score = 1.0 if extracted == true_answer else 0.0
            scores.append(score)

    return scores


# ============================================================================
# Hybrid Verification
# ============================================================================

class HybridVerifier:
    """
    Combines multiple verification methods with fallback.

    Tries verification methods in order of confidence:
    1. Symbolic math (high confidence)
    2. Code execution (high confidence)
    3. External API (medium confidence)
    4. String matching (low confidence)
    """

    def __init__(self):
        self.verification_methods = []

    def add_verifier(self, name: str, verifier: Callable, priority: int = 0):
        """Add a verification method."""
        self.verification_methods.append({
            'name': name,
            'verifier': verifier,
            'priority': priority
        })
        # Sort by priority (higher first)
        self.verification_methods.sort(key=lambda x: x['priority'], reverse=True)

    def verify(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Verify using the highest priority available method.
        """
        for method in self.verification_methods:
            try:
                scores = method['verifier'](prompts, completions, **kwargs)
                logger.info(f"Verification succeeded using: {method['name']}")
                return scores
            except Exception as e:
                logger.warning(f"Verification method '{method['name']}' failed: {e}")
                continue

        # All methods failed, return zeros
        logger.error("All verification methods failed")
        return [0.0] * len(completions)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test symbolic math verification
    prompts = ["What is 2+2?"]
    completions = ["<reasoning>2+2=4</reasoning><answer>4</answer>"]
    answers = ["4"]

    scores = verify_symbolic_math(prompts, completions, answers)
    print(f"Symbolic math scores: {scores}")

    # Test with algebraic expression
    completions2 = ["<reasoning>Simplify</reasoning><answer>2*x + 4</answer>"]
    answers2 = ["4 + 2*x"]  # Equivalent but different form

    scores2 = verify_symbolic_math(prompts, completions2, answers2)
    print(f"Algebraic equivalence scores: {scores2}")

    # Test code execution
    completions3 = ["""
<reasoning>I'll write a function to add two numbers</reasoning>
<answer>
```python
def add(a, b):
    return a + b
```
</answer>
"""]
    test_cases3 = [[
        {"input": {"a": 2, "b": 3}, "expected": 5},
        {"input": {"a": 0, "b": 0}, "expected": 0},
        {"input": {"a": -1, "b": 1}, "expected": 0}
    ]]

    scores3 = verify_code_execution(prompts, completions3, test_cases3)
    print(f"Code execution scores: {scores3}")
