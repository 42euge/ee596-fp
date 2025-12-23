"""
Examples of custom rubric designs

This file demonstrates how to create custom rubrics for your specific evaluation needs.
"""

import re
from difflib import SequenceMatcher
from src.rubric_testing import BaseRubric, RubricScore, RubricDesigner, create_rubric, CompositeRubric


# Example 1: Simple custom rubric using a function
def count_words_rubric(prompt, completion, rubric, **kwargs):
    """Simple rubric that scores based on word count"""
    word_count = len(completion.split())

    # Score from 0-10 based on word count
    if word_count < 50:
        score = word_count / 50 * 5.0  # 0-5 points for < 50 words
    elif word_count < 200:
        score = 5.0 + (word_count - 50) / 150 * 3.0  # 5-8 points for 50-200
    else:
        score = 8.0 + min((word_count - 200) / 100 * 2.0, 2.0)  # 8-10 for 200+

    return RubricScore(
        total=min(score, 10.0),
        components={"word_count": word_count}
    )


# Example 2: Class-based custom rubric
class ReasoningQualityRubric(BaseRubric):
    """
    Evaluates the quality of reasoning based on multiple factors:
    - Logical connectives ("therefore", "because", "thus")
    - Step markers ("first", "second", "finally")
    - Justifications and explanations
    """

    def __init__(self, name="reasoning_quality", weight=1.0):
        super().__init__(name, weight)
        self._score_range = (0.0, 10.0)

        # Keywords that indicate reasoning
        self.logical_words = [
            "therefore", "thus", "hence", "consequently",
            "because", "since", "as", "so",
            "if", "then", "implies", "follows"
        ]
        self.step_words = [
            "first", "second", "third", "next", "then", "finally",
            "step 1", "step 2", "step 3"
        ]

    def score(self, prompt, completion, rubric, reference_response=None, target_score=None, **kwargs):
        completion_lower = completion.lower()

        # Count logical connectives
        logical_count = sum(1 for word in self.logical_words if word in completion_lower)
        logical_score = min(logical_count * 0.5, 3.0)

        # Count step markers
        step_count = sum(1 for word in self.step_words if word in completion_lower)
        step_score = min(step_count * 0.5, 2.0)

        # Check for explanation patterns
        explanation_patterns = [
            r"this is because",
            r"the reason is",
            r"we can see that",
            r"this shows that",
            r"which means"
        ]
        explanation_count = sum(
            1 for pattern in explanation_patterns
            if re.search(pattern, completion_lower)
        )
        explanation_score = min(explanation_count * 1.0, 3.0)

        # Check for structured reasoning sections
        structure_score = 0.0
        if "<reasoning>" in completion and "</reasoning>" in completion:
            structure_score = 2.0

        total = logical_score + step_score + explanation_score + structure_score

        return RubricScore(
            total=min(total, 10.0),
            components={
                "logical_connectives": logical_score,
                "step_markers": step_score,
                "explanations": explanation_score,
                "structure": structure_score,
            }
        )


# Example 3: Rubric with configurable parameters
class SimilarityRubric(BaseRubric):
    """
    Scores based on similarity to reference response using different methods
    """

    def __init__(
        self,
        name="similarity",
        weight=1.0,
        method="sequence",  # "sequence", "word_overlap", "ngram"
        n=2  # for n-gram similarity
    ):
        super().__init__(name, weight)
        self.method = method
        self.n = n
        self._score_range = (0.0, 10.0)

    def score(self, prompt, completion, rubric, reference_response=None, target_score=None, **kwargs):
        if reference_response is None:
            return RubricScore(total=0.0, components={"error": "no_reference"})

        if self.method == "sequence":
            similarity = SequenceMatcher(None, completion, reference_response).ratio()
        elif self.method == "word_overlap":
            completion_words = set(completion.lower().split())
            reference_words = set(reference_response.lower().split())
            if not reference_words:
                similarity = 0.0
            else:
                overlap = len(completion_words & reference_words)
                similarity = overlap / len(reference_words)
        elif self.method == "ngram":
            completion_ngrams = self._get_ngrams(completion, self.n)
            reference_ngrams = self._get_ngrams(reference_response, self.n)
            if not reference_ngrams:
                similarity = 0.0
            else:
                overlap = len(completion_ngrams & reference_ngrams)
                similarity = overlap / len(reference_ngrams)
        else:
            similarity = 0.0

        score = similarity * 10.0

        return RubricScore(
            total=score,
            components={
                "similarity": similarity,
                "method": self.method,
            }
        )

    def _get_ngrams(self, text, n):
        """Extract n-grams from text"""
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngrams.add(tuple(words[i:i+n]))
        return ngrams


# Example 4: Domain-specific rubric (Math)
class MathSolutionRubric(BaseRubric):
    """
    Evaluates math solutions based on:
    - Presence of mathematical notation
    - Step-by-step work
    - Final answer formatting
    """

    def __init__(self, name="math_solution", weight=1.0):
        super().__init__(name, weight)
        self._score_range = (0.0, 10.0)

    def score(self, prompt, completion, rubric, reference_response=None, target_score=None, **kwargs):
        score = 0.0
        components = {}

        # Check for mathematical notation
        math_patterns = [
            r'\d+\s*[\+\-\*\/\=]\s*\d+',  # Basic operations
            r'\([^)]*\)',  # Parentheses
            r'\d+\^\d+',  # Exponents
            r'\\frac|\\sqrt|\\sum',  # LaTeX
        ]

        math_notation_count = sum(
            len(re.findall(pattern, completion))
            for pattern in math_patterns
        )
        math_score = min(math_notation_count * 0.5, 3.0)
        components["math_notation"] = math_score
        score += math_score

        # Check for step-by-step work (numbered or bulleted)
        steps = re.findall(r'(?:step\s+\d+|^\s*\d+[\.\))]|\*\s+|\-\s+)', completion, re.MULTILINE | re.IGNORECASE)
        step_score = min(len(steps) * 0.5, 3.0)
        components["steps"] = step_score
        score += step_score

        # Check for answer formatting
        answer_patterns = [
            r'<answer>.*?</answer>',
            r'(?:answer|result|solution)\s*(?:is|=|:)\s*(\d+)',
            r'therefore.*?(\d+)',
        ]

        has_answer = any(re.search(pattern, completion, re.IGNORECASE) for pattern in answer_patterns)
        answer_score = 2.0 if has_answer else 0.0
        components["has_answer"] = answer_score
        score += answer_score

        # Check for explanation of solution
        explanation_keywords = ["because", "since", "therefore", "thus", "so"]
        explanation_count = sum(1 for kw in explanation_keywords if kw in completion.lower())
        explanation_score = min(explanation_count * 0.4, 2.0)
        components["explanation"] = explanation_score
        score += explanation_score

        return RubricScore(total=min(score, 10.0), components=components)


# Example 5: Using the designer pattern
designer = RubricDesigner()


@designer.register("brevity", weight=1.0)
def brevity_rubric(prompt, completion, rubric, **kwargs):
    """Rewards concise but complete answers"""
    length = len(completion)

    # Optimal length: 100-300 characters
    if 100 <= length <= 300:
        score = 10.0
    elif length < 100:
        score = (length / 100) * 8.0  # Penalty for too short
    else:
        score = max(10.0 - (length - 300) / 100, 0.0)  # Penalty for too long

    return RubricScore(total=score, components={"length": length})


@designer.register("technical_depth", weight=1.0)
def technical_depth_rubric(prompt, completion, rubric, **kwargs):
    """Evaluates technical depth using domain terminology"""
    # This would be customized for your domain
    technical_terms = [
        "algorithm", "complexity", "optimization", "implementation",
        "architecture", "framework", "protocol", "interface",
        "gradient", "parameter", "tensor", "activation"
    ]

    completion_lower = completion.lower()
    term_count = sum(1 for term in technical_terms if term in completion_lower)

    score = min(term_count * 1.5, 10.0)

    return RubricScore(
        total=score,
        components={"technical_terms": term_count}
    )


# Example 6: Composite rubric combining multiple factors
def create_comprehensive_rubric():
    """Create a comprehensive rubric combining multiple aspects"""
    return CompositeRubric(
        name="comprehensive",
        rubrics=[
            ReasoningQualityRubric(weight=0.3),
            MathSolutionRubric(weight=0.25),
            SimilarityRubric(method="word_overlap", weight=0.2),
            create_rubric("brevity_check", score_func=brevity_rubric, weight=0.15),
            create_rubric("tech_depth", score_func=technical_depth_rubric, weight=0.1),
        ],
        normalize=True
    )


# Example usage demonstration
if __name__ == "__main__":
    # Example completion to test
    test_completion = """
    <reasoning>
    To solve this problem, we need to first understand the underlying principles.

    Step 1: Identify the key variables
    The equation x^2 + 5x + 6 = 0 is a quadratic equation.

    Step 2: Apply the quadratic formula
    Because this is a quadratic, we can use the formula: x = (-b ± sqrt(b^2 - 4ac)) / 2a

    Step 3: Calculate
    Therefore, x = (-5 ± sqrt(25 - 24)) / 2 = (-5 ± 1) / 2

    This gives us x = -2 or x = -3
    </reasoning>

    <answer>x = -2 or x = -3</answer>
    """

    print("Testing custom rubrics on sample completion:\n")

    # Test individual rubrics
    rubrics_to_test = [
        ("Reasoning Quality", ReasoningQualityRubric()),
        ("Math Solution", MathSolutionRubric()),
        ("Word Count", create_rubric("word_count", score_func=count_words_rubric)),
    ]

    for name, rubric in rubrics_to_test:
        result = rubric.score(
            prompt="Solve x^2 + 5x + 6 = 0",
            completion=test_completion,
            rubric="Show step-by-step work and explain your reasoning"
        )
        print(f"{name}:")
        print(f"  Total Score: {result.total:.2f}")
        print(f"  Components: {result.components}")
        print()

    # Test comprehensive rubric
    comprehensive = create_comprehensive_rubric()
    result = comprehensive.score(
        prompt="Solve x^2 + 5x + 6 = 0",
        completion=test_completion,
        rubric="Show step-by-step work"
    )
    print("Comprehensive Rubric:")
    print(f"  Total Score: {result.total:.2f}")
    print(f"  Components: {result.components}")
