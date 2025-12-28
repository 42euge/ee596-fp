"""Prompt templates for rubric generation and response scoring.

Contains templates for:
- Generating evaluation rubrics for different question types
- Scoring responses against rubrics (LLM-as-judge)
- Reference-based scoring
"""

from typing import Optional, Tuple

# =============================================================================
# Question Type Hints for Rubric Generation
# =============================================================================

QUESTION_TYPE_RUBRIC_HINTS = {
    "math": """For mathematical questions, focus on:
- Correctness of the final answer
- Validity of mathematical reasoning and steps
- Proper use of mathematical notation
- Handling of edge cases
- Clarity of explanation""",
    "creative": """For creative writing tasks, focus on:
- Originality and creativity
- Coherence and structure
- Language quality and style
- Engagement and interest
- Adherence to any constraints""",
    "science": """For scientific questions, focus on:
- Accuracy of scientific facts
- Use of proper terminology
- Logical reasoning and methodology
- Evidence-based conclusions
- Clarity of explanation""",
    "summarization": """For summarization tasks, focus on:
- Coverage of key points
- Conciseness (no unnecessary details)
- Accuracy (no hallucinations)
- Coherence and readability
- Appropriate length""",
    "coding": """For coding/programming tasks, focus on:
- Correctness of the solution
- Code quality and readability
- Efficiency and performance
- Handling of edge cases
- Proper documentation/comments""",
    "reasoning": """For logical reasoning tasks, focus on:
- Validity of logical steps
- Completeness of the argument
- Identification of assumptions
- Consideration of alternatives
- Clear conclusion""",
    "default": """Focus on:
- Accuracy and correctness
- Completeness of response
- Clarity and organization
- Relevance to the question
- Quality of reasoning""",
}

# =============================================================================
# Rubric Generation Templates
# =============================================================================

RUBRIC_GENERATION_TEMPLATE = """You are an expert evaluator creating a scoring rubric. Generate a detailed evaluation rubric for the following question/task.

Question/Task:
{question}

{context_block}

Question Type: {question_type}

{type_hints}

Create a rubric with {num_criteria} evaluation criteria. For each criterion, provide:
1. Name: A short descriptive name
2. Description: What this criterion measures
3. Weight: Relative importance (weights should sum to 1.0)
4. Scoring Levels: 3 levels from {min_score} to {max_score}

Output your rubric in the following JSON format:
```json
{{
  "criteria": [
    {{
      "name": "Criterion Name",
      "description": "What this criterion evaluates",
      "weight": 0.XX,
      "levels": [
        {{"score": {min_score}, "description": "Poor performance description"}},
        {{"score": {mid_score}, "description": "Average performance description"}},
        {{"score": {max_score}, "description": "Excellent performance description"}}
      ]
    }}
  ]
}}
```

Generate the rubric now:"""


def get_rubric_generation_prompt(
    question: str,
    question_type: str = "default",
    context: Optional[str] = None,
    num_criteria: int = 5,
    score_range: Tuple[int, int] = (0, 10),
    include_examples: bool = True,
) -> str:
    """Build the rubric generation prompt.

    Args:
        question: The question/task to create a rubric for
        question_type: Type of question for specialized hints
        context: Additional context for the question
        num_criteria: Number of criteria to generate
        score_range: (min_score, max_score) tuple
        include_examples: Whether to include example descriptions

    Returns:
        Formatted prompt string
    """
    context_block = f"Additional Context:\n{context}" if context else ""

    type_hints = QUESTION_TYPE_RUBRIC_HINTS.get(
        question_type, QUESTION_TYPE_RUBRIC_HINTS["default"]
    )

    min_score, max_score = score_range
    mid_score = (min_score + max_score) // 2

    return RUBRIC_GENERATION_TEMPLATE.format(
        question=question,
        context_block=context_block,
        question_type=question_type,
        type_hints=type_hints,
        num_criteria=num_criteria,
        min_score=min_score,
        mid_score=mid_score,
        max_score=max_score,
    )


# =============================================================================
# Response Scoring Templates
# =============================================================================

SCORING_TEMPLATE_JSON = """You are an expert evaluator. Score the following response according to the provided rubric.

Question:
{question}

Response to Evaluate:
{response}

Evaluation Rubric:
{rubric}

Carefully evaluate the response against each criterion in the rubric.

Output your evaluation in the following JSON format:
```json
{{
  "total_score": <number between 0 and {max_score}>,
  "criterion_scores": {{
    "<criterion_name>": <score>,
    ...
  }},
  "reasoning": "<brief explanation of the score>"
}}
```

Provide your evaluation:"""


SCORING_TEMPLATE_XML = """You are an expert evaluator. Score the following response according to the provided rubric.

Question:
{question}

Response to Evaluate:
{response}

Evaluation Rubric:
{rubric}

Evaluate the response against each criterion.

Output your evaluation in the following format:
<evaluation>
  <total_score>score between 0 and {max_score}</total_score>
  <criterion_scores>
    <criterion name="criterion_name">score</criterion>
  </criterion_scores>
  <reasoning>Your explanation here</reasoning>
</evaluation>

Provide your evaluation:"""


SCORING_TEMPLATE_PLAIN = """You are an expert evaluator. Score the following response according to the provided rubric.

Question:
{question}

Response to Evaluate:
{response}

Evaluation Rubric:
{rubric}

Evaluate the response and provide:
1. A total score from 0 to {max_score}
2. Brief reasoning for your score

Format your response as:
Score: [number]
Reasoning: [your explanation]

Your evaluation:"""


def get_scoring_prompt(
    question: str,
    response: str,
    rubric: str,
    output_format: str = "json",
    include_reasoning: bool = True,
    max_score: int = 10,
) -> str:
    """Build the response scoring prompt.

    Args:
        question: The original question
        response: The response to score
        rubric: The rubric text to score against
        output_format: Output format ("json", "xml", or "plain")
        include_reasoning: Whether to request reasoning
        max_score: Maximum score value

    Returns:
        Formatted prompt string
    """
    templates = {
        "json": SCORING_TEMPLATE_JSON,
        "xml": SCORING_TEMPLATE_XML,
        "plain": SCORING_TEMPLATE_PLAIN,
    }

    template = templates.get(output_format, SCORING_TEMPLATE_JSON)

    return template.format(
        question=question,
        response=response,
        rubric=rubric,
        max_score=max_score,
    )


# =============================================================================
# Reference-Based Scoring Templates
# =============================================================================

REFERENCE_SCORING_TEMPLATE = """You are an expert evaluator. Score the following response by comparing it to a reference answer.

Question:
{question}

Response to Evaluate:
{response}

Reference Answer:
{reference}

Evaluate how well the response matches the quality and correctness of the reference answer.

Consider:
1. Accuracy: Does the response give the same correct answer?
2. Completeness: Does it cover all important points from the reference?
3. Clarity: Is the explanation as clear as the reference?
4. Reasoning: Is the reasoning as thorough?

Output your evaluation in JSON format:
```json
{{
  "total_score": <number between 0 and {max_score}>,
  "accuracy_match": <true/false>,
  "reasoning": "<brief explanation comparing response to reference>"
}}
```

Provide your evaluation:"""


def get_reference_scoring_prompt(
    question: str,
    response: str,
    reference: str,
    output_format: str = "json",
    max_score: int = 10,
) -> str:
    """Build the reference-based scoring prompt.

    Args:
        question: The original question
        response: The response to score
        reference: The reference answer to compare against
        output_format: Output format (currently only json supported)
        max_score: Maximum score value

    Returns:
        Formatted prompt string
    """
    return REFERENCE_SCORING_TEMPLATE.format(
        question=question,
        response=response,
        reference=reference,
        max_score=max_score,
    )
