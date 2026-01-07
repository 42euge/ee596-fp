"""
Step Parser for PRIME RL

Extracts intermediate reasoning steps from model completions for
step-wise reward calculation and process supervision.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .config import StepParsingStrategy, PRIMEConfig


@dataclass
class ParsedStep:
    """A single parsed reasoning step."""

    index: int
    text: str
    original_line_numbers: Optional[Tuple[int, int]] = None  # (start, end)
    is_calculation: bool = False
    is_conclusion: bool = False

    def __repr__(self):
        return f"Step {self.index}: {self.text[:50]}..."


class StepParser:
    """
    Parser for extracting intermediate reasoning steps from completions.

    Supports multiple parsing strategies:
    - NUMBERED: Detect numbered steps (Step 1:, 1., etc.)
    - LINE_BASED: Each non-empty line is a step
    - SENTENCE_BASED: Each sentence is a step
    - SEMANTIC: Semantic chunking based on paragraphs/ideas
    - CUSTOM_DELIMITER: Custom delimiter
    """

    def __init__(self, config: PRIMEConfig):
        self.config = config
        self.strategy = config.step_parsing_strategy

        # Patterns for numbered steps
        self.numbered_patterns = [
            r"^Step\s+(\d+)\s*[:\.]\s*(.+)$",  # "Step 1: ..."
            r"^(\d+)\s*[:\.\)]\s*(.+)$",        # "1. ..." or "1) ..."
            r"^\[Step\s+(\d+)\]\s*(.+)$",       # "[Step 1] ..."
            r"^Stage\s+(\d+)\s*[:\.]\s*(.+)$", # "Stage 1: ..."
        ]

        # Calculation indicators
        self.calculation_pattern = re.compile(
            r"=\s*[\d\.\+\-\*/\(\)]+|→|⇒|\b(equals|becomes)\b",
            re.IGNORECASE
        )

        # Conclusion indicators
        self.conclusion_pattern = re.compile(
            r"\b(therefore|thus|hence|so|consequently|in conclusion|finally)\b",
            re.IGNORECASE
        )

    def parse(self, completion: str) -> List[ParsedStep]:
        """
        Parse completion into intermediate reasoning steps.

        Args:
            completion: Model completion text

        Returns:
            List of ParsedStep objects
        """
        # Extract reasoning section if using structured output
        reasoning = self._extract_reasoning_section(completion)

        # Apply parsing strategy
        if self.strategy == StepParsingStrategy.NUMBERED:
            steps = self._parse_numbered(reasoning)
        elif self.strategy == StepParsingStrategy.LINE_BASED:
            steps = self._parse_line_based(reasoning)
        elif self.strategy == StepParsingStrategy.SENTENCE_BASED:
            steps = self._parse_sentence_based(reasoning)
        elif self.strategy == StepParsingStrategy.SEMANTIC:
            steps = self._parse_semantic(reasoning)
        elif self.strategy == StepParsingStrategy.CUSTOM_DELIMITER:
            steps = self._parse_custom_delimiter(reasoning)
        else:
            raise ValueError(f"Unknown parsing strategy: {self.strategy}")

        # Filter steps by minimum length
        steps = [s for s in steps if len(s.text.strip()) >= self.config.min_step_length]

        # Limit to max steps
        if len(steps) > self.config.max_steps:
            steps = steps[:self.config.max_steps]

        # Annotate steps with metadata
        for step in steps:
            step.is_calculation = bool(self.calculation_pattern.search(step.text))
            step.is_conclusion = bool(self.conclusion_pattern.search(step.text))

        return steps

    def _extract_reasoning_section(self, completion: str) -> str:
        """Extract reasoning section from structured output."""
        # Try to extract content between <reasoning> tags
        match = re.search(r"<reasoning>(.*?)</reasoning>", completion, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Try to extract content between thinking/reasoning markers
        match = re.search(r"(?:Reasoning|Thinking|Solution):\s*(.*?)(?:<answer>|Answer:|$)",
                         completion, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Fall back to full completion
        return completion.strip()

    def _parse_numbered(self, text: str) -> List[ParsedStep]:
        """Parse numbered steps (Step 1:, 1., etc.)."""
        steps = []
        lines = text.split('\n')

        current_step = None
        current_step_text = []
        step_number = 0

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if line starts a new numbered step
            is_new_step = False
            for pattern in self.numbered_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous step
                    if current_step_text:
                        steps.append(ParsedStep(
                            index=step_number,
                            text='\n'.join(current_step_text).strip()
                        ))

                    # Start new step
                    step_number = int(match.group(1)) if match.lastindex >= 1 else step_number + 1
                    step_text = match.group(2) if match.lastindex >= 2 else line
                    current_step_text = [step_text]
                    is_new_step = True
                    break

            # If not a new step, append to current step
            if not is_new_step and current_step_text:
                current_step_text.append(line)

        # Add final step
        if current_step_text:
            steps.append(ParsedStep(
                index=step_number,
                text='\n'.join(current_step_text).strip()
            ))

        # If no numbered steps found, fall back to line-based
        if not steps:
            return self._parse_line_based(text)

        return steps

    def _parse_line_based(self, text: str) -> List[ParsedStep]:
        """Parse each non-empty line as a step."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return [
            ParsedStep(index=i, text=line)
            for i, line in enumerate(lines)
        ]

    def _parse_sentence_based(self, text: str) -> List[ParsedStep]:
        """Parse each sentence as a step."""
        # Simple sentence splitting (can be improved with NLTK/spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return [
            ParsedStep(index=i, text=sentence)
            for i, sentence in enumerate(sentences)
        ]

    def _parse_semantic(self, text: str) -> List[ParsedStep]:
        """Parse based on semantic chunks (paragraphs/ideas)."""
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # If only one paragraph, fall back to sentence-based
        if len(paragraphs) <= 1:
            return self._parse_sentence_based(text)

        return [
            ParsedStep(index=i, text=para)
            for i, para in enumerate(paragraphs)
        ]

    def _parse_custom_delimiter(self, text: str) -> List[ParsedStep]:
        """Parse using custom delimiter."""
        delimiter = self.config.custom_delimiter
        chunks = text.split(delimiter)
        chunks = [c.strip() for c in chunks if c.strip()]

        return [
            ParsedStep(index=i, text=chunk)
            for i, chunk in enumerate(chunks)
        ]


def parse_steps(
    completion: str,
    config: Optional[PRIMEConfig] = None
) -> List[ParsedStep]:
    """
    Convenience function to parse completion into steps.

    Args:
        completion: Model completion text
        config: PRIME RL configuration (uses defaults if None)

    Returns:
        List of ParsedStep objects
    """
    if config is None:
        config = PRIMEConfig()

    parser = StepParser(config)
    return parser.parse(completion)


def format_steps_for_display(steps: List[ParsedStep]) -> str:
    """Format parsed steps for human-readable display."""
    lines = []
    for step in steps:
        markers = []
        if step.is_calculation:
            markers.append("📊")
        if step.is_conclusion:
            markers.append("✓")
        marker_str = " ".join(markers) + " " if markers else ""

        lines.append(f"Step {step.index + 1}: {marker_str}{step.text}")

    return "\n".join(lines)
