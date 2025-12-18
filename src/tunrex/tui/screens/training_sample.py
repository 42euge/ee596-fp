"""
Training sample viewer - for viewing individual evaluation/training samples
"""

from typing import List, Dict
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from rich.syntax import Syntax
from rich.panel import Panel


class TrainingSampleScreen(Screen):
    """
    Screen for viewing individual training/evaluation samples
    Shows question, correct answer, model response, and scoring
    """

    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("q", "quit", "Quit"),
        Binding("right", "next_sample", "Next"),
        Binding("left", "previous_sample", "Previous"),
    ]

    def __init__(self, samples: List[Dict], current_index: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = samples
        self.current_index = current_index

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        with Vertical(id="sample-container"):
            # Sample navigation header
            yield Static("", id="sample-header", classes="info")

            # Main content
            with Horizontal(id="sample-content"):
                # Left: Question
                with Vertical(classes="sample-panel"):
                    yield Static("[bold cyan]Question[/bold cyan]", classes="panel-title")
                    with VerticalScroll(classes="sample-scroll"):
                        yield Static("", id="question-display")

                # Middle: Correct Answer
                with Vertical(classes="sample-panel"):
                    yield Static("[bold green]Correct Answer[/bold green]", classes="panel-title")
                    with VerticalScroll(classes="sample-scroll"):
                        yield Static("", id="correct-answer-display")

                # Right: Model Response
                with Vertical(classes="sample-panel"):
                    yield Static("[bold yellow]Model Response[/bold yellow]", classes="panel-title")
                    with VerticalScroll(classes="sample-scroll"):
                        yield Static("", id="model-response-display")

            # Score/metrics footer
            yield Static("", id="sample-metrics", classes="info")

        yield Footer()

    def on_mount(self) -> None:
        """Display current sample"""
        self._display_sample()

    def _display_sample(self):
        """Display the current sample"""
        if not self.samples or self.current_index >= len(self.samples):
            return

        sample = self.samples[self.current_index]

        # Update header
        header = self.query_one("#sample-header", Static)
        header.update(
            f"[bold]Sample {self.current_index + 1} of {len(self.samples)}[/bold]"
        )

        # Question
        question_display = self.query_one("#question-display", Static)
        question = sample.get("question", sample.get("input", "N/A"))
        question_display.update(question)

        # Correct answer
        correct_display = self.query_one("#correct-answer-display", Static)
        correct_answer = sample.get("correct_answer", sample.get("answer", "N/A"))

        # If it has reasoning, show both
        if "reasoning" in sample:
            correct_content = f"[bold]Reasoning:[/bold]\n{sample['reasoning']}\n\n"
            correct_content += f"[bold]Answer:[/bold]\n{correct_answer}"
            correct_display.update(correct_content)
        else:
            correct_display.update(str(correct_answer))

        # Model response
        response_display = self.query_one("#model-response-display", Static)
        model_response = sample.get("model_response", sample.get("response", "N/A"))

        # Parse model response if it has reasoning/answer tags
        if isinstance(model_response, str) and "<reasoning>" in model_response:
            response_display.update(self._format_response(model_response))
        else:
            response_display.update(str(model_response))

        # Metrics
        metrics_display = self.query_one("#sample-metrics", Static)
        metrics_text = self._format_metrics(sample)
        metrics_display.update(metrics_text)

    def _format_response(self, response: str) -> Syntax:
        """Format model response with syntax highlighting"""
        return Syntax(
            response,
            "text",
            theme="monokai",
            word_wrap=True,
        )

    def _format_metrics(self, sample: Dict) -> str:
        """Format sample metrics"""
        parts = []

        # Check if answers match
        correct_answer = sample.get("correct_answer")
        model_response = sample.get("model_response")

        if correct_answer is not None and model_response is not None:
            # Simple string comparison
            is_correct = str(correct_answer).strip() == str(model_response).strip()

            if is_correct:
                parts.append("[green]✓ Correct[/green]")
            else:
                parts.append("[red]✗ Incorrect[/red]")

        # Add score if available
        if "score" in sample:
            parts.append(f"Score: {sample['score']}")

        # Add reasoning score if available
        if "score_reasoning" in sample:
            parts.append(f"Reasoning: {sample['score_reasoning']}")

        return " | ".join(parts) if parts else "No metrics available"

    def action_next_sample(self) -> None:
        """Navigate to next sample"""
        if self.current_index < len(self.samples) - 1:
            self.current_index += 1
            self._display_sample()

    def action_previous_sample(self) -> None:
        """Navigate to previous sample"""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_sample()

    def action_back(self) -> None:
        """Go back"""
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Quit the application"""
        self.app.exit()
