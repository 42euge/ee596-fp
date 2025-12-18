"""
Detail view screen for examining a single dataset row
"""

from typing import Dict, Any
import numpy as np
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table as RichTable
from tunrex.tui.helpers.formatters import format_json_pretty, extract_reasoning_answer


class DetailScreen(Screen):
    """
    Screen for viewing a single dataset row in detail
    Shows fields side-by-side or in formatted layout
    """

    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("q", "quit", "Quit"),
        Binding("left", "previous_row", "Previous"),
        Binding("right", "next_row", "Next"),
    ]

    def __init__(self, row_data: Dict[str, Any], row_index: int, total_rows: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_data = self._convert_to_serializable(row_data)
        self.row_index = row_index
        self.total_rows = total_rows

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        with Vertical(id="detail-container"):
            # Row info header
            yield Static(
                f"[bold]Row {self.row_index + 1} of {self.total_rows}[/bold]",
                id="detail-header",
                classes="info"
            )

            # Main content area
            with Horizontal(id="detail-content"):
                # Check if this has question/reasoning/solution structure (OpenThoughts format)
                if "question" in self.row_data and ("reasoning" in self.row_data or "solution" in self.row_data):
                    # Three-panel view for question/reasoning/solution
                    with Vertical(classes="detail-panel"):
                        yield Static("[bold cyan]Question[/bold cyan]", classes="panel-title")
                        with VerticalScroll(classes="detail-scroll"):
                            yield Static(
                                self._format_field(self.row_data.get("question", "")),
                                id="question-content"
                            )

                    # Show reasoning if present and non-empty
                    reasoning = self.row_data.get("reasoning", "")
                    if reasoning:
                        with Vertical(classes="detail-panel"):
                            yield Static("[bold yellow]Reasoning[/bold yellow]", classes="panel-title")
                            with VerticalScroll(classes="detail-scroll"):
                                yield Static(
                                    self._format_field(reasoning),
                                    id="reasoning-content"
                                )

                    # Show solution/answer
                    solution = self.row_data.get("solution", self.row_data.get("answer", ""))
                    with Vertical(classes="detail-panel"):
                        yield Static("[bold green]Solution[/bold green]", classes="panel-title")
                        with VerticalScroll(classes="detail-scroll"):
                            yield Static(
                                self._format_field(solution),
                                id="solution-content"
                            )

                # Check if this has input/reasoning/answer (prepared format)
                elif "input" in self.row_data and ("reasoning" in self.row_data or "answer" in self.row_data):
                    # Split view for input and reasoning/answer
                    with Vertical(classes="detail-panel"):
                        yield Static("[bold cyan]Input[/bold cyan]", classes="panel-title")
                        with VerticalScroll(classes="detail-scroll"):
                            yield Static(
                                self._format_field(self.row_data.get("input", "")),
                                id="input-content"
                            )

                    with Vertical(classes="detail-panel"):
                        yield Static("[bold green]Solution[/bold green]", classes="panel-title")
                        with VerticalScroll(classes="detail-scroll"):
                            solution = self._format_solution(self.row_data)
                            yield Static(solution, id="solution-content")

                # Check if there's a full_prompt field
                elif "full_prompt" in self.row_data:
                    # Show full prompt with highlighting
                    with VerticalScroll(id="full-prompt-container"):
                        yield Static(
                            self._format_full_prompt(self.row_data["full_prompt"]),
                            id="full-prompt-content"
                        )

                else:
                    # Generic view - show all fields
                    with VerticalScroll(id="generic-container"):
                        yield Static(
                            self._format_generic(self.row_data),
                            id="generic-content"
                        )

        yield Footer()

    def _format_field(self, value: Any) -> str:
        """Format a single field value"""
        if isinstance(value, (dict, list)):
            return format_json_pretty(value)
        return str(value)

    def _format_solution(self, row_data: Dict) -> Syntax:
        """Format reasoning and answer together"""
        parts = []

        if "reasoning" in row_data:
            parts.append("[bold]Reasoning:[/bold]")
            parts.append(str(row_data["reasoning"]))
            parts.append("")

        if "answer" in row_data:
            parts.append("[bold]Answer:[/bold]")
            parts.append(str(row_data["answer"]))

        content = "\n".join(parts)

        return Syntax(
            content,
            "text",
            theme="monokai",
            word_wrap=True,
        )

    def _format_full_prompt(self, full_prompt: str) -> Syntax:
        """Format full prompt with syntax highlighting"""
        return Syntax(
            full_prompt,
            "text",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )

    def _format_generic(self, row_data: Dict) -> Syntax:
        """Format all fields as JSON"""
        json_str = format_json_pretty(row_data)
        return Syntax(
            json_str,
            "json",
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )

    def action_back(self) -> None:
        """Go back to previous screen"""
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Quit the application"""
        self.app.exit()

    def action_previous_row(self) -> None:
        """Navigate to previous row"""
        # This would require passing a callback to get the previous row
        # For now, just go back
        self.app.pop_screen()

    def action_next_row(self) -> None:
        """Navigate to next row"""
        # This would require passing a callback to get the next row
        # For now, just go back
        self.app.pop_screen()
