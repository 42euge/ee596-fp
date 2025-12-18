"""
Comparison panel widget for side-by-side dataset comparison
"""

from pathlib import Path
from typing import Optional
from textual.app import ComposeResult
from textual.widgets import Static
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from rich.syntax import Syntax
from rich.table import Table as RichTable
from tunrex.tui.helpers.loaders import load_dataset
from tunrex.tui.helpers.formatters import format_json_pretty


class ComparisonPanel(Container):
    """
    Widget for comparing two datasets side-by-side
    Shows original dataset on left, prepared dataset on right
    """

    current_index = reactive(0)
    total_samples = reactive(0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_data = []
        self.right_data = []
        self.left_file = None
        self.right_file = None

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Static("No datasets loaded for comparison", id="comparison-header", classes="info")

        with Horizontal(id="comparison-container"):
            # Left panel (original)
            with Vertical(classes="comparison-side"):
                yield Static("[bold cyan]Original Dataset[/bold cyan]", classes="panel-header")
                with VerticalScroll(classes="comparison-scroll"):
                    yield Static("", id="left-panel")

            # Right panel (prepared)
            with Vertical(classes="comparison-side"):
                yield Static("[bold green]Prepared Dataset[/bold green]", classes="panel-header")
                with VerticalScroll(classes="comparison-scroll"):
                    yield Static("", id="right-panel")

        yield Static("", id="comparison-nav", classes="info")

    def load_comparison(self, left_file: Path, right_file: Path):
        """
        Load two datasets for comparison

        Args:
            left_file: Path to original dataset
            right_file: Path to prepared dataset
        """
        self.left_file = left_file
        self.right_file = right_file
        self.current_index = 0

        try:
            # Load both datasets
            self.left_data, left_total, left_cols = load_dataset(left_file, 0, 1000)
            self.right_data, right_total, right_cols = load_dataset(right_file, 0, 1000)

            self.total_samples = min(left_total, right_total)

            # Update header
            header = self.query_one("#comparison-header", Static)
            header.update(
                f"Comparing: [cyan]{left_file.name}[/cyan] ({left_total} rows) ↔ "
                f"[green]{right_file.name}[/green] ({right_total} rows)"
            )

            # Display first sample
            self._display_current_sample()

        except Exception as e:
            header = self.query_one("#comparison-header", Static)
            header.update(f"[red]Error loading datasets: {str(e)}[/red]")

    def _display_current_sample(self):
        """Display the current sample in both panels"""
        if not self.left_data or not self.right_data:
            return

        if self.current_index >= len(self.left_data) or self.current_index >= len(self.right_data):
            return

        left_sample = self.left_data[self.current_index]
        right_sample = self.right_data[self.current_index]

        # Update left panel
        left_panel = self.query_one("#left-panel", Static)
        left_panel.update(self._format_sample(left_sample, "original"))

        # Update right panel
        right_panel = self.query_one("#right-panel", Static)
        right_panel.update(self._format_sample(right_sample, "prepared"))

        # Update navigation
        nav = self.query_one("#comparison-nav", Static)
        nav.update(
            f"Sample {self.current_index + 1} of {self.total_samples} | "
            f"[dim]← → to navigate, Esc to exit[/dim]"
        )

    def _format_sample(self, sample: dict, dataset_type: str) -> Syntax:
        """
        Format a single sample for display

        Args:
            sample: Sample data dict
            dataset_type: Type of dataset ('original' or 'prepared')

        Returns:
            Formatted Rich Syntax object
        """
        # For prepared datasets, highlight the reasoning/answer structure
        if dataset_type == "prepared" and 'full_prompt' in sample:
            content = sample['full_prompt']
            lexer = "text"
        else:
            # Show as formatted JSON
            content = format_json_pretty(sample)
            lexer = "json"

        return Syntax(
            content,
            lexer,
            theme="monokai",
            line_numbers=False,
            word_wrap=True,
        )

    def next_sample(self):
        """Navigate to next sample"""
        if self.current_index < self.total_samples - 1:
            self.current_index += 1
            self._display_current_sample()

    def previous_sample(self):
        """Navigate to previous sample"""
        if self.current_index > 0:
            self.current_index -= 1
            self._display_current_sample()

    def clear_comparison(self):
        """Clear the comparison view"""
        header = self.query_one("#comparison-header", Static)
        header.update("No datasets loaded for comparison")

        left_panel = self.query_one("#left-panel", Static)
        left_panel.update("")

        right_panel = self.query_one("#right-panel", Static)
        right_panel.update("")

        nav = self.query_one("#comparison-nav", Static)
        nav.update("")

        self.left_data = []
        self.right_data = []
