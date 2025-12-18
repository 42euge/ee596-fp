"""
Comparison screen for side-by-side dataset comparison
"""

from pathlib import Path
from typing import Optional, Tuple
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, DirectoryTree, Static
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from tunrex.tui.widgets import DatasetTree, ComparisonPanel


class ComparisonScreen(Screen):
    """
    Screen for comparing original and prepared datasets side-by-side
    """

    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("q", "quit", "Quit"),
        Binding("right", "next_sample", "Next"),
        Binding("left", "previous_sample", "Previous"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_file = None
        self.right_file = None

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        with Container(id="comparison-main"):
            # Instructions
            yield Static(
                "[bold]Comparison Mode[/bold]\n"
                "Select two files to compare:\n"
                "1. Select original dataset (left)\n"
                "2. Select prepared dataset (right)\n"
                "Use Tab to switch between trees",
                id="comparison-instructions",
                classes="info"
            )

            # File selection trees
            with Horizontal(id="file-selection", classes="hidden"):
                with Vertical(classes="selection-side"):
                    yield Static("[cyan]Original Dataset[/cyan]", classes="label")
                    # Get data directory
                    data_dir = Path.cwd() / "data"
                    yield DatasetTree(str(data_dir / "sets"), id="left-tree")

                with Vertical(classes="selection-side"):
                    yield Static("[green]Prepared Dataset[/green]", classes="label")
                    yield DatasetTree(str(data_dir / "input" / "datasets"), id="right-tree")

            # Comparison panel (hidden until both files selected)
            yield ComparisonPanel(id="comparison-panel", classes="hidden")

        yield Footer()

    def on_mount(self) -> None:
        """Setup when screen is mounted"""
        # Show file selection initially
        instructions = self.query_one("#comparison-instructions")
        instructions.display = True

        file_selection = self.query_one("#file-selection")
        file_selection.remove_class("hidden")

        # Focus left tree
        self.query_one("#left-tree").focus()

    def on_directory_tree_file_selected(self, event) -> None:
        """
        Handle file selection from either tree

        Args:
            event: DirectoryTree.FileSelected event
        """
        file_path = event.path

        # Determine which tree was clicked
        if event.control.id == "left-tree":
            self.left_file = file_path
            self._update_status()
        elif event.control.id == "right-tree":
            self.right_file = file_path
            self._update_status()

        # If both files selected, load comparison
        if self.left_file and self.right_file:
            self._load_comparison()

    def _update_status(self):
        """Update status message"""
        instructions = self.query_one("#comparison-instructions")

        left_name = self.left_file.name if self.left_file else "[dim]not selected[/dim]"
        right_name = self.right_file.name if self.right_file else "[dim]not selected[/dim]"

        instructions.update(
            f"[bold]Select files to compare:[/bold]\n"
            f"Original: {left_name}\n"
            f"Prepared: {right_name}\n\n"
            f"[dim]Use Tab to switch between trees, Enter to select[/dim]"
        )

    def _load_comparison(self):
        """Load and display the comparison"""
        # Hide file selection
        file_selection = self.query_one("#file-selection")
        file_selection.add_class("hidden")

        instructions = self.query_one("#comparison-instructions")
        instructions.display = False

        # Show and load comparison panel
        panel = self.query_one("#comparison-panel", ComparisonPanel)
        panel.remove_class("hidden")
        panel.load_comparison(self.left_file, self.right_file)

    def action_next_sample(self) -> None:
        """Navigate to next sample"""
        if self.left_file and self.right_file:
            panel = self.query_one("#comparison-panel", ComparisonPanel)
            panel.next_sample()

    def action_previous_sample(self) -> None:
        """Navigate to previous sample"""
        if self.left_file and self.right_file:
            panel = self.query_one("#comparison-panel", ComparisonPanel)
            panel.previous_sample()

    def action_back(self) -> None:
        """Go back to browser screen"""
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Quit the application"""
        self.app.exit()
