"""
Main browser screen for exploring datasets
"""

from pathlib import Path
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, DataTable
from textual.containers import Horizontal
from textual.binding import Binding
from tunrex.tui.widgets import DatasetTree, DataViewer
from tunrex.tui.screens.detail import DetailScreen
from tunrex.tui.screens.training import TrainingDataScreen


class BrowserScreen(Screen):
    """
    Main screen for browsing and previewing datasets
    Shows file tree on left, data preview on right
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("c", "compare_mode", "Compare", priority=True),
        Binding("d", "view_detail", "Detail", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("right", "next_page", "Next Page"),
        Binding("left", "previous_page", "Prev Page"),
        Binding("?", "help", "Help"),
    ]

    def __init__(self, start_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_path = start_path

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        with Horizontal(id="browser-container"):
            # File tree on the left (30% width)
            yield DatasetTree(str(self.start_path), id="dataset-tree")

            # Data viewer on the right (70% width)
            yield DataViewer(id="data-viewer")

        yield Footer()

    def on_mount(self) -> None:
        """Set up event handlers when screen is mounted"""
        # Focus the tree initially
        self.query_one("#dataset-tree").focus()

    def on_directory_tree_directory_selected(self, event) -> None:
        """
        Handle directory selection from directory tree

        Args:
            event: DirectoryTree.DirectorySelected event
        """
        dir_path = event.path

        # Check if this is a training session directory
        if self._is_training_session(dir_path):
            # Launch training data viewer
            training_screen = TrainingDataScreen(dir_path)
            self.app.push_screen(training_screen)

    def on_directory_tree_file_selected(self, event) -> None:
        """
        Handle file selection from directory tree

        Args:
            event: DirectoryTree.FileSelected event
        """
        file_path = event.path

        # Load the file in the data viewer
        viewer = self.query_one("#data-viewer", DataViewer)
        viewer.load_file(file_path)

    def _is_training_session(self, dir_path: Path) -> bool:
        """
        Check if a directory is a training session

        Args:
            dir_path: Path to directory

        Returns:
            True if it's a training session directory
        """
        # Training sessions have evaluation_data and/or training_responses directories
        has_evaluation = (dir_path / "evaluation_data").exists()
        has_responses = (dir_path / "training_responses").exists()
        has_checkpoints = (dir_path / "checkpoints").exists()

        return has_evaluation or has_responses or has_checkpoints

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """
        Handle row selection in the data table (Enter key)

        Args:
            event: DataTable.RowSelected event
        """
        viewer = self.query_one("#data-viewer", DataViewer)

        # Get the selected row data
        row_data = viewer.get_selected_row()
        row_index = viewer.get_selected_row_index()

        if row_data is not None and row_index is not None:
            # Push detail screen
            detail_screen = DetailScreen(
                row_data=row_data,
                row_index=row_index,
                total_rows=viewer.total_rows
            )
            self.app.push_screen(detail_screen)

    def action_next_page(self) -> None:
        """Navigate to next page in data viewer"""
        viewer = self.query_one("#data-viewer", DataViewer)
        viewer.next_page()

    def action_previous_page(self) -> None:
        """Navigate to previous page in data viewer"""
        viewer = self.query_one("#data-viewer", DataViewer)
        viewer.previous_page()

    def action_view_detail(self) -> None:
        """View details of the selected row"""
        viewer = self.query_one("#data-viewer", DataViewer)

        # Get the selected row data
        row_data = viewer.get_selected_row()
        row_index = viewer.get_selected_row_index()

        if row_data is not None and row_index is not None:
            # Push detail screen
            detail_screen = DetailScreen(
                row_data=row_data,
                row_index=row_index,
                total_rows=viewer.total_rows
            )
            self.app.push_screen(detail_screen)

    def action_compare_mode(self) -> None:
        """Switch to comparison mode"""
        self.app.push_screen("comparison")

    def action_refresh(self) -> None:
        """Refresh the dataset tree"""
        tree = self.query_one("#dataset-tree", DatasetTree)
        tree.reload()

    def action_help(self) -> None:
        """Show help dialog"""
        # TODO: Implement help dialog
        pass

    def action_quit(self) -> None:
        """Quit the application"""
        self.app.exit()
