"""
Main Dataset Explorer TUI Application
"""

from pathlib import Path

from textual.app import App

from tunrex.tui.screens import BrowserScreen, ComparisonScreen


class DatasetExplorerApp(App):
    """
    Terminal UI for exploring and comparing datasets
    """

    CSS_PATH = "styles.tcss"
    TITLE = "TunRex Dataset Explorer"
    SUB_TITLE = "Browse and compare ML datasets"

    def __init__(self, start_path: Path | None = None):
        super().__init__()

        # Default to data directory
        if start_path is None:
            start_path = Path.cwd() / "data"

        # Ensure path exists
        if not start_path.exists():
            start_path = Path.cwd()

        self.start_path = start_path

    def on_mount(self) -> None:
        """Set up the application when it starts"""
        # Install screens
        self.install_screen(BrowserScreen(self.start_path), name="browser")
        self.install_screen(ComparisonScreen(), name="comparison")

        # Start with browser screen
        self.push_screen("browser")


def run_explorer(start_path: Path | None = None):
    """
    Run the dataset explorer TUI

    Args:
        start_path: Optional starting directory path
    """
    app = DatasetExplorerApp(start_path=start_path)
    app.run()
