"""TUI screens for TunRex Dataset Explorer."""

from tunrex.tui.screens.browser import BrowserScreen
from tunrex.tui.screens.comparison import ComparisonScreen
from tunrex.tui.screens.detail import DetailScreen
from tunrex.tui.screens.training import TrainingDataScreen
from tunrex.tui.screens.training_sample import TrainingSampleScreen

__all__ = [
    "BrowserScreen",
    "ComparisonScreen",
    "DetailScreen",
    "TrainingDataScreen",
    "TrainingSampleScreen",
]
