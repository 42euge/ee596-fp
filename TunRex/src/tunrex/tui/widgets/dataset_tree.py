"""
Dataset file tree widget for browsing datasets
"""

from pathlib import Path
from textual.widgets import DirectoryTree
from textual.widgets._directory_tree import DirEntry


class DatasetTree(DirectoryTree):
    """
    Custom DirectoryTree that filters to show only dataset files
    """

    SUPPORTED_EXTENSIONS = {'.parquet', '.json', '.jsonl', '.csv'}

    def filter_paths(self, paths: list[Path]) -> list[Path]:
        """
        Filter paths to show only directories and supported dataset files

        Args:
            paths: List of paths to filter

        Returns:
            Filtered list of paths
        """
        filtered = []
        for path in paths:
            # Always show directories
            if path.is_dir():
                # Skip hidden directories and cache directories
                if not path.name.startswith('.') and path.name not in ['__pycache__', 'checkpoints']:
                    filtered.append(path)
            # Show supported file types
            elif path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                filtered.append(path)

        return filtered
