"""
Data viewer widget for displaying datasets
"""

from pathlib import Path
from typing import List, Dict, Optional
from textual.app import ComposeResult
from textual.widgets import DataTable, Static
from textual.containers import Container, VerticalScroll
from textual.reactive import reactive
from rich.syntax import Syntax
from tunrex.tui.helpers.loaders import load_dataset
from tunrex.tui.helpers.formatters import format_value, format_json_pretty


class DataViewer(Container):
    """
    Widget for viewing dataset contents
    Supports both tabular view (DataTable) and JSON view (Syntax)
    """

    current_file = reactive(None)
    current_page = reactive(0)
    rows_per_page = reactive(100)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_rows = 0
        self.columns = []
        self.data = []

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        # Info bar showing file info
        yield Static("No file loaded", id="info-bar", classes="info")

        # Main data display
        with VerticalScroll(id="data-container"):
            yield DataTable(id="data-table", zebra_stripes=True)
            yield Static("", id="json-view")

        # Pagination controls
        yield Static("", id="pagination-bar", classes="info")

    def load_file(self, file_path: Path):
        """
        Load and display a dataset file

        Args:
            file_path: Path to the dataset file
        """
        self.current_file = file_path
        self.current_page = 0

        try:
            # Load data
            start_row = self.current_page * self.rows_per_page
            self.data, self.total_rows, self.columns = load_dataset(
                file_path, start_row, self.rows_per_page
            )

            # Update info bar
            info_bar = self.query_one("#info-bar", Static)
            info_bar.update(
                f"[bold]{file_path.name}[/bold] | "
                f"{self.total_rows:,} rows | "
                f"{len(self.columns)} columns"
            )

            # Detect if this is JSON/JSONL with single object or few columns
            if file_path.suffix.lower() in ['.json', '.jsonl'] and len(self.columns) <= 5:
                self._display_json_view()
            else:
                self._display_table_view()

            self._update_pagination()

        except Exception as e:
            info_bar = self.query_one("#info-bar", Static)
            info_bar.update(f"[red]Error loading file: {str(e)}[/red]")

    def _display_table_view(self):
        """Display data in table format"""
        table = self.query_one("#data-table", DataTable)
        json_view = self.query_one("#json-view", Static)

        # Show table, hide JSON view
        table.display = True
        json_view.display = False

        # Clear and setup table
        table.clear(columns=True)

        # Add columns
        for col in self.columns:
            table.add_column(col, key=col)

        # Add rows
        for row in self.data:
            # Format values for display
            row_values = [format_value(row.get(col, ""), max_length=60) for col in self.columns]
            table.add_row(*row_values)

    def _display_json_view(self):
        """Display data in JSON format with syntax highlighting"""
        table = self.query_one("#data-table", DataTable)
        json_view = self.query_one("#json-view", Static)

        # Hide table, show JSON view
        table.display = False
        json_view.display = True

        # Format as JSON
        json_str = format_json_pretty(self.data)

        # Create syntax-highlighted view
        syntax = Syntax(
            json_str,
            "json",
            theme="monokai",
            line_numbers=True,
            word_wrap=False,
        )

        json_view.update(syntax)

    def _update_pagination(self):
        """Update pagination information"""
        total_pages = (self.total_rows + self.rows_per_page - 1) // self.rows_per_page
        current_page_num = self.current_page + 1

        start_row = self.current_page * self.rows_per_page + 1
        end_row = min((self.current_page + 1) * self.rows_per_page, self.total_rows)

        pagination_bar = self.query_one("#pagination-bar", Static)
        pagination_bar.update(
            f"Showing rows {start_row:,}-{end_row:,} of {self.total_rows:,} | "
            f"Page {current_page_num}/{total_pages} | "
            f"[dim]← → for pagination, d to view details[/dim]"
        )

    def next_page(self):
        """Go to next page"""
        total_pages = (self.total_rows + self.rows_per_page - 1) // self.rows_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            if self.current_file:
                self.load_file(self.current_file)

    def previous_page(self):
        """Go to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            if self.current_file:
                self.load_file(self.current_file)

    def get_selected_row(self) -> Optional[Dict]:
        """
        Get the currently selected row data from the DataTable

        Returns:
            Dict of row data or None if no selection
        """
        table = self.query_one("#data-table", DataTable)

        if not table.display or table.cursor_row < 0:
            return None

        # Get the row index in current page
        row_idx = table.cursor_row

        if row_idx < len(self.data):
            return self.data[row_idx]

        return None

    def get_selected_row_index(self) -> Optional[int]:
        """
        Get the global row index of the currently selected row

        Returns:
            Global row index or None
        """
        table = self.query_one("#data-table", DataTable)

        if not table.display or table.cursor_row < 0:
            return None

        # Calculate global index: page offset + table cursor position
        global_index = (self.current_page * self.rows_per_page) + table.cursor_row

        if global_index < self.total_rows:
            return global_index

        return None

    def clear_view(self):
        """Clear the current view"""
        info_bar = self.query_one("#info-bar", Static)
        info_bar.update("No file loaded")

        table = self.query_one("#data-table", DataTable)
        table.clear(columns=True)

        json_view = self.query_one("#json-view", Static)
        json_view.update("")

        pagination_bar = self.query_one("#pagination-bar", Static)
        pagination_bar.update("")
