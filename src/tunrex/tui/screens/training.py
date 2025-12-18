"""
Training data exploration screen
"""

from pathlib import Path
from typing import List, Dict, Any
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, DataTable, Tree
from textual.widgets.tree import TreeNode
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from rich.syntax import Syntax
from rich.table import Table as RichTable
from rich.panel import Panel
import json


class TrainingDataScreen(Screen):
    """
    Specialized screen for exploring training session data
    """

    BINDINGS = [
        Binding("escape", "back", "Back", priority=True),
        Binding("q", "quit", "Quit"),
        Binding("d", "view_sample", "View Sample"),
        Binding("m", "view_metrics", "Metrics"),
    ]

    def __init__(self, training_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_dir = training_dir
        self.current_file = None
        self.current_data = None

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header()

        with Horizontal(id="training-container"):
            # Left: Training session tree
            with Vertical(id="training-nav", classes="nav-panel"):
                yield Static(
                    f"[bold cyan]Training Session[/bold cyan]\n{self.training_dir.name}",
                    classes="panel-title"
                )
                yield Tree("Training Data", id="training-tree")

            # Right: Content viewer
            with Vertical(id="training-viewer", classes="viewer-panel"):
                yield Static("Select a file to view", id="viewer-header", classes="info")
                with VerticalScroll(id="viewer-content"):
                    yield Static("", id="content-display")

        yield Footer()

    def on_mount(self) -> None:
        """Setup when screen is mounted"""
        self._populate_tree()

    def _populate_tree(self):
        """Populate the training data tree"""
        tree = self.query_one("#training-tree", Tree)
        tree.clear()

        root = tree.root
        root.expand()

        # Add evaluation data
        eval_dir = self.training_dir / "evaluation_data"
        if eval_dir.exists():
            eval_node = root.add("📊 Evaluation Data", expand=True)
            for file in sorted(eval_dir.glob("*.json")):
                # Parse filename for display
                if "pretrain" in file.name:
                    label = f"Pre-training: {file.name}"
                    eval_node.add_leaf(label, data={"path": file, "type": "eval_pretrain"})
                elif "posttrain" in file.name:
                    label = f"Post-training: {file.name}"
                    eval_node.add_leaf(label, data={"path": file, "type": "eval_posttrain"})

        # Add training responses
        responses_dir = self.training_dir / "training_responses"
        if responses_dir.exists():
            responses_node = root.add("💬 Training Responses", expand=True)
            for file in sorted(responses_dir.glob("*.json")):
                if "summary" in file.name:
                    label = f"Summary: {file.name}"
                    responses_node.add_leaf(label, data={"path": file, "type": "summary"})
                else:
                    label = f"Responses: {file.name}"
                    responses_node.add_leaf(label, data={"path": file, "type": "responses"})

        # Add checkpoints
        checkpoints_dir = self.training_dir / "checkpoints"
        if checkpoints_dir.exists():
            checkpoints_node = root.add("💾 Checkpoints", expand=False)
            for checkpoint in sorted(checkpoints_dir.glob("*/*/"), reverse=True):
                if checkpoint.is_dir():
                    checkpoints_node.add_leaf(
                        f"Step {checkpoint.parent.name}",
                        data={"path": checkpoint, "type": "checkpoint"}
                    )

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection"""
        if event.node.data:
            file_data = event.node.data
            file_path = file_data["path"]
            file_type = file_data["type"]

            if file_path.is_file():
                self._load_file(file_path, file_type)

    def _load_file(self, file_path: Path, file_type: str):
        """Load and display a training data file"""
        self.current_file = file_path

        # Update header
        header = self.query_one("#viewer-header", Static)
        header.update(f"[bold]{file_path.name}[/bold] | Type: {file_type}")

        # Load JSON data
        with open(file_path, 'r') as f:
            self.current_data = json.load(f)

        # Display based on type
        content_display = self.query_one("#content-display", Static)

        if file_type in ["eval_pretrain", "eval_posttrain"]:
            content_display.update(self._format_evaluation_data(self.current_data))
        elif file_type == "responses":
            content_display.update(self._format_training_responses(self.current_data))
        elif file_type == "summary":
            content_display.update(self._format_summary(self.current_data))
        else:
            # Generic JSON view
            content_display.update(self._format_json(self.current_data))

    def _format_evaluation_data(self, data: Dict) -> RichTable:
        """Format evaluation data for display"""
        table = RichTable(title="Evaluation Results", show_header=True, header_style="bold magenta")

        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green", width=15)

        # Metrics
        metrics = data.get("metrics", {})
        table.add_row("Timestamp", data.get("timestamp", "N/A"))
        table.add_row("Stage", data.get("stage", "N/A"))
        table.add_row("Checkpoint Step", str(data.get("checkpoint_step", "N/A")))
        table.add_row("", "")  # Separator
        table.add_row("Correct", str(metrics.get("correct", 0)))
        table.add_row("Total", str(metrics.get("total", 0)))
        table.add_row("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        table.add_row("Partial Accuracy", f"{metrics.get('partial_accuracy', 0):.2%}")
        table.add_row("Format Accuracy", f"{metrics.get('format_accuracy', 0):.2%}")

        # Samples info
        samples = data.get("samples", [])
        table.add_row("", "")  # Separator
        table.add_row("Total Samples", str(len(samples)))

        if samples:
            correct_samples = sum(1 for s in samples if s.get("correct_answer") == s.get("model_response"))
            table.add_row("Correct Samples", str(correct_samples))

        return table

    def _format_training_responses(self, data: List[Dict]) -> Static:
        """Format training responses for display"""
        if isinstance(data, list):
            total = len(data)
            # Show summary
            summary = f"[bold]Training Responses[/bold]\n\n"
            summary += f"Total responses: {total:,}\n\n"
            summary += "[dim]Press 'd' to view individual samples[/dim]\n\n"

            # Show first few questions
            summary += "[bold]Sample Questions:[/bold]\n"
            for i, item in enumerate(data[:5]):
                question = item.get("question", "N/A")
                if len(question) > 100:
                    question = question[:100] + "..."
                summary += f"{i+1}. {question}\n"

            return Static(summary)
        else:
            return Static(str(data))

    def _format_summary(self, data: Dict) -> Syntax:
        """Format summary data as JSON"""
        json_str = json.dumps(data, indent=2)
        return Syntax(json_str, "json", theme="monokai", line_numbers=False)

    def _format_json(self, data: Any) -> Syntax:
        """Format generic JSON data"""
        json_str = json.dumps(data, indent=2)
        return Syntax(json_str, "json", theme="monokai", line_numbers=True, word_wrap=True)

    def action_view_sample(self) -> None:
        """View a specific sample from evaluation or training data"""
        if not self.current_data:
            return

        # Check if this is evaluation data with samples
        if isinstance(self.current_data, dict) and "samples" in self.current_data:
            samples = self.current_data["samples"]
            if samples:
                # Show first sample (could be enhanced to navigate through samples)
                from tunrex.tui.screens.training_sample import TrainingSampleScreen
                self.app.push_screen(TrainingSampleScreen(samples, 0))

        # Check if this is training responses list
        elif isinstance(self.current_data, list) and len(self.current_data) > 0:
            from tunrex.tui.screens.training_sample import TrainingSampleScreen
            self.app.push_screen(TrainingSampleScreen(self.current_data, 0))

    def action_view_metrics(self) -> None:
        """View detailed metrics"""
        # TODO: Implement metrics comparison view
        pass

    def action_back(self) -> None:
        """Go back to browser"""
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Quit the application"""
        self.app.exit()
