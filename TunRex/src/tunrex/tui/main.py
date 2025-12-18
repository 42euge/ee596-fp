"""Main TUI application for TunRex."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Static, ListView, ListItem, Label

from tunrex import __version__


class MenuItem(ListItem):
    """A menu item with a label and description."""

    def __init__(self, label: str, description: str, action: str) -> None:
        super().__init__()
        self.label = label
        self.description = description
        self.action = action

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.label}[/bold]")
        yield Label(f"[dim]{self.description}[/dim]")


class MainMenu(Container):
    """Main menu container."""

    def compose(self) -> ComposeResult:
        yield Static(
            f"[bold cyan]TunRex[/bold cyan] v{__version__}\n"
            "[dim]A flexible TUI/CLI toolkit[/dim]",
            id="header-text",
        )
        yield Static("\n[bold]Select an application:[/bold]\n", id="menu-label")
        yield ListView(
            MenuItem("Dataset Explorer", "Browse and compare ML datasets", "dataset"),
            MenuItem("Quit", "Exit TunRex", "quit"),
            id="main-menu",
        )


class MainApp(App):
    """The main TunRex TUI application."""

    TITLE = "TunRex"
    SUB_TITLE = "A flexible TUI/CLI toolkit"

    CSS = """
    Screen {
        align: center middle;
    }

    MainMenu {
        width: 60;
        height: auto;
        border: solid $primary;
        padding: 1 2;
    }

    #header-text {
        text-align: center;
        padding-bottom: 1;
    }

    #menu-label {
        text-align: center;
    }

    #main-menu {
        height: auto;
        max-height: 20;
        margin: 1 0;
    }

    MenuItem {
        padding: 1 2;
    }

    MenuItem > Label {
        width: 100%;
    }

    MenuItem:hover {
        background: $accent;
    }

    MenuItem.-active {
        background: $accent;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "open_dataset", "Dataset Explorer"),
        Binding("?", "help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield MainMenu()
        yield Footer()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle menu item selection."""
        item = event.item
        if isinstance(item, MenuItem):
            if item.action == "dataset":
                self.action_open_dataset()
            elif item.action == "quit":
                self.action_quit()

    def action_open_dataset(self) -> None:
        """Open the Dataset Explorer."""
        from tunrex.tui.dataset_explorer import DatasetExplorerApp

        # Exit main app and launch dataset explorer
        self.exit(result="dataset")

    def action_help(self) -> None:
        """Show help information."""
        self.notify(
            "Press 'd' for Dataset Explorer, 'q' to quit",
            title="TunRex Help",
        )


def main() -> None:
    """Run the main TunRex application with app switching support."""
    while True:
        app = MainApp()
        result = app.run()

        if result == "dataset":
            from tunrex.tui.dataset_explorer import DatasetExplorerApp

            dataset_app = DatasetExplorerApp()
            dataset_app.run()
            # After dataset explorer exits, loop back to main menu
        else:
            # No result or explicit quit - exit the loop
            break
