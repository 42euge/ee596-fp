"""CLI entry point for TunRex."""

import click

from tunrex import __version__


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="trex")
@click.pass_context
def main(ctx: click.Context) -> None:
    """TunRex - A flexible TUI/CLI toolkit.

    Run without arguments to open the main TUI.
    Use 'trex <app>' to open a specific TUI app.
    Use 'trex cli <command>' for pure CLI operations.
    """
    if ctx.invoked_subcommand is None:
        # No subcommand provided, launch main TUI with app switching
        from tunrex.tui.main import main as run_main_tui

        run_main_tui()


@main.command()
@click.argument("app_name")
def app(app_name: str) -> None:
    """Launch a specific TUI application.

    APP_NAME is the name of the TUI app to launch.
    """
    from tunrex.tui import get_app

    tui_app = get_app(app_name)
    if tui_app is None:
        click.echo(f"Error: Unknown TUI app '{app_name}'", err=True)
        click.echo("Available apps: main", err=True)
        raise SystemExit(1)
    tui_app().run()


@main.group()
def cli() -> None:
    """Pure CLI commands (no TUI)."""
    pass


@cli.command()
def info() -> None:
    """Show TunRex information."""
    click.echo(f"TunRex v{__version__}")
    click.echo("A flexible TUI/CLI toolkit")


@cli.command()
def list_apps() -> None:
    """List available TUI applications."""
    from tunrex.tui import AVAILABLE_APPS

    click.echo("Available TUI applications:")
    for name, description in AVAILABLE_APPS.items():
        click.echo(f"  {name}: {description}")


if __name__ == "__main__":
    main()
