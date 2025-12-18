# TunRex

A flexible TUI/CLI toolkit.

## Installation

```bash
uv pip install tunrex
```

Or for development:

```bash
git clone git@github.com:42euge/TunRex.git
cd TunRex
uv sync
```

## Usage

### Main TUI

Launch the main TUI dashboard:

```bash
trex
```

### Specific TUI Apps

Launch a specific TUI application:

```bash
trex app <app_name>
```

List available apps:

```bash
trex cli list-apps
```

### Pure CLI Mode

Run CLI commands without TUI:

```bash
trex cli <command>
```

Available CLI commands:

- `trex cli info` - Show TunRex information
- `trex cli list-apps` - List available TUI applications

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run linter
uv run ruff check .
```

## License

MIT
