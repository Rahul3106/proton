"""
CLI entrypoint for the `proton` command.

Phase 2 scope: `proton start` launches an interactive session (see
cli/repl.py for the loop itself - kept out of this file so command
wiring and session logic stay independently testable).
"""
from __future__ import annotations

import typer

from proton_agent import __version__
from proton_agent.cli.repl import Repl
from proton_agent.ui.console import get_console

app = typer.Typer(
    name="proton",
    help="Proton Data Science Agent - an autonomous Data Science Agent you talk to through a terminal.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Proton Data Science Agent (proton)."""


@app.command()
def version() -> None:
    """Show the installed proton version."""
    typer.echo(f"proton {__version__}")


@app.command()
def start() -> None:
    """Launch an interactive proton session."""
    Repl(get_console()).run()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the FastAPI HTTP server using Uvicorn."""
    try:
        import uvicorn
        from proton_agent.api import app as api_app

        typer.echo(f"Starting Proton-Agent API on http://{host}:{port}")
        uvicorn.run(api_app, host=host, port=port)
    except ImportError:
        typer.echo("Error: FastAPI or Uvicorn is not installed. Install with 'pip install fastapi uvicorn'.")


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Alias for 'serve': Launch the FastAPI HTTP server."""
    serve(host=host, port=port)



def main() -> None:
    app()


if __name__ == "__main__":
    main()
