"""
Shared Rich Console instance and top-level Arch/Cyberpunk-inspired UI theme & starting banner.
Rendering lives exclusively here per Section 19 of the architecture spec.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from proton_agent import __version__

if TYPE_CHECKING:
    from proton_agent.context.session import SessionState

_console: Console | None = None


def get_console() -> Console:
    """Process-wide Console singleton so every part of the CLI prints
    through the same instance."""
    global _console
    if _console is None:
        _console = Console()
    return _console


def show_banner(console: Console, session: Optional[SessionState] = None) -> None:
    """Renders a visually striking, modern starting dashboard for `proton start`."""
    ascii_banner = (
        "  ██████╗ ██████╗  ██████╗ ████████╗ ██████╗ ███╗   ██╗\n"
        "  ██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗████╗  ██║\n"
        "  ██████╔╝██████╔╝██║   ██║   ██║   ██║   ██║██╔██╗ ██║\n"
        "  ██╔═══╝ ██╔══██╗██║   ██║   ██║   ██║   ██║██║╚██╗██║\n"
        "  ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚████║\n"
        "  ╚═╝     ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═══╝"
    )

    banner_text = Text(ascii_banner, style="bold bright_cyan", justify="center")

    # Determine LLM Provider status
    has_api_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    if has_api_key:
        llm_status_text = "[bold green]● ONLINE[/bold green] [bright_cyan]Gemini 1.5 Flash (Generative AI Active)[/bright_cyan]"
    else:
        llm_status_text = "[bold yellow]● HEURISTIC MODE[/bold yellow] [dim white]Rule Engine Active (Set GEMINI_API_KEY in .env for Full Custom AI)[/dim white]"

    # Build Engine Status Table
    status_table = Table.grid(padding=(0, 2))
    status_table.add_column(style="dim white", justify="left")
    status_table.add_column(style="bold white", justify="left")

    status_table.add_row("Agent Version:", f"[bold bright_cyan]v{__version__}[/bold bright_cyan]")
    status_table.add_row("LLM Engine:", llm_status_text)
    status_table.add_row("Core Stack:", "[bold white]Python 3.12 | Pandas | Scikit-Learn | Matplotlib[/bold white]")

    if session and session.current_dataset:
        ds = session.current_dataset
        ds_text = f"[bold green]✓ Loaded:[/bold green] [bold bright_cyan]{ds.name}[/bold bright_cyan] [dim]({ds.rows:,} rows × {ds.columns_count:,} cols)[/dim]"
    else:
        ds_text = "[bold yellow]○ None[/bold yellow] [dim white](Use 'load train <file.csv>' to load data)[/dim white]"

    status_table.add_row("Train Dataset:", ds_text)

    if session and session.test_dataset:
        test_ds = session.test_dataset
        test_ds_text = f"[bold green]✓ Loaded:[/bold green] [bold bright_cyan]{test_ds.name}[/bold bright_cyan] [dim]({test_ds.rows:,} rows × {test_ds.columns_count:,} cols)[/dim]"
        status_table.add_row("Test Dataset:", test_ds_text)

    # Build Quick Command Showcase Grid
    cmds_table = Table(
        title="[bold bright_white]⚡ Quick Command Showcase[/bold bright_white]",
        header_style="bold bright_cyan",
        border_style="cyan",
        show_lines=False,
        padding=(0, 1),
    )
    cmds_table.add_column("Category", style="bold cyan", min_width=18)
    cmds_table.add_column("Example Natural Language Command", style="bold bright_white")

    cmds_table.add_row("📁 Ingestion", "load train <file.csv>  |  load test <test.csv>")
    cmds_table.add_row("🔍 EDA & Quality", "analyze dataset  |  audit data quality  |  detect missing values")
    cmds_table.add_row("🧹 Preprocessing", "clean dataset  |  impute missing  |  encode features")
    cmds_table.add_row("🤖 Machine Learning", "train model to predict churn  |  compare models  |  predict")
    cmds_table.add_row("📊 Visualization", "plot distribution of age  |  plot correlation matrix")
    cmds_table.add_row("📄 Reporting", "generate report  |  help  |  exit")

    # Combine into main layout with centered column alignment
    layout_grid = Table.grid(expand=True, padding=(0, 1))
    layout_grid.add_column(justify="center")
    layout_grid.add_row(banner_text)
    layout_grid.add_row("")
    layout_grid.add_row(Panel(status_table, title="[bold bright_white]System Status & Engine Capabilities[/bold bright_white]", border_style="bright_blue", padding=(0, 2)))
    layout_grid.add_row("")
    layout_grid.add_row(cmds_table)

    main_panel = Panel(
        layout_grid,
        title="[bold bright_cyan]▲ PROTON (prot)[/bold bright_cyan]",
        subtitle="[bold dim white]Autonomous Interactive Terminal Environment[/bold dim white]",
        border_style="bright_cyan",
        padding=(1, 2),
        expand=False,
    )

    console.print()
    console.print(main_panel)
    console.print()
