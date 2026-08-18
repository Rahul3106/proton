"""
Tests for interactive REPL commands in Phase 3.
"""
from pathlib import Path
from rich.console import Console
import pandas as pd
import pytest

from proton_agent.cli.repl import Repl
from proton_agent.context import SessionState


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10.5, 20.0, 30.2],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_repl_load_command(sample_csv: Path):
    console = Console(record=True)
    session = SessionState()
    repl = Repl(console=console, session=session)

    repl.handle(f"load {sample_csv}")

    output = console.export_text()
    assert "✓ Dataset loaded" in output
    assert "3" in output  # rows count
    assert session.current_dataset is not None
    assert session.current_dataset.rows == 3


def test_repl_inspect_command(sample_csv: Path):
    console = Console(record=True)
    session = SessionState()
    repl = Repl(console=console, session=session)

    # Inspect before load
    repl.handle("inspect")
    out1 = console.export_text()
    assert "No dataset loaded" in out1

    # Load dataset then inspect
    repl.handle(f"load {sample_csv}")
    repl.handle("inspect")
    out2 = console.export_text()
    assert "Dataset Overview" in out2
    assert "Schema & Missing Value Distribution" in out2
    assert "Data Preview" in out2


def test_repl_load_not_found():
    console = Console(record=True)
    session = SessionState()
    repl = Repl(console=console, session=session)

    repl.handle("load non_existent_file.csv")
    output = console.export_text()
    assert "Dataset Load Error" in output or "File not found" in output


def test_repl_strips_prompt_prefix(sample_csv: Path):
    console = Console(record=True)
    session = SessionState()
    repl = Repl(console=console, session=session)

    # User accidentally copy-pastes 'prot > load data.csv' or 'prot > inspect'
    repl.handle(f"prot > load {sample_csv}")
    assert session.current_dataset is not None
    assert session.current_dataset.rows == 3

    repl.handle("prot > inspect")
    output = console.export_text()
    assert "Dataset Overview" in output


def test_repl_tool_commands(sample_csv: Path):
    console = Console(record=True)
    session = SessionState()
    repl = Repl(console=console, session=session)

    repl.handle(f"load {sample_csv}")
    repl.handle("missing")
    repl.handle("duplicates")
    repl.handle("dtypes")
    repl.handle("stats")
    repl.handle("correlation")

    out = console.export_text()
    assert "Missing Values" in out
    assert "Duplicates Analysis" in out
    assert "Data Types Categorization" in out
    assert "Descriptive Statistics" in out
    assert "Top Feature Correlations" in out


def test_repl_load_train_and_test_commands(sample_csv: Path):
    console = Console(record=True)
    session = SessionState()
    repl = Repl(console=console, session=session)

    repl.handle(f"load train {sample_csv}")
    assert session.current_dataset is not None
    assert session.current_dataset.rows == 3

    repl.handle(f"load test {sample_csv}")
    assert session.test_dataset is not None
    assert session.test_dataset.rows == 3

    out = console.export_text()
    assert "✓ Dataset loaded" in out
    assert "✓ Test Dataset loaded" in out


def test_repl_autocompletion_and_history_setup():
    from proton_agent.cli.repl import BUILTIN_COMMANDS
    assert "analyze dataset" in BUILTIN_COMMANDS

    console = Console(record=True)
    session = SessionState()
    repl = Repl(console=console, session=session)

    if repl._completer is not None:
        assert "analyze dataset" in repl._completer.words
