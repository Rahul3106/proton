"""Phase 1 tests: the package installs and `proton` boots. No agent
behavior exists yet, so there is nothing else to test at this phase."""
from typer.testing import CliRunner

from proton_agent import __version__
from proton_agent.cli.app import app

runner = CliRunner()


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "proton" in result.output.lower()


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help_not_error():
    result = runner.invoke(app, [])
    # Click's no_args_is_help exits 2 (its usage-error convention) even
    # though it correctly prints help rather than a traceback - 2 here
    # means "no command given", not "the app crashed".
    assert result.exit_code == 2
    assert "usage" in result.output.lower()
