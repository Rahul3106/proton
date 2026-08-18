"""
The interactive `proton start` session loop.

Phase 5 scope: integrating Agent core for natural language understanding,
intent identification, tool selection, and tool execution.
"""
from __future__ import annotations

from collections.abc import Callable

from rich.console import Console

from proton_agent.agent import Agent
from proton_agent.config.settings import get_settings
from proton_agent.context import (
    DatasetNotFoundError,
    InvalidDatasetError,
    NoDatasetLoadedError,
    SessionState,
)
from proton_agent.tools import default_registry
from proton_agent.ui.console import show_banner
from proton_agent.ui.panels import (
    render_agent_response,
    render_clean_preview,
    render_config_panel,
    render_correlation,
    render_data_types,
    render_dataset_inspection,
    render_dataset_loaded,
    render_duplicates,
    render_error,
    render_export_result,
    render_help,
    render_missing_values,
    render_statistics,
    render_test_dataset_loaded,
)


BUILTIN_COMMANDS = [
    "load train ",
    "load test ",
    "load dataset ",
    "load model ",
    "analyze dataset",
    "audit data quality",
    "detect missing values",
    "detect duplicates",
    "inspect data types",
    "calculate correlation",
    "calculate statistics",
    "describe dataset",
    "clean dataset",
    "impute missing values",
    "encode features",
    "scale features",
    "train model",
    "tune model",
    "compare models",
    "predict",
    "explain model",
    "export model",
    "export predictions",
    "plot distribution of ",
    "plot correlation matrix",
    "plot feature importance",
    "generate report",
    "inspect dataset",
    "config show",
    "clear",
    "help",
    "exit",
]


class Repl:
    """Owns the session loop and dispatches user commands to Agent & UI helpers."""

    def __init__(
        self,
        console: Console,
        session: SessionState | None = None,
        agent: Agent | None = None,
    ) -> None:
        self.console = console
        self.session = session if session is not None else SessionState()
        self.agent = agent if agent is not None else Agent(registry=default_registry)
        self.running = True
        self._settings = get_settings()

        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import InMemoryHistory

            self._completer = WordCompleter(BUILTIN_COMMANDS, ignore_case=True, sentence=True)
            self._history = InMemoryHistory()
            self._prompt_session = PromptSession(
                completer=self._completer,
                history=self._history,
            )
        except Exception:
            self._prompt_session = None

    def run(self, get_input: Callable[[], str] | None = None) -> None:
        get_input = get_input or self._default_input
        show_banner(self.console, self.session)

        while self.running:
            try:
                raw = get_input()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                self._exit("Goodbye.")
                break
            self.handle(raw.strip())

    def handle(self, text: str) -> None:
        if not text:
            return

        # Strip prompt prefixes if accidentally pasted or typed
        for prefix in ("▲ proton ❯", "▲ proton >", "proton ❯", "proton >", "proton>", "prot ❯", "prot >", "prot>", "ds >", "ds>"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break

        if not text:
            return

        lower_text = text.lower()

        if lower_text in ("exit", "quit"):
            self._exit("Goodbye.")
        elif lower_text == "help":
            self._show_help()
        elif lower_text == "clear":
            self.console.clear()
        elif lower_text in ("config", "config show"):
            self._handle_config_show()
        elif lower_text.startswith("load model") or lower_text.startswith("load model "):
            self._handle_load_model(text)
        elif lower_text.startswith("load ") or lower_text.startswith("load dataset ") or lower_text.startswith("load data "):
            self._handle_load(text)
        elif lower_text.startswith("export model"):
            self._handle_export_model(text)
        elif lower_text.startswith("export predictions") or lower_text.startswith("save predictions"):
            self._handle_export_predictions(text)
        elif lower_text in ("inspect", "inspect dataset"):
            self._handle_inspect()
        elif lower_text in ("clean dataset", "clean data", "clean"):
            self._handle_clean()
        else:
            # Delegate natural language queries to Agent core
            response = self.agent.process_request(text, self.session)
            render_agent_response(self.console, response)

    def _handle_clean(self) -> None:
        """Preview changes first, then confirm before applying."""
        tool = default_registry.get("clean_dataset")
        if not tool:
            render_error(self.console, "Tool Error", "clean_dataset tool not found.")
            return

        # Dry-run preview
        preview_result = tool.execute(self.session, dry_run=True)
        if not preview_result.success:
            render_error(self.console, "Clean Error", preview_result.error or "Unknown error.")
            return

        render_clean_preview(self.console, preview_result.data)

        changes = preview_result.data.get("pending_changes", [])
        if not changes:
            return  # Already clean, nothing to apply

        # Confirm prompt
        try:
            answer = self.console.input("[bold yellow]Apply these changes? \\[y/N]: [/bold yellow]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.console.print("[dim]Cleaning cancelled.[/dim]")
            return

        if answer in ("y", "yes"):
            result = tool.execute(self.session, dry_run=False)
            if result.success:
                from proton_agent.ui.panels import render_clean_result
                render_clean_result(self.console, result.data)
            else:
                render_error(self.console, "Clean Error", result.error or "Unknown error.")
        else:
            self.console.print("[dim]Cleaning cancelled. Dataset unchanged.[/dim]")

    def _handle_export_model(self, text: str) -> None:
        lower = text.lower()
        # Extract optional name: 'export model mymodel' or 'export model'
        name = ""
        if lower.startswith("export model "):
            name = text[13:].strip()
        tool = default_registry.get("export_model")
        if not tool:
            render_error(self.console, "Tool Error", "export_model tool not found.")
            return
        result = tool.execute(self.session, model_name=name)
        if result.success:
            render_export_result(self.console, result.data)
        else:
            render_error(self.console, "Export Model Error", result.error or "Unknown error.")

    def _handle_load_model(self, text: str) -> None:
        lower = text.lower()
        filepath = ""
        if lower.startswith("load model "):
            filepath = text[11:].strip().strip("'\"")
        tool = default_registry.get("load_model")
        if not tool:
            render_error(self.console, "Tool Error", "load_model tool not found.")
            return
        result = tool.execute(self.session, filepath=filepath)
        if result.success:
            render_export_result(self.console, result.data)
        else:
            render_error(self.console, "Load Model Error", result.error or "Unknown error.")

    def _handle_export_predictions(self, text: str) -> None:
        lower = text.lower()
        filename = ""
        for prefix in ("export predictions ", "save predictions "):
            if lower.startswith(prefix):
                filename = text[len(prefix):].strip().strip("'\"")
                break
        tool = default_registry.get("export_predictions")
        if not tool:
            render_error(self.console, "Tool Error", "export_predictions tool not found.")
            return
        result = tool.execute(self.session, filename=filename)
        if result.success:
            render_export_result(self.console, result.data)
        else:
            render_error(self.console, "Export Predictions Error", result.error or "Unknown error.")

    def _handle_config_show(self) -> None:
        render_config_panel(self.console, self.session, self._settings)

    def _handle_load(self, text: str) -> None:
        lower = text.lower()
        is_test_load = False

        if lower.startswith("load test dataset "):
            raw_target = text[18:].strip()
            is_test_load = True
        elif lower.startswith("load test data "):
            raw_target = text[15:].strip()
            is_test_load = True
        elif lower.startswith("load testing "):
            raw_target = text[13:].strip()
            is_test_load = True
        elif lower.startswith("load test "):
            raw_target = text[10:].strip()
            is_test_load = True
        elif lower.startswith("load train dataset "):
            raw_target = text[19:].strip()
        elif lower.startswith("load train data "):
            raw_target = text[16:].strip()
        elif lower.startswith("load training "):
            raw_target = text[14:].strip()
        elif lower.startswith("load train "):
            raw_target = text[11:].strip()
        elif lower.startswith("load dataset "):
            raw_target = text[13:].strip()
        elif lower.startswith("load data "):
            raw_target = text[10:].strip()
        elif lower.startswith("load "):
            raw_target = text[5:].strip()
        else:
            raw_target = text

        if raw_target.lower().startswith("dataset "):
            raw_target = raw_target[8:].strip()

        filepath = raw_target.strip("'\"")
        if not filepath:
            render_error(
                self.console,
                "Missing file path",
                "Please specify a file path to load.\nExample: load train data.csv  or  load test test_data.csv",
            )
            return

        try:
            if is_test_load:
                dataset = self.session.load_test_dataset(filepath)
                render_test_dataset_loaded(self.console, dataset)
            else:
                dataset = self.session.load_dataset(filepath)
                render_dataset_loaded(self.console, dataset)
        except Exception as err:
            render_error(
                self.console,
                "Dataset Load Error",
                str(err),
            )

    def _handle_inspect(self) -> None:
        tool = default_registry.get("inspect_dataset")
        if tool:
            result = tool.execute(self.session)
            if result.success:
                render_dataset_inspection(self.console, self.session.current_dataset)
            else:
                render_error(
                    self.console,
                    "No dataset loaded",
                    "No dataset is currently loaded in this session.\nUse 'load <path/to/file.csv>' to load a dataset first.",
                )

    def _exit(self, message: str) -> None:
        self.running = False
        self.session.clear()
        self.console.print(message)

    def _show_help(self) -> None:
        render_help(self.console, registry=self.agent.registry)

    def _default_input(self) -> str:
        if self._prompt_session:
            try:
                return self._prompt_session.prompt("prot > ")
            except Exception:
                pass
        return self.console.input("[bold bright_cyan]prot > [/bold bright_cyan]")
