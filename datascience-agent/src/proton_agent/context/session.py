"""
SessionState: maintains in-memory dataset and session context strictly for the active execution turn.
Session state lives in memory and resets cleanly upon exiting proton.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from proton_agent.context.dataset import (
    DatasetContext,
    DatasetNotFoundError,
    InvalidDatasetError,
    NoDatasetLoadedError,
)
from proton_agent.context.experiment import ExperimentTracker


@dataclass
class SessionState:
    """Holds active dataset context, transformation history, models, experiments, and command history in memory for the active proton session."""

    current_dataset: DatasetContext | None = None
    original_dataset: DatasetContext | None = None
    test_dataset: DatasetContext | None = None
    history: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    target_column: str | None = None
    models: dict[str, Any] = field(default_factory=dict)
    current_model: Any = None
    experiments: ExperimentTracker = field(default_factory=ExperimentTracker)

    def _read_file(self, filepath: str | Path) -> DatasetContext:
        """Internal helper to resolve and read dataset files into DatasetContext."""
        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            raise DatasetNotFoundError(f"File not found: '{filepath}'")
        if not path.is_file():
            raise DatasetNotFoundError(f"Path is not a regular file: '{filepath}'")

        ext = path.suffix.lower()
        try:
            if ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
                df = pd.read_excel(path)
            elif ext == ".parquet":
                df = pd.read_parquet(path)
            elif ext == ".json":
                df = pd.read_json(path)
            elif ext in (".tsv", ".tab"):
                df = pd.read_csv(path, sep="\t")
            else:
                try:
                    df = pd.read_csv(path)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(path, encoding="latin-1")
                    except Exception:
                        df = pd.read_csv(path, encoding="cp1252")
        except Exception as err:
            raise InvalidDatasetError(f"Could not load dataset from '{filepath}': {err}") from err

        return DatasetContext(filepath=path, df=df)

    def load_dataset(self, filepath: str | Path) -> DatasetContext:
        """Load a CSV/Excel dataset into the current session state as the active training dataset."""
        dataset = self._read_file(filepath)
        self.current_dataset = dataset
        self.original_dataset = DatasetContext(filepath=dataset.filepath, df=dataset.df.copy())
        self.history.append(f"load {filepath}")
        self.transformations.clear()
        self.target_column = None
        self.models.clear()
        self.current_model = None
        return dataset

    def revert_dataset(self) -> DatasetContext:
        """Reverts the active dataset to its initial unmodified state upon loading."""
        if self.original_dataset is None:
            if self.current_dataset is not None:
                return self.current_dataset
            raise NoDatasetLoadedError("No dataset is currently loaded to revert.")

        self.current_dataset = DatasetContext(
            filepath=self.original_dataset.filepath,
            df=self.original_dataset.df.copy(),
        )
        self.transformations.clear()
        self.transformations.append("Reverted to original unmodified dataset.")
        return self.current_dataset

    def load_test_dataset(self, filepath: str | Path) -> DatasetContext:
        """Load a test dataset specifically for model predictions and evaluation."""
        dataset = self._read_file(filepath)
        self.test_dataset = dataset
        self.history.append(f"load test {filepath}")
        return dataset

    def inspect_dataset(self) -> DatasetContext:
        """Retrieve the currently loaded dataset context.

        Raises:
            NoDatasetLoadedError: If no dataset is currently loaded.
        """
        if self.current_dataset is None:
            raise NoDatasetLoadedError("No dataset is currently loaded in this session.")
        return self.current_dataset

    def record_transformation(self, description: str) -> None:
        """Records a data transformation step applied to the active session dataset."""
        self.transformations.append(description)

    def clear(self) -> None:
        """Reset the active session state."""
        self.current_dataset = None
        self.original_dataset = None
        self.test_dataset = None
        self.history.clear()
        self.transformations.clear()
        self.target_column = None
        self.models.clear()
        self.current_model = None



