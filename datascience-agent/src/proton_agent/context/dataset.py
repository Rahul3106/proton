"""
DatasetContext: encapsulates the loaded DataFrame and its structural metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetError(Exception):
    """Base exception for dataset context operations."""


class DatasetNotFoundError(DatasetError):
    """Raised when the specified dataset file does not exist."""


class InvalidDatasetError(DatasetError):
    """Raised when a dataset file cannot be parsed or is corrupt/empty."""


class NoDatasetLoadedError(DatasetError):
    """Raised when an operation requires a dataset but none is loaded."""


@dataclass
class DatasetContext:
    """Encapsulates a loaded dataset and its computed metadata."""

    filepath: Path
    df: pd.DataFrame
    name: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.filepath, str):
            self.filepath = Path(self.filepath)
        if not self.name:
            self.name = self.filepath.name

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape

    @property
    def rows(self) -> int:
        return self.df.shape[0]

    @property
    def columns_count(self) -> int:
        return self.df.shape[1]

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    @property
    def dtypes(self) -> dict[str, str]:
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    @property
    def missing_counts(self) -> dict[str, int]:
        return {col: int(count) for col, count in self.df.isnull().sum().items()}

    @property
    def total_missing(self) -> int:
        return int(self.df.isnull().sum().sum())

    @property
    def memory_usage_bytes(self) -> int:
        return int(self.df.memory_usage(deep=True).sum())

    def get_column_info(self) -> list[dict[str, Any]]:
        """Returns structured metadata per column."""
        info = []
        for col in self.df.columns:
            dtype_str = str(self.df[col].dtype)
            missing = int(self.df[col].isnull().sum())
            non_null = self.rows - missing
            pct = (missing / self.rows * 100) if self.rows > 0 else 0.0
            info.append(
                {
                    "name": str(col),
                    "dtype": dtype_str,
                    "non_null": non_null,
                    "missing": missing,
                    "missing_pct": pct,
                }
            )
        return info

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.df.head(n)
