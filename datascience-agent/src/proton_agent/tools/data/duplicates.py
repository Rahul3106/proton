"""
DetectDuplicatesTool: tool for detecting duplicate rows in dataset.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.dataset import NoDatasetLoadedError
from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class DetectDuplicatesTool(BaseTool):
    name = "detect_duplicates"
    description = "Detect duplicate rows in the active dataset."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
            dup_count = int(dataset.df.duplicated().sum())
            total_rows = dataset.rows
            dup_pct = (dup_count / total_rows * 100) if total_rows > 0 else 0.0

            return ToolResult(
                success=True,
                data={
                    "total_rows": total_rows,
                    "duplicate_rows": dup_count,
                    "duplicate_pct": round(dup_pct, 2),
                    "has_duplicates": dup_count > 0,
                },
            )
        except NoDatasetLoadedError as err:
            return ToolResult(success=False, error=str(err))
