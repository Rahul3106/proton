"""
LoadDatasetTool: tool for loading datasets into session state.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.dataset import DatasetNotFoundError, InvalidDatasetError
from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class LoadDatasetTool(BaseTool):
    name = "load_dataset"
    description = "Load a CSV or Excel dataset (.csv, .xlsx, .xls, .tsv, .parquet, .json) into session state."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        filepath = kwargs.get("filepath")
        if not filepath:
            return ToolResult(
                success=False,
                error="Missing required argument 'filepath'.",
            )

        try:
            dataset = session.load_dataset(str(filepath))
            return ToolResult(
                success=True,
                data={
                    "name": dataset.name,
                    "filepath": str(dataset.filepath),
                    "rows": dataset.rows,
                    "columns": dataset.columns_count,
                    "columns_list": dataset.columns,
                },
            )
        except (DatasetNotFoundError, InvalidDatasetError) as err:
            return ToolResult(success=False, error=str(err))
