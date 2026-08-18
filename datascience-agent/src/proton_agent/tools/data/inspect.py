"""
InspectDatasetTool: tool for retrieving active dataset metadata and data preview.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.dataset import NoDatasetLoadedError
from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class InspectDatasetTool(BaseTool):
    name = "inspect_dataset"
    description = "Retrieve active dataset metadata, column dtypes, missing value counts, and preview."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
            head_df = dataset.head(5)
            head_records = head_df.to_dict(orient="records")

            return ToolResult(
                success=True,
                data={
                    "name": dataset.name,
                    "filepath": str(dataset.filepath),
                    "rows": dataset.rows,
                    "columns_count": dataset.columns_count,
                    "total_missing": dataset.total_missing,
                    "memory_kb": round(dataset.memory_usage_bytes / 1024, 1),
                    "columns": dataset.get_column_info(),
                    "preview": head_records,
                },
            )
        except NoDatasetLoadedError as err:
            return ToolResult(success=False, error=str(err))
