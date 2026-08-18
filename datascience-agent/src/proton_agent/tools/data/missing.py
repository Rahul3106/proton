"""
DetectMissingValuesTool: tool for analyzing missing values across dataset columns.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.dataset import NoDatasetLoadedError
from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class DetectMissingValuesTool(BaseTool):
    name = "detect_missing_values"
    description = "Detect missing values per column and return missingness summary statistics."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
            col_info = dataset.get_column_info()

            missing_details = [col for col in col_info if col["missing"] > 0]
            total_missing = dataset.total_missing
            total_rows = dataset.rows
            overall_missing_pct = (
                (total_missing / (total_rows * dataset.columns_count) * 100)
                if (total_rows * dataset.columns_count) > 0
                else 0.0
            )

            return ToolResult(
                success=True,
                data={
                    "total_missing": total_missing,
                    "overall_missing_pct": round(overall_missing_pct, 2),
                    "columns_with_missing_count": len(missing_details),
                    "missing_by_column": col_info,
                    "columns_with_missing": missing_details,
                },
            )
        except NoDatasetLoadedError as err:
            return ToolResult(success=False, error=str(err))
