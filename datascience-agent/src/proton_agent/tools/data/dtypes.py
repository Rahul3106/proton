"""
InspectDataTypesTool: tool for analyzing column data types and categorization.
"""
from __future__ import annotations

from typing import Any
import pandas as pd

from proton_agent.context.dataset import NoDatasetLoadedError
from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class InspectDataTypesTool(BaseTool):
    name = "inspect_data_types"
    description = "Categorize columns into numerical, categorical, boolean, and datetime data types."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
            df = dataset.df

            numerical_cols = list(df.select_dtypes(include=["number"]).columns)
            categorical_cols = list(df.select_dtypes(include=["object", "string", "category"]).columns)
            boolean_cols = list(df.select_dtypes(include=["bool"]).columns)
            datetime_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

            all_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

            return ToolResult(
                success=True,
                data={
                    "total_columns": dataset.columns_count,
                    "numerical_columns": numerical_cols,
                    "categorical_columns": categorical_cols,
                    "boolean_columns": boolean_cols,
                    "datetime_columns": datetime_cols,
                    "dtypes": all_dtypes,
                },
            )
        except NoDatasetLoadedError as err:
            return ToolResult(success=False, error=str(err))
