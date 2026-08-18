"""
DescribeDatasetTool: tool for generating pandas summary statistics (df.describe()).
"""
from __future__ import annotations

from typing import Any
import pandas as pd

from proton_agent.context.dataset import NoDatasetLoadedError
from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class DescribeDatasetTool(BaseTool):
    name = "describe_dataset"
    description = "Generate descriptive statistics for numerical and categorical columns."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
            df = dataset.df

            num_df = df.select_dtypes(include=["number"])
            cat_df = df.select_dtypes(include=["object", "string", "category"])

            num_summary = num_df.describe().to_dict() if not num_df.empty else {}
            cat_summary = cat_df.describe().to_dict() if not cat_df.empty else {}

            return ToolResult(
                success=True,
                data={
                    "numerical_summary": num_summary,
                    "categorical_summary": cat_summary,
                    "numerical_columns": list(num_df.columns),
                    "categorical_columns": list(cat_df.columns),
                },
            )
        except NoDatasetLoadedError as err:
            return ToolResult(success=False, error=str(err))
