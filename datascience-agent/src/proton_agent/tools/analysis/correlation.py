"""
CalculateCorrelationTool: tool for computing Pearson correlation matrix and top feature pairs.
"""
from __future__ import annotations

from typing import Any
import pandas as pd

from proton_agent.context.dataset import NoDatasetLoadedError
from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class CalculateCorrelationTool(BaseTool):
    name = "calculate_correlation"
    description = "Calculate numerical correlation matrix and identify strongest feature correlations."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
            df = dataset.df
            num_df = df.select_dtypes(include=["number"])

            if num_df.shape[1] < 2:
                return ToolResult(
                    success=True,
                    data={
                        "matrix": {},
                        "top_correlations": [],
                        "message": "At least 2 numerical columns are required to calculate correlation matrix.",
                    },
                )

            corr_df = num_df.corr(method=kwargs.get("method", "pearson"))
            corr_matrix = corr_df.round(4).to_dict()

            # Find top pairwise correlations excluding self-correlation
            top_pairs = []
            cols = list(corr_df.columns)
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    col1, col2 = cols[i], cols[j]
                    val = corr_df.loc[col1, col2]
                    if not pd.isna(val):
                        top_pairs.append(
                            {
                                "feature_1": str(col1),
                                "feature_2": str(col2),
                                "correlation": round(float(val), 4),
                                "abs_correlation": round(abs(float(val)), 4),
                            }
                        )

            top_pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)

            return ToolResult(
                success=True,
                data={
                    "columns": [str(c) for c in cols],
                    "matrix": corr_matrix,
                    "top_correlations": top_pairs[:10],
                },
            )
        except NoDatasetLoadedError as err:
            return ToolResult(success=False, error=str(err))
