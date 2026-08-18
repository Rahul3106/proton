"""
CalculateStatisticsTool: tool for computing detailed numerical column statistics.
"""
from __future__ import annotations

from typing import Any
import pandas as pd

from proton_agent.context.dataset import NoDatasetLoadedError
from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class CalculateStatisticsTool(BaseTool):
    name = "calculate_statistics"
    description = "Compute mean, std, median, min, max, variance, IQR, skewness, and kurtosis per numerical column."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
            df = dataset.df
            num_df = df.select_dtypes(include=["number"])

            if num_df.empty:
                return ToolResult(
                    success=True,
                    data={"statistics": {}, "message": "No numerical columns found in dataset."},
                )

            stats_dict = {}
            for col in num_df.columns:
                series = num_df[col].dropna()
                if series.empty:
                    continue

                q25 = float(series.quantile(0.25))
                q75 = float(series.quantile(0.75))
                iqr = q75 - q25

                stats_dict[str(col)] = {
                    "count": int(series.count()),
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                    "variance": round(float(series.var()), 4) if len(series) > 1 else 0.0,
                    "median": round(float(series.median()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                    "q25": round(q25, 4),
                    "q75": round(q75, 4),
                    "iqr": round(iqr, 4),
                    "skew": round(float(series.skew()), 4) if len(series) > 2 else 0.0,
                    "kurtosis": round(float(series.kurtosis()), 4) if len(series) > 3 else 0.0,
                }

            return ToolResult(
                success=True,
                data={
                    "total_numerical_columns": len(num_df.columns),
                    "statistics": stats_dict,
                },
            )
        except NoDatasetLoadedError as err:
            return ToolResult(success=False, error=str(err))
