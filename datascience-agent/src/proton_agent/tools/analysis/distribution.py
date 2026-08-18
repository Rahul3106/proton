"""
Analyze Distribution Tool: numerical distribution profiling and categorical frequency analysis.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class AnalyzeDistributionTool(BaseTool):
    """Profiles numerical distributions (skewness, kurtosis, modality) and categorical frequency distributions."""

    name: str = "analyze_distribution"
    description: str = "Analyze distribution shapes for numerical features and value frequencies for categorical columns."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "string", "boolean"]).columns.tolist()

        num_dist = {}
        for col in num_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            skew_val = float(series.skew())
            kurt_val = float(series.kurtosis())

            if abs(skew_val) < 0.5:
                skew_desc = "Approximately Symmetric"
            elif skew_val >= 0.5:
                skew_desc = "Positively Skewed (Right-tailed)"
            else:
                skew_desc = "Negatively Skewed (Left-tailed)"

            num_dist[col] = {
                "count": len(series),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "skewness": round(skew_val, 4),
                "kurtosis": round(kurt_val, 4),
                "shape": skew_desc,
            }

        cat_dist = {}
        for col in cat_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            val_counts = series.value_counts()
            total = len(series)
            top_5 = [
                {
                    "category": str(k),
                    "count": int(v),
                    "percentage": round((v / total) * 100, 2),
                }
                for k, v in val_counts.head(5).items()
            ]

            cat_dist[col] = {
                "unique_values": int(series.nunique()),
                "total_non_null": total,
                "top_categories": top_5,
            }

        return ToolResult(
            success=True,
            data={
                "numerical_distributions": num_dist,
                "categorical_distributions": cat_dist,
            },
        )
