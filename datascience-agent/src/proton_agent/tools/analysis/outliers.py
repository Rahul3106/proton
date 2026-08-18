"""
Detect Outliers Tool: IQR-based outlier identification for numerical columns.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class DetectOutliersTool(BaseTool):
    """Detects statistical outliers in numerical dataset columns using the Interquartile Range (IQR) method."""

    name: str = "detect_outliers"
    description: str = "Detect statistical outliers in numerical columns using Interquartile Range (IQR)."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not num_cols:
            return ToolResult(
                success=True,
                data={
                    "outliers_by_column": {},
                    "total_outliers": 0,
                    "affected_columns": [],
                    "message": "No numerical columns found for outlier detection.",
                },
            )

        outliers_by_column = {}
        total_outliers_count = 0
        affected_cols = []

        for col in num_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25

            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outliers_mask = (series < lower_bound) | (series > upper_bound)
            count = int(outliers_mask.sum())
            total = len(series)
            pct = round((count / total) * 100, 2) if total > 0 else 0.0

            if count > 0:
                affected_cols.append(col)
                total_outliers_count += count

            outliers_by_column[col] = {
                "outlier_count": count,
                "outlier_pct": pct,
                "lower_bound": round(lower_bound, 4),
                "upper_bound": round(upper_bound, 4),
                "q25": round(q25, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr, 4),
            }

        return ToolResult(
            success=True,
            data={
                "outliers_by_column": outliers_by_column,
                "total_outliers": total_outliers_count,
                "affected_columns": affected_cols,
                "total_numerical_columns": len(num_cols),
            },
        )
