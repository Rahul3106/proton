"""
Scale Features Tool: numerical feature standardization & normalization.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class ScaleFeaturesTool(BaseTool):
    """Scales numerical features using standardization (z-score) or min-max normalization."""

    name: str = "scale_features"
    description: str = "Standardize or normalize numerical features."

    def execute(self, session: SessionState, method: str = "standard", **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df.copy()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not num_cols:
            return ToolResult(
                success=True,
                data={"scaled_columns": [], "message": "No numerical columns found for scaling."},
            )

        method = method.lower()
        for col in num_cols:
            series = df[col]
            if method == "minmax":
                min_val, max_val = series.min(), series.max()
                if max_val > min_val:
                    df[col] = (series - min_val) / (max_val - min_val)
            else:
                mean_val, std_val = series.mean(), series.std()
                if std_val > 0:
                    df[col] = (series - mean_val) / std_val

        desc = f"Scaled {len(num_cols)} numerical feature(s) using {method} scaling"
        dataset.df = df
        session.record_transformation(desc)

        return ToolResult(
            success=True,
            data={
                "method": method,
                "scaled_columns": num_cols,
                "transformation": desc,
            },
        )
