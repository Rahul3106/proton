"""
Handle Missing Values Tool: custom missing value imputation.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class HandleMissingValuesTool(BaseTool):
    """Fills or drops missing values in active dataset using specified strategy (mean, median, mode, drop)."""

    name: str = "handle_missing_values"
    description: str = "Impute or drop missing values using strategy (mean, median, mode, or drop)."

    def execute(self, session: SessionState, strategy: str = "auto", **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df.copy()
        total_missing = int(df.isnull().sum().sum())

        if total_missing == 0:
            return ToolResult(
                success=True,
                data={
                    "total_missing_before": 0,
                    "total_missing_after": 0,
                    "message": "No missing values found in active dataset.",
                },
            )

        strategy = strategy.lower()
        if strategy == "drop":
            df = df.dropna().reset_index(drop=True)
            desc = f"Dropped rows with missing values (remaining: {len(df):,} rows)"
        else:
            # Auto / Impute strategy
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category", "string", "boolean"]).columns.tolist()

            for col in num_cols:
                if df[col].isnull().sum() > 0:
                    val = df[col].mean() if strategy == "mean" else df[col].median()
                    df[col] = df[col].fillna(val)

            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    mode_res = df[col].mode()
                    val = mode_res.iloc[0] if not mode_res.empty else "Missing"
                    df[col] = df[col].fillna(val)

            desc = f"Imputed missing values across numerical & categorical features (strategy: {strategy})"

        dataset.df = df
        session.record_transformation(desc)

        return ToolResult(
            success=True,
            data={
                "total_missing_before": total_missing,
                "total_missing_after": int(df.isnull().sum().sum()),
                "strategy": strategy,
                "transformation": desc,
            },
        )
