"""
Audit Data Quality Tool: calculates data health score (0-100%) and identifies critical quality issues.
"""
from __future__ import annotations

from typing import Any

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class AuditDataQualityTool(BaseTool):
    """Audits data health across missing values, duplicates, zero-variance columns, and high cardinality IDs."""

    name: str = "audit_data_quality"
    description: str = "Audit dataset health, calculate data quality score (0-100%), and highlight quality warnings."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols if total_rows * total_cols > 0 else 1

        # Missing values impact
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2)

        # Duplicate rows impact
        dup_rows = int(df.duplicated().sum())
        dup_pct = round((dup_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

        # Zero variance columns
        zero_var_cols = []
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 1:
                zero_var_cols.append(col)

        # High cardinality ID columns (categorical columns with > 90% unique values)
        high_cardinality_cols = []
        for col in df.select_dtypes(include=["object", "string", "category"]).columns:
            if total_rows > 10 and (df[col].nunique() / total_rows) >= 0.9:
                high_cardinality_cols.append(col)

        # Calculate Quality Score (100 base)
        quality_score = 100.0
        quality_score -= min(30.0, missing_pct * 1.5)
        quality_score -= min(20.0, dup_pct * 2.0)
        quality_score -= len(zero_var_cols) * 10.0
        quality_score -= len(high_cardinality_cols) * 5.0
        quality_score = max(0.0, round(quality_score, 1))

        warnings = []
        if missing_pct > 0:
            warnings.append(f"Missing Data: {total_missing:,} missing cell(s) ({missing_pct}% of total dataset)")
        if dup_rows > 0:
            warnings.append(f"Duplicate Rows: {dup_rows:,} duplicate row(s) detected ({dup_pct}%)")
        if zero_var_cols:
            warnings.append(f"Zero-Variance Columns: {len(zero_var_cols)} constant feature(s) ({', '.join(zero_var_cols)})")
        if high_cardinality_cols:
            warnings.append(f"High-Cardinality ID Features: {', '.join(high_cardinality_cols)}")

        status = "Excellent" if quality_score >= 90 else ("Good" if quality_score >= 75 else ("Fair" if quality_score >= 50 else "Poor"))

        return ToolResult(
            success=True,
            data={
                "quality_score": quality_score,
                "status": status,
                "total_rows": total_rows,
                "total_columns": total_cols,
                "total_missing": total_missing,
                "missing_pct": missing_pct,
                "duplicate_rows": dup_rows,
                "duplicate_pct": dup_pct,
                "zero_variance_columns": zero_var_cols,
                "high_cardinality_columns": high_cardinality_cols,
                "warnings": warnings,
            },
        )
