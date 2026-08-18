"""
Generate Report Tool: compiles EDA findings, data health scores, and ML experiment comparisons into Markdown reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class GenerateReportTool(BaseTool):
    """Generates comprehensive Data Science EDA and Machine Learning reports as plain text files."""

    name: str = "generate_report"
    description: str = "Generate an EDA or Machine Learning summary report and save it as a plain text file."

    def execute(self, session: SessionState, report_type: str = "eda", **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        reports_dir = Path("outputs/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        df = dataset.df
        rep_type = report_type.lower()

        if rep_type == "model" and session.current_model:
            report_path = reports_dir / f"model_report_{dataset.name.replace('.csv', '')}.txt"
            model_info = session.current_model

            lines = [
                f"Machine Learning Report — {dataset.name}",
                "=" * 60,
                "",
                f"Generated at      : active proton session",
                f"Target Column     : {model_info['target_column']}",
                f"Problem Type      : {model_info['problem_type'].title()}",
                "",
                "SELECTED BEST MODEL",
                "-" * 40,
                f"  Model Algorithm : {model_info['name']}",
                f"  Training Samples: {model_info['train_samples']:,}",
                f"  Validation Samples: {model_info['test_samples']:,}",
                "",
                "EVALUATION METRICS",
                "-" * 40,
            ]

            for metric, val in model_info["metrics"].items():
                lines.append(f"  {metric.replace('_', ' ').title():<20}: {val}")

            lines.extend([
                "",
                "TOP FEATURE IMPORTANCES",
                "-" * 40,
            ])

            for feat, imp in model_info.get("feature_importances", {}).items():
                lines.append(f"  {feat:<30}: {imp:.4f}")

            if session.experiments.runs:
                lines.extend([
                    "",
                    "EXPERIMENT LOG HISTORY",
                    "-" * 40,
                    f"  {'Run ID':<10}  {'Timestamp':<22}  {'Model':<30}  {'Target':<20}  Metrics",
                    f"  {'-'*10}  {'-'*22}  {'-'*30}  {'-'*20}  {'-'*30}",
                ])
                for run in session.experiments.runs:
                    metrics_str = ", ".join([f"{k}: {v}" for k, v in list(run.metrics.items())[:2]])
                    lines.append(f"  {run.id:<10}  {str(run.timestamp):<22}  {run.model_name:<30}  {run.target_column:<20}  {metrics_str}")

        else:
            report_path = reports_dir / f"eda_report_{dataset.name.replace('.csv', '')}.txt"

            total_missing = dataset.total_missing
            missing_pct = round((total_missing / (dataset.rows * dataset.columns_count if dataset.rows * dataset.columns_count > 0 else 1)) * 100, 2)
            dup_rows = int(df.duplicated().sum())

            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

            lines = [
                f"Exploratory Data Analysis Report — {dataset.name}",
                "=" * 60,
                "",
                "DATASET OVERVIEW",
                "-" * 40,
                f"  Dimensions        : {dataset.rows:,} rows x {dataset.columns_count:,} columns",
                f"  Memory Usage      : {dataset.memory_usage_bytes / 1024:.1f} KB",
                f"  Numerical Features ({len(num_cols)}): {', '.join(num_cols) if num_cols else 'None'}",
                f"  Categorical Features ({len(cat_cols)}): {', '.join(cat_cols) if cat_cols else 'None'}",
                "",
                "DATA QUALITY & INTEGRITY",
                "-" * 40,
                f"  Total Missing Cells : {total_missing:,} ({missing_pct}%)",
                f"  Duplicate Rows      : {dup_rows:,}",
                "",
                "COLUMN DETAILS",
                "-" * 40,
                f"  {'Column Name':<35}  {'Data Type':<12}  {'Non-Null':>10}  {'Missing':>8}  {'Missing%':>8}",
                f"  {'-'*35}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}",
            ]

            for info in dataset.get_column_info():
                lines.append(
                    f"  {info['name']:<35}  {info['dtype']:<12}  {info['non_null']:>10,}  {info['missing']:>8,}  {info['missing_pct']:>7.1f}%"
                )

            if session.transformations:
                lines.extend([
                    "",
                    "RECORDED DATA TRANSFORMATIONS",
                    "-" * 40,
                ])
                for t in session.transformations:
                    lines.append(f"  - {t}")

        report_content = "\n".join(lines) + "\n"
        report_path.write_text(report_content, encoding="utf-8")

        return ToolResult(
            success=True,
            data={
                "report_type": rep_type,
                "report_path": str(report_path.resolve()),
                "message": f"Report saved as plain text to '{report_path}'.",
            },
        )
