"""
Plotting tools: distribution histogram, correlation matrix heatmap, and feature importance bar plots.
Uses non-interactive Agg backend to save images safely to outputs/plots/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


def _ensure_output_dir() -> Path:
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


class PlotDistributionTool(BaseTool):
    """Generates distribution histogram plots for numerical features."""

    name: str = "plot_distribution"
    description: str = "Generate and save a distribution plot (histogram) for a dataset column."

    def execute(self, session: SessionState, column: str | None = None, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not num_cols:
            return ToolResult(
                success=False,
                data={},
                error="No numerical columns available for distribution plotting.",
            )

        # Filter out ID columns for default plotting
        valid_default_cols = [c for c in num_cols if c.lower() not in ("id", "index", "unnamed: 0")] or num_cols

        target_col = None
        user_query = column or kwargs.get("user_request") or kwargs.get("query") or ""

        if user_query:
            q_lower = user_query.lower()
            target_word = user_query
            if "distribution of " in q_lower:
                target_word = user_query[q_lower.index("distribution of ") + 16:].strip().strip("'\"")
            elif "distribution " in q_lower:
                target_word = user_query[q_lower.index("distribution ") + 13:].strip().strip("'\"")

            # Check if target_word matches a real dataset column (case-insensitive)
            matched = next((c for c in df.columns if c.lower() == target_word.lower()), None)
            if matched:
                if matched in num_cols:
                    target_col = matched
                else:
                    return ToolResult(
                        success=False,
                        data={},
                        error=f"Column '{matched}' is categorical. Distribution plotting requires a numerical feature. Available numerical columns: {', '.join(valid_default_cols)}",
                    )
            elif target_word and target_word.lower() not in ["plot_distribution", "plot distribution", "plot", "distribution", "of", "the"]:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Column '{target_word}' not found in dataset '{dataset.name}'. Available columns: {', '.join(df.columns)}",
                )

        if not target_col:
            target_col = valid_default_cols[0]

        series = df[target_col].dropna()

        out_dir = _ensure_output_dir()
        plot_path = out_dir / f"distribution_{target_col}.png"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series, bins=20, color="#00bcd4", edgecolor="#1a1a1a", alpha=0.8)
        ax.set_title(f"Distribution of {target_col}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(target_col, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        return ToolResult(
            success=True,
            data={
                "column": target_col,
                "plot_path": str(plot_path.resolve()),
                "message": f"Distribution plot saved to '{plot_path}'.",
            },
        )


class PlotCorrelationMatrixTool(BaseTool):
    """Generates a correlation matrix heatmap image."""

    name: str = "plot_correlation_matrix"
    description: str = "Generate and save a correlation matrix heatmap image."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if len(num_cols) < 2:
            return ToolResult(
                success=False,
                data={},
                error="At least 2 numerical columns are required to generate a correlation matrix plot.",
            )

        corr_df = df[num_cols].corr()
        out_dir = _ensure_output_dir()
        plot_path = out_dir / "correlation_matrix.png"

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr_df, cmap="Blues")
        fig.colorbar(cax)

        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=45, ha="left")
        ax.set_yticklabels(num_cols)
        ax.set_title("Correlation Matrix Heatmap", fontsize=14, fontweight="bold", pad=20)

        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                val = corr_df.iloc[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val) < 0.7 else "white", fontsize=9)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        return ToolResult(
            success=True,
            data={
                "numerical_columns": num_cols,
                "plot_path": str(plot_path.resolve()),
                "message": f"Correlation matrix plot saved to '{plot_path}'.",
            },
        )


class PlotFeatureImportanceTool(BaseTool):
    """Generates feature importance bar plot for trained model."""

    name: str = "plot_feature_importance"
    description: str = "Generate feature importance bar plot for the trained model."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        if not session.current_model or not session.current_model.get("feature_importances"):
            return ToolResult(
                success=False,
                data={},
                error="No active model with feature importances found in session.",
            )

        model_record = session.current_model
        feat_imps = model_record["feature_importances"]

        features = list(feat_imps.keys())[::-1]
        importances = [feat_imps[f] for f in features]

        out_dir = _ensure_output_dir()
        plot_path = out_dir / "feature_importance.png"

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(features, importances, color="#4caf50", edgecolor="#1a1a1a", alpha=0.85)
        ax.set_title(f"Feature Importance ({model_record['name']})", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        return ToolResult(
            success=True,
            data={
                "model_name": model_record["name"],
                "plot_path": str(plot_path.resolve()),
                "message": f"Feature importance plot saved to '{plot_path}'.",
            },
        )
