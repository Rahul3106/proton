"""
ExportPredictionsTool: runs full-dataset predictions and saves results as CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class ExportPredictionsTool(BaseTool):
    """Runs predictions on all rows of the loaded dataset and exports results to a CSV file."""

    name: str = "export_predictions"
    description: str = "Export predictions from the trained model to a CSV file in outputs/."

    def execute(self, session: SessionState, filename: str = "", **kwargs: Any) -> ToolResult:
        if not session.current_model:
            return ToolResult(
                success=False,
                data={},
                error="No active trained model in session. Train a model first.",
            )

        model_record = session.current_model
        model = model_record["model"]
        feature_names = model_record["feature_names"]
        target_col = model_record["target_column"]

        try:
            dataset = session.test_dataset or session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df.copy()

        # Preprocessing: align to trained feature schema
        X_raw = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
        for col in X_raw.columns:
            if X_raw[col].dtype in ("int64", "float64", "number"):
                X_raw[col] = X_raw[col].fillna(X_raw[col].median() if not X_raw[col].isnull().all() else 0.0)
            else:
                mode_val = X_raw[col].mode()
                X_raw[col] = X_raw[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Missing")

        cat_cols = X_raw.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float) if cat_cols else X_raw.copy()
        X_aligned = X_encoded.reindex(columns=feature_names, fill_value=0.0)

        preds = model.predict(X_aligned)
        result_df = df.copy()
        result_df[f"predicted_{target_col}"] = preds

        # Save to CSV
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_name = filename.strip() or f"{dataset.name.replace('.csv','')}_predictions.csv"
        if not save_name.endswith(".csv"):
            save_name += ".csv"
        save_path = out_dir / save_name
        result_df.to_csv(save_path, index=False)

        return ToolResult(
            success=True,
            data={
                "saved_path": str(save_path),
                "rows": len(result_df),
                "prediction_column": f"predicted_{target_col}",
                "model_name": model_record["name"],
                "message": f"✓ Predictions for {len(result_df):,} rows saved to '{save_path}'.",
            },
        )
