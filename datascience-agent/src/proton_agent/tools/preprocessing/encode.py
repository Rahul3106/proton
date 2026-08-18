"""
Encode Features Tool: categorical variable encoding (one-hot / label encoding).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class EncodeFeaturesTool(BaseTool):
    """Encodes categorical columns into numerical representations (one-hot or label encoding)."""

    name: str = "encode_features"
    description: str = "Encode categorical features using one-hot encoding or label encoding."

    def execute(self, session: SessionState, method: str = "onehot", **kwargs: Any) -> ToolResult:
        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df.copy()
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        if not cat_cols:
            return ToolResult(
                success=True,
                data={"encoded_columns": [], "message": "No categorical columns found for encoding."},
            )

        method = method.lower()
        if method == "label":
            for col in cat_cols:
                df[col] = df[col].astype("category").cat.codes
            desc = f"Applied label encoding to {len(cat_cols)} column(s): {', '.join(cat_cols)}"
        else:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
            desc = f"Applied one-hot encoding to {len(cat_cols)} column(s)"

        dataset.df = df
        session.record_transformation(desc)

        return ToolResult(
            success=True,
            data={
                "method": method,
                "original_categorical_columns": cat_cols,
                "new_shape": list(df.shape),
                "transformation": desc,
            },
        )
