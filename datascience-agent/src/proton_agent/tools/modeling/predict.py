"""
Predict Tool: generates predictions using the active trained machine learning model.
Supports selecting a specific model by name from the session's model registry.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


# Keyword aliases for model type → model name lookup
_MODEL_NAME_ALIASES: dict[str, list[str]] = {
    "random_forest": ["random forest", "rf", "random_forest"],
    "decision_tree": ["decision tree", "dtree", "dt", "decision_tree"],
    "gradient_boosting": ["gradient boosting", "gbm", "boosting", "gradient_boosting"],
    "xgboost": ["xgboost", "xgb", "xg_boost", "extreme gradient boosting"],
    "lightgbm": ["lightgbm", "lgbm", "light_gbm", "light gbm"],
    "catboost": ["catboost", "cb", "cat_boost", "cat boost"],
    "logistic_regression": ["logistic regression", "logistic", "logistic_regression"],
    "linear_regression": ["linear regression", "linear", "linear_regression"],
}

_MODEL_DISPLAY_NAMES: dict[str, str] = {
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "logistic_regression": "Logistic Regression",
    "linear_regression": "Linear Regression",
}


def _resolve_model_record(session: SessionState, model_name: str | None) -> Any | None:
    """
    Look up a specific model record from session.models by alias or partial name.
    Returns None if not found (caller falls back to current_model).
    """
    if not model_name or not session.models:
        return None

    q = model_name.lower()

    # Direct key match first
    if q in session.models:
        return session.models[q]

    # Alias match → get display name prefix → search session.models keys
    for type_key, aliases in _MODEL_NAME_ALIASES.items():
        if any(alias in q for alias in aliases):
            prefix = _MODEL_DISPLAY_NAMES.get(type_key, "").lower()
            # Find a session.models key that contains this prefix (case-insensitive)
            for stored_name, record in session.models.items():
                if prefix and prefix in stored_name.lower():
                    return record
            break

    # Partial match on stored model name
    for stored_name, record in session.models.items():
        if q in stored_name.lower() or stored_name.lower() in q:
            return record

    return None


class PredictTool(BaseTool):
    """Generates predictions on dataset or sample rows using the currently trained active model."""

    name: str = "predict"
    description: str = "Generate predictions using the currently trained machine learning model."

    def execute(
        self,
        session: SessionState,
        n_samples: int = 5,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        # Resolve which model to use
        model_record = _resolve_model_record(session, model_name)
        if model_record is None:
            model_record = session.current_model

        if not model_record:
            return ToolResult(
                success=False,
                data={},
                error="No active trained model found in session. Train a model first using 'train model' or 'compare models'.",
            )

        # Set resolved model as current for session continuity
        session.current_model = model_record

        model = model_record["model"]
        feature_names = model_record["feature_names"]
        target_col = model_record["target_column"]

        dataset = session.test_dataset or session.inspect_dataset()
        df = dataset.df.copy()

        # Prepare X matching trained feature names
        X_raw = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()

        for col in X_raw.columns:
            if X_raw[col].dtype in ("int64", "float64", "number"):
                X_raw[col] = X_raw[col].fillna(X_raw[col].median() if not X_raw[col].isnull().all() else 0.0)
            else:
                mode_val = X_raw[col].mode()
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Missing"
                X_raw[col] = X_raw[col].fillna(fill_val)

        cat_cols = X_raw.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if cat_cols:
            X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float)
        else:
            X_encoded = X_raw.copy()

        # Align columns to feature_names
        X_aligned = X_encoded.reindex(columns=feature_names, fill_value=0.0)
        samples = X_aligned.head(n_samples)

        preds = model.predict(samples)
        sample_preds = [
            {"row": i + 1, "prediction": str(p)}
            for i, p in enumerate(preds)
        ]

        return ToolResult(
            success=True,
            data={
                "model_name": model_record["name"],
                "target_column": target_col,
                "n_samples": len(samples),
                "predictions": sample_preds,
                "message": f"Generated predictions for top {len(samples)} sample row(s) using {model_record['name']}.",
            },
        )
