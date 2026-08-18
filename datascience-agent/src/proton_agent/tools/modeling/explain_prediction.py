"""
ExplainPredictionTool: generates instance-level prediction explanations
using local SHAP values and feature attribution with plain-English rationales.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class ExplainPredictionTool(BaseTool):
    """Explains a single-row prediction with ranked SHAP/feature contribution values and plain-English summaries."""

    name: str = "explain_prediction"
    description: str = "Explain why a model made a specific prediction on a single instance (row), ranking feature contributions with plain-English rationales."

    def execute(
        self,
        session: SessionState,
        features: Optional[Dict[str, Any]] = None,
        row_index: Optional[int] = None,
        **kwargs: Any,
    ) -> ToolResult:
        if not session.current_model:
            return ToolResult(
                success=False,
                data={},
                error="No active trained model found in session. Train a model first via 'train model' or 'compare models'.",
            )

        model_record = session.current_model
        model = model_record["model"]
        feature_names = model_record.get("feature_names", [])
        target_col = model_record.get("target_column", "Target")
        problem_type = model_record.get("problem_type", "classification")

        # Resolve single row feature values
        raw_features: Dict[str, Any] = {}
        dataset_df = None
        if session.current_dataset and session.current_dataset.df is not None:
            dataset_df = session.current_dataset.df

        if features and isinstance(features, dict) and len(features) > 0:
            raw_features = dict(features)
        elif dataset_df is not None and not dataset_df.empty:
            idx = row_index if (row_index is not None and 0 <= row_index < len(dataset_df)) else 0
            row_series = dataset_df.iloc[idx].copy()
            if target_col in row_series:
                row_series = row_series.drop(labels=[target_col])
            raw_features = row_series.to_dict()
        else:
            return ToolResult(
                success=False,
                data={},
                error="No input feature values provided and no dataset loaded in session to sample from.",
            )

        # Prepare single-row aligned DataFrame
        input_df = pd.DataFrame([raw_features])

        # Categorical dummy encoding matching feature names
        cat_cols = input_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if cat_cols:
            input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True, dtype=float)
        else:
            input_encoded = input_df.copy()

        aligned_df = input_encoded.reindex(columns=feature_names, fill_value=0.0)
        for col in aligned_df.columns:
            aligned_df[col] = aligned_df[col].fillna(0.0)

        # Generate model prediction
        try:
            pred_raw = model.predict(aligned_df)[0]
            prediction_val = str(round(float(pred_raw), 4)) if problem_type == "regression" else str(pred_raw)
        except Exception as err:
            return ToolResult(
                success=False,
                data={},
                error=f"Prediction inference failed for input features: {err}",
            )

        # Probabilities if classification
        probabilities: Optional[Dict[str, float]] = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(aligned_df)[0]
                classes = getattr(model, "classes_", [str(i) for i in range(len(probs))])
                probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
            except Exception:
                pass

        # Calculate dataset baselines (mean / median / mode)
        baselines: Dict[str, Any] = {}
        stds: Dict[str, float] = {}
        if dataset_df is not None:
            for feat in feature_names:
                if feat in dataset_df.columns:
                    if pd.api.types.is_numeric_dtype(dataset_df[feat]):
                        med = float(dataset_df[feat].median()) if not dataset_df[feat].isnull().all() else 0.0
                        std_val = float(dataset_df[feat].std()) if len(dataset_df) > 1 else 1.0
                        baselines[feat] = round(med, 2)
                        stds[feat] = std_val if std_val > 1e-6 else 1.0
                    else:
                        mode_v = dataset_df[feat].mode()
                        baselines[feat] = str(mode_v.iloc[0]) if not mode_v.empty else "N/A"
                        stds[feat] = 1.0
                else:
                    baselines[feat] = 0.0
                    stds[feat] = 1.0
        else:
            for feat in feature_names:
                baselines[feat] = 0.0
                stds[feat] = 1.0

        # Compute SHAP or baseline-deviation attributions
        contributions_map: Dict[str, float] = {}
        method_used = "Deterministic Feature Attribution"
        base_value = 0.0

        shap_computed = False
        try:
            import shap  # type: ignore

            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
                shap_obj = explainer(aligned_df)
                vals = shap_obj.values[0]

                # Multiclass handling
                if vals.ndim > 1:
                    if hasattr(model, "classes_") and str(pred_raw) in [str(c) for c in model.classes_]:
                        cls_idx = list(model.classes_).index(pred_raw)
                        vals = vals[:, cls_idx]
                    else:
                        vals = vals[:, 0]

                for feat, v in zip(feature_names, vals):
                    contributions_map[feat] = float(v)

                base_val_raw = explainer.expected_value
                if isinstance(base_val_raw, (list, np.ndarray)):
                    base_value = float(base_val_raw[0])
                else:
                    base_value = float(base_val_raw)

                method_used = "SHAP (TreeExplainer)"
                shap_computed = True
        except Exception:
            shap_computed = False

        if not shap_computed:
            # Deterministic attribution fallback using deviation from baseline & model weights
            if hasattr(model, "coef_") and model.coef_ is not None:
                coefs = model.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]
                for i, feat in enumerate(feature_names):
                    in_val = float(aligned_df.iloc[0, i])
                    base_v = float(baselines.get(feat, 0.0)) if isinstance(baselines.get(feat), (int, float)) else 0.0
                    c_weight = float(coefs[i]) if i < len(coefs) else 0.0
                    contributions_map[feat] = (in_val - base_v) * c_weight
                method_used = "Linear Coefficient Attribution"
            elif hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
                imps = model.feature_importances_
                for i, feat in enumerate(feature_names):
                    in_val = float(aligned_df.iloc[0, i])
                    base_v = float(baselines.get(feat, 0.0)) if isinstance(baselines.get(feat), (int, float)) else 0.0
                    std_v = stds.get(feat, 1.0)
                    imp = float(imps[i]) if i < len(imps) else 0.0
                    z_score = (in_val - base_v) / std_v
                    contributions_map[feat] = z_score * imp
                method_used = "Feature Importance Weighted Deviation"
            else:
                for i, feat in enumerate(feature_names):
                    in_val = float(aligned_df.iloc[0, i])
                    base_v = float(baselines.get(feat, 0.0)) if isinstance(baselines.get(feat), (int, float)) else 0.0
                    std_v = stds.get(feat, 1.0)
                    contributions_map[feat] = (in_val - base_v) / std_v
                method_used = "Standardized Baseline Deviation"

        # Format structured contribution list
        ranked_features: List[Dict[str, Any]] = []
        for feat in feature_names:
            contrib = contributions_map.get(feat, 0.0)
            in_val = raw_features.get(feat, aligned_df.iloc[0].get(feat, 0.0))
            base_v = baselines.get(feat, "N/A")

            direction = "positive" if contrib > 0 else ("negative" if contrib < 0 else "neutral")

            # Generate natural language rationale
            if isinstance(in_val, (int, float)) and isinstance(base_v, (int, float)):
                diff_desc = "above" if in_val > base_v else ("below" if in_val < base_v else "matching")
                summary = (
                    f"'{feat}' ({in_val}) contributed {contrib:+.3f} toward predicting '{prediction_val}' "
                    f"because it is {diff_desc} the dataset baseline ({base_v})."
                )
            else:
                summary = (
                    f"'{feat}' ('{in_val}') contributed {contrib:+.3f} toward predicting '{prediction_val}' "
                    f"(dataset typical value: '{base_v}')."
                )

            ranked_features.append({
                "feature": feat,
                "value": in_val,
                "baseline": base_v,
                "contribution": round(contrib, 4),
                "direction": direction,
                "summary": summary,
            })

        # Sort by magnitude of contribution descending
        ranked_features.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        # Build human-readable plain text summary
        top_k = ranked_features[:5]
        summary_lines = [
            f"Prediction Explanation for '{model_record['name']}':",
            f"  • Predicted Value : {prediction_val} (Target: {target_col})",
            f"  • Explanation Engine : {method_used}",
            "",
            "Top Feature Contributions (Ranked by Impact):",
        ]
        for i, item in enumerate(top_k, 1):
            sign_symbol = "+" if item["contribution"] >= 0 else ""
            summary_lines.append(
                f"  {i}. {item['feature']:<20} = {str(item['value']):<8} -> Impact: {sign_symbol}{item['contribution']:.4f} ({item['direction']})"
            )
            summary_lines.append(f"     Rationale: {item['summary']}")

        return ToolResult(
            success=True,
            data={
                "model_name": model_record["name"],
                "target_column": target_col,
                "problem_type": problem_type,
                "prediction": prediction_val,
                "probabilities": probabilities,
                "features_input": raw_features,
                "explanation_method": method_used,
                "base_value": round(base_value, 4),
                "top_contributions": ranked_features,
                "plain_text_summary": "\n".join(summary_lines),
                "message": f"Successfully generated prediction explanation for target '{target_col}' ({prediction_val}).",
            },
        )
