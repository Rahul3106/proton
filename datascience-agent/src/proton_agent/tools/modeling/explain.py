"""
ExplainModelTool: generates confusion matrix, ROC curve, classification report, and optional SHAP values.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class ExplainModelTool(BaseTool):
    """Explains model predictions with confusion matrix, ROC/AUC, per-class metrics, and optional SHAP values."""

    name: str = "explain_model"
    description: str = "Explain model predictions: confusion matrix, ROC curve, per-class metrics, and SHAP values."

    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        if not session.current_model:
            return ToolResult(
                success=False,
                data={},
                error="No active trained model. Train a model first.",
            )

        model_record = session.current_model
        model = model_record["model"]
        feature_names = model_record["feature_names"]
        target_col = model_record["target_column"]
        problem_type = model_record.get("problem_type", "classification")

        try:
            dataset = session.inspect_dataset()
        except Exception as err:
            return ToolResult(success=False, data={}, error=str(err))

        df = dataset.df.copy()
        X_raw = df.drop(columns=[target_col]) if target_col in df.columns else df.copy()
        y_true_series = df[target_col] if target_col in df.columns else None

        # Preprocessing
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

        result_data: dict[str, Any] = {
            "model_name": model_record["name"],
            "problem_type": problem_type,
        }

        plots_dir = Path("outputs/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        if problem_type == "classification" and y_true_series is not None:
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                roc_auc_score,
                roc_curve,
            )

            y_true = y_true_series.astype(str)
            y_pred = [str(p) for p in preds]
            labels = sorted(y_true.unique().tolist())

            # Confusion matrix as nested dict
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_dict = {
                "labels": labels,
                "matrix": cm.tolist(),
            }
            result_data["confusion_matrix"] = cm_dict

            # Per-class report
            report_str = classification_report(y_true, y_pred, zero_division=0)
            result_data["classification_report"] = report_str

            # Save confusion matrix plot
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
                im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                plt.colorbar(im, ax=ax)
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels)
                thresh = cm.max() / 2.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"Confusion Matrix — {model_record['name']}")
                plt.tight_layout()
                cm_path = plots_dir / "confusion_matrix.png"
                plt.savefig(cm_path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                result_data["confusion_matrix_plot"] = str(cm_path)
            except Exception as plot_err:
                result_data["confusion_matrix_plot_error"] = str(plot_err)

            # ROC AUC (binary or macro)
            try:
                if len(labels) == 2:
                    proba = model.predict_proba(X_aligned)[:, 1]
                    fpr, tpr, _ = roc_curve(y_true, proba, pos_label=labels[1])
                    auc = float(roc_auc_score(y_true, proba))
                    result_data["roc_auc"] = round(auc, 4)

                    fig2, ax2 = plt.subplots(figsize=(7, 5))
                    ax2.plot(fpr, tpr, color="darkcyan", lw=2, label=f"AUC = {auc:.3f}")
                    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--")
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    ax2.set_title(f"ROC Curve — {model_record['name']}")
                    ax2.legend()
                    roc_path = plots_dir / "roc_curve.png"
                    plt.savefig(roc_path, dpi=120, bbox_inches="tight")
                    plt.close(fig2)
                    result_data["roc_curve_plot"] = str(roc_path)
                else:
                    # Multiclass: macro OVR
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_aligned)
                        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                                  labels=labels))
                        result_data["roc_auc"] = round(auc, 4)
            except Exception as roc_err:
                result_data["roc_auc_error"] = str(roc_err)

        elif problem_type == "regression" and y_true_series is not None:
            import numpy as _np

            y_true_vals = y_true_series.dropna().astype(float).values
            y_pred_vals = _np.array(preds[: len(y_true_vals)], dtype=float)
            residuals = y_true_vals - y_pred_vals

            result_data["residuals_summary"] = {
                "mean_residual": round(float(residuals.mean()), 4),
                "std_residual": round(float(residuals.std()), 4),
                "max_error": round(float(_np.abs(residuals).max()), 4),
            }

            # ── Residual histogram ───────────────────────────────────────────
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
                ax.axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="Zero residual")
                ax.set_xlabel("Residual (Actual − Predicted)")
                ax.set_ylabel("Count")
                ax.set_title(f"Residual Distribution — {model_record['name']}")
                ax.legend()
                plt.tight_layout()
                res_path = plots_dir / "residual_plot.png"
                plt.savefig(res_path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                result_data["residual_plot"] = str(res_path)
            except Exception as res_plot_err:
                result_data["residual_plot_error"] = str(res_plot_err)

            # ── Predicted-vs-actual scatter ──────────────────────────────────
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig2, ax2 = plt.subplots(figsize=(7, 7))
                ax2.scatter(y_pred_vals, y_true_vals, alpha=0.5, color="darkcyan", s=20, label="Samples")
                lo = min(y_pred_vals.min(), y_true_vals.min())
                hi = max(y_pred_vals.max(), y_true_vals.max())
                ax2.plot([lo, hi], [lo, hi], color="crimson", linestyle="--",
                         linewidth=1.5, label="Perfect prediction")
                ax2.set_xlabel("Predicted")
                ax2.set_ylabel("Actual")
                ax2.set_title(f"Predicted vs Actual — {model_record['name']}")
                ax2.legend()
                plt.tight_layout()
                pva_path = plots_dir / "predicted_vs_actual.png"
                plt.savefig(pva_path, dpi=120, bbox_inches="tight")
                plt.close(fig2)
                result_data["predicted_vs_actual_plot"] = str(pva_path)
            except Exception as pva_err:
                result_data["predicted_vs_actual_plot_error"] = str(pva_err)


        # Optional SHAP
        shap_values_summary: dict[str, Any] = {}
        try:
            import shap  # type: ignore

            sample = X_aligned.head(min(100, len(X_aligned)))
            explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.Explainer(model, sample)
            shap_vals = explainer(sample)
            mean_abs = np.abs(shap_vals.values).mean(axis=0)
            if mean_abs.ndim > 1:
                mean_abs = mean_abs.mean(axis=1)
            top = sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)[:10]
            shap_values_summary = {k: round(v, 4) for k, v in top}
        except ImportError:
            shap_values_summary = {"note": "Install 'shap' package to enable SHAP value explanations."}
        except Exception as shap_err:
            shap_values_summary = {"error": str(shap_err)}

        result_data["shap_feature_importance"] = shap_values_summary
        result_data["message"] = f"✓ Explanation complete for '{model_record['name']}'."

        return ToolResult(success=True, data=result_data)
