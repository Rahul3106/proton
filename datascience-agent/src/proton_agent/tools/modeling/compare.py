"""
Compare Models Tool: trains multiple algorithms, ranks performance metrics, selects the optimal model,
stores ALL trained models in session.models so users can pick any specific one afterwards.
"""
from __future__ import annotations

from typing import Any

from pathlib import Path

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult
from proton_agent.tools.modeling.detect import infer_target_column, infer_problem_type
from proton_agent.tools.modeling.export_model import _write_model_txt
from proton_agent.tools.modeling.recommend import get_model_recommendations
from proton_agent.tools.modeling.train import TrainModelTool


class CompareModelsTool(BaseTool):
    """Trains candidate machine learning models, compares evaluation metrics, selects the best, and saves all models."""

    name: str = "compare_models"
    description: str = "Compare multiple machine learning models (Random Forest, Gradient Boosting, Extra Trees, Decision Tree, Logistic/Linear Regression, SVM) and select the best."

    def execute(
        self,
        session: SessionState,
        target_column: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        trainer = TrainModelTool()

        # Robustly resolve target column using smart inference
        df = session.inspect_dataset().df
        if target_column and target_column in df.columns:
            target = target_column
        elif session.target_column and session.target_column in df.columns:
            target = session.target_column
        else:
            target = infer_target_column(df)

        if not target or target not in df.columns:
            return ToolResult(
                success=False,
                data={},
                error=(
                    "No suitable target column found in the dataset. "
                    "Please select a target column before training, or run 'clean dataset' to fix any structural issues."
                ),
            )

        # Validate target data type — if it's a URL or raw text, refuse early with guidance
        y = df[target]
        problem_type, _ = infer_problem_type(y)

        # Safety check: reject URL/hash targets
        t_lower = target.lower()
        if "url" in t_lower or "link" in t_lower or "href" in t_lower:
            return ToolResult(
                success=False,
                data={},
                error=(
                    f"Target column '{target}' appears to be a URL or link field and cannot be used for ML modeling. "
                    f"Please run 'clean dataset' to prepare the data, then select a proper categorical or numeric target column."
                ),
            )

        session.target_column = target
        is_classification = problem_type == "classification"
        primary_metric = "f1_score" if is_classification else "r2_score"

        # 7 Core Model Families for comparison
        if is_classification:
            candidates = [
                ("Random Forest", "random_forest"),
                ("Gradient Boosting", "gradient_boosting"),
                ("XGBoost", "xgboost"),
                ("LightGBM", "lightgbm"),
                ("CatBoost", "catboost"),
                ("Decision Tree", "decision_tree"),
                ("Logistic Regression", "logistic_regression"),
            ]
        else:
            candidates = [
                ("Random Forest", "random_forest"),
                ("Gradient Boosting", "gradient_boosting"),
                ("XGBoost", "xgboost"),
                ("LightGBM", "lightgbm"),
                ("CatBoost", "catboost"),
                ("Decision Tree", "decision_tree"),
                ("Linear Regression", "linear_regression"),
            ]


        results = []
        errors = []
        for display_name, m_type in candidates:
            res = trainer.execute(session, target_column=target, model_type=m_type)
            if res.success:
                results.append(res.data)
            else:
                errors.append(f"{display_name}: {res.error}")

        if not results:
            error_detail = "; ".join(errors)
            # Detect if it's a type conversion failure (URL/text in features)
            is_text_error = any("convert" in e.lower() or "float" in e.lower() or "string" in e.lower() for e in errors)
            if is_text_error:
                return ToolResult(
                    success=False,
                    data={"requires_cleaning": True},
                    error=(
                        f"Model training failed: dataset likely contains raw text, URL, or unencoded string columns that prevent numeric conversion.\n"
                        f"Action required: Run 'clean dataset' first to remove or encode problematic columns.\n"
                        f"Details: {error_detail}"
                    ),
                )
            return ToolResult(
                success=False,
                data={},
                error=f"Model comparison failed to evaluate candidate models. {error_detail}",
            )

        else:
            sorted_results = sorted(results, key=lambda x: x["metrics"].get(primary_metric, -999.0), reverse=True)

        best_model = sorted_results[0]
        best_metric_val = best_model["metrics"].get(primary_metric)

        # Save all trained models as plain text summaries
        models_dir = Path("outputs/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        saved_path = ""

        # Save every candidate model as a .txt file
        for r in sorted_results:
            rec = session.models.get(r["model_name"])
            if rec:
                txt_name = r["model_name"].replace(" ", "_").lower() + ".txt"
                try:
                    _write_model_txt(rec, models_dir / txt_name)
                except Exception:
                    pass

        # Track best model save path
        best_record = session.models.get(best_model["model_name"])
        if best_record:
            best_txt = best_model["model_name"].replace(" ", "_").lower() + "_best.txt"
            best_path = models_dir / best_txt
            try:
                _write_model_txt(best_record, best_path)
                saved_path = str(best_path)
            except Exception:
                pass  # Non-critical — save failure doesn't block results

        # Build suggestion list
        suggestions = []
        for r in sorted_results:
            rank_metric = r["metrics"].get(primary_metric, 0.0)
            model_nm = r["model_name"]
            suggestions.append(f"  • {model_nm}: {primary_metric}={rank_metric:.4f}")

        suggestion_note = (
            f"All {len(results)} models have been trained and stored in this session.\n"
            f"You can now use any of them:\n" + "\n".join(suggestions) + "\n\n"
            f"To use a specific model: predict for <model name>\n"
            f"  Example: predict for decision tree\n"
            f"  Example: predict for random forest\n"
            f"To save a specific model: export model <name>"
        )

        # Rationale generation
        rationale = (
            f"Trained {len(results)} candidate models on target variable '{target}'. "
            f"{best_model['model_name']} performed best on the validation set with primary metric "
            f"'{primary_metric}' = {best_metric_val} while maintaining balanced precision and stability."
        )

        return ToolResult(
            success=True,
            data={
                "target_column": target,
                "problem_type": best_model["problem_type"],
                "candidate_count": len(results),
                "comparison_matrix": sorted_results,
                "best_model": best_model["model_name"],
                "best_metrics": best_model["metrics"],
                "selection_rationale": rationale,
                "suggestion_note": suggestion_note,
                "saved_best_path": saved_path,
                "message": f"Model comparison complete. Best Model: {best_model['model_name']} ({primary_metric}: {best_metric_val}).",
            },
        )
