"""
Planner: plan generation component driven by multi-step & single-step intent logic.
"""
from __future__ import annotations

import re
from typing import List, Optional

from proton_agent.agent.llm import LLMClient
from proton_agent.agent.state import Plan, PlanStep
from proton_agent.tools import ToolRegistry, default_registry


# Keyword sets for model-type detection in user requests.
# Each key must exactly match a model_key accepted by TrainModelTool.execute().
_MODEL_TYPE_KEYWORDS: dict[str, list[str]] = {
    # ── Tree / Ensemble ──────────────────────────────────────────────────────
    "random_forest": ["random forest", "random_forest", "rf"],
    "decision_tree": ["decision tree", "decision_tree", "dtree", "dt"],
    "gradient_boosting": ["gradient boosting", "gradient_boosting", "gbm", "boosting"],
    "xgboost": ["xgboost", "xgb", "xg_boost", "extreme gradient boosting"],
    "lightgbm": ["lightgbm", "lgbm", "light_gbm", "light gbm"],
    "catboost": ["catboost", "cb", "cat_boost", "cat boost"],
    "extra_trees": ["extra trees", "extra_trees", "et", "extratrees"],
    "hist_gradient": ["hist gradient", "hist_gradient", "histgb", "hist gb"],
    "adaboost": ["adaboost", "ada boost", "ada_boost"],

    # ── Linear / Regularised ─────────────────────────────────────────────────
    "logistic_regression": ["logistic regression", "logistic_regression", "logistic"],
    "linear_regression": ["linear regression", "linear_regression"],
    "ridge": ["ridge regression", "ridge_regression", "ridge"],
    "lasso": ["lasso regression", "lasso_regression", "lasso"],
    "bayesian_ridge": ["bayesian ridge", "bayesian_ridge", "bayesian"],
    # ── Instance / Kernel ────────────────────────────────────────────────────
    "knn": ["knn", "k-nearest", "k nearest", "neighbors", "neighbour"],
    "svm": ["svm", "svc", "support vector classifier", "support vector machine"],
    "svr": ["svr", "support vector regressor", "support vector regression"],
    # ── Neural ───────────────────────────────────────────────────────────────
    "mlp": ["mlp", "neural net", "neural network", "multi layer perceptron"],
    # ── Naive Bayes ──────────────────────────────────────────────────────────
    "naive_bayes": ["naive bayes", "naive_bayes", "gaussian nb", "gaussiannb"],
}


def _detect_model_type(q: str) -> Optional[str]:
    """Return model_type key if user names a specific algorithm, else None."""
    for model_key, aliases in _MODEL_TYPE_KEYWORDS.items():
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", q):
                return model_key
    return None



class Planner:
    """Selects tools and constructs execution plans for given user queries."""

    def __init__(self, llm_client: LLMClient, registry: Optional[ToolRegistry] = None) -> None:
        self.llm_client = llm_client
        self.registry = registry or default_registry

    def create_plan(self, user_request: str) -> Plan:
        """Constructs a multi-step plan for high-level goals or a 1-step plan for targeted queries."""
        q = user_request.lower()

        # ── EXPLAIN INSTANCE / ROW PREDICTION ────────────────────────────────
        if any(p in q for p in ["explain prediction", "explain last prediction", "explain row", "why this prediction", "explain single prediction", "why predicted"]):
            return Plan(
                goal=f"Explain Instance Prediction: '{user_request}'",
                steps=[
                    PlanStep(
                        step_number=1,
                        description="Explain instance prediction with local SHAP feature contributions",
                        tool_name="explain_prediction",
                    ),
                ],
            )

        # Meta-questions asking about agent/model behavior ('how you are predicting', 'why did...', etc.) route to Q&A reasoning
        meta_question_phrases = ["how you", "why you", "how are you", "why did", "why is", "how do you predict", "how are you predicting"]
        if any(p in q for p in meta_question_phrases):
            return Plan(goal=user_request, steps=[])


        # ── SPECIFIC MODEL TRAINING ──────────────────────────────────────────
        # If user mentions "train" AND a specific model type → train only that model
        if any(w in q for w in ["train", "build model"]):
            specific_type = _detect_model_type(q)
            if specific_type:
                model_label = specific_type.replace("_", " ").title()
                return Plan(
                    goal=f"Train Specific Model: '{model_label}' — '{user_request}'",
                    steps=[
                        PlanStep(
                            step_number=1,
                            description=f"Train {model_label} on the active dataset",
                            tool_name="train_model",
                            tool_kwargs={"model_type": specific_type},
                        ),
                        PlanStep(
                            step_number=2,
                            description=f"Generate sample predictions using the trained {model_label}",
                            tool_name="predict",
                        ),
                    ],
                )

        # ── COMPARE ALL / TRAIN ALL ──────────────────────────────────────────
        # "train model", "train models", "compare models", etc. → run all candidates
        if any(w in q for w in ["train", "build model", "compare models", "classifier", "regression"]):
            return Plan(
                goal=f"Autonomous Machine Learning Modeling: '{user_request}'",
                steps=[
                    PlanStep(step_number=1, description="Detect target column & problem type (classification vs regression)", tool_name="detect_problem_type"),
                    PlanStep(step_number=2, description="Train & compare candidate models (Random Forest, Tree, Boosting, Linear/Logistic)", tool_name="compare_models"),
                    PlanStep(step_number=3, description="Generate sample predictions using the best model", tool_name="predict"),
                ],
            )

        # ── PREDICT WITH OPTIONAL MODEL SELECTION ───────────────────────────
        # "predict using decision tree" / "predict for random forest" etc.
        if any(w in q for w in ["predict"]):
            specific_type = _detect_model_type(q)
            if specific_type:
                model_label = specific_type.replace("_", " ").title()
                return Plan(
                    goal=f"Predict using {model_label}",
                    steps=[
                        PlanStep(
                            step_number=1,
                            description=f"Generate predictions using {model_label}",
                            tool_name="predict",
                            tool_kwargs={"model_name": specific_type},
                        ),
                    ],
                )

        # Multi-step Autonomous EDA & Dataset Analysis queries
        if any(w in q for w in ["analyze", "analysis", "eda", "explore", "quality", "audit", "overview"]):
            return Plan(
                goal=f"Autonomous Exploratory Data Analysis: '{user_request}'",
                steps=[
                    PlanStep(step_number=1, description="Inspect dataset schema & preview data", tool_name="inspect_dataset"),
                    PlanStep(step_number=2, description="Audit data quality & health score", tool_name="audit_data_quality"),
                    PlanStep(step_number=3, description="Check for missing values", tool_name="detect_missing_values"),
                    PlanStep(step_number=4, description="Detect duplicate rows", tool_name="detect_duplicates"),
                    PlanStep(step_number=5, description="Profile distributions & skewness", tool_name="analyze_distribution"),
                    PlanStep(step_number=6, description="Detect statistical outliers (IQR)", tool_name="detect_outliers"),
                    PlanStep(step_number=7, description="Calculate feature correlations", tool_name="calculate_correlation"),
                ],
            )

        # Single-intent queries mapped via select_tool
        single_tool = self.select_tool(user_request)
        if single_tool:
            tool_obj = self.registry.get(single_tool)
            desc = tool_obj.description if tool_obj else single_tool
            return Plan(
                goal=user_request,
                steps=[
                    PlanStep(step_number=1, description=desc, tool_name=single_tool)
                ],
            )

        return Plan(goal=user_request, steps=[])

    def select_tool(self, user_request: str) -> Optional[str]:
        """Maps user request to a registered tool name."""
        tools_metadata = [
            {"name": tool.name, "description": tool.description}
            for tool in self.registry.list_tools()
        ]
        return self.llm_client.select_tool(user_request, tools_metadata)
