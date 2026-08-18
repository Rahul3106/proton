"""
Model Recommendation Engine: analyzes dataset dimensions, column distributions,
and target variable to suggest the Top 5 optimal machine learning models.
"""
from __future__ import annotations

from typing import Any, Dict, List
import pandas as pd

from proton_agent.context.session import SessionState
from proton_agent.tools.modeling.detect import infer_problem_type, infer_target_column


def get_model_recommendations(session: SessionState, target_column: str | None = None) -> Dict[str, Any]:
    """
    Analyzes the loaded dataset and returns tailored model recommendations including the Top 5 picks.
    """
    if not session.current_dataset or session.current_dataset.df is None:
        return {
            "has_dataset": False,
            "problem_type": "classification",
            "target_column": None,
            "top_5_models": [],
            "all_categories": get_all_model_categories("classification"),
        }

    df = session.current_dataset.df
    if target_column and target_column in df.columns:
        target = target_column
    elif session.target_column and session.target_column in df.columns:
        target = session.target_column
    else:
        target = infer_target_column(df)

    session.target_column = target
    y_raw = df[target]

    # Detect classification vs regression
    problem_type, _ = infer_problem_type(y_raw)
    is_classification = problem_type == "classification"


    rows = len(df)
    features_count = max(1, len(df.columns) - 1)

    top_5 = []

    if is_classification:
        if rows < 3000:
            top_5 = [
                {
                    "model_type": "random_forest",
                    "display_name": "Random Forest Classifier",
                    "badge": "★ #1 Top Pick",
                    "tag": "Ensemble",
                    "rationale": f"Optimal accuracy & generalization on {rows:,} rows; robust to outliers & noise.",
                },
                {
                    "model_type": "gradient_boosting",
                    "display_name": "Gradient Boosting Classifier",
                    "badge": "★ High Precision",
                    "tag": "Boosting",
                    "rationale": "Sequentially optimizes tree loss gradients to capture subtle decision patterns.",
                },
                {
                    "model_type": "extra_trees",
                    "display_name": "Extra Trees Classifier",
                    "badge": "★ Low Variance",
                    "tag": "Randomized Forest",
                    "rationale": "Extremely randomized threshold splits offer rapid training and reduced variance.",
                },
                {
                    "model_type": "logistic_regression",
                    "display_name": "Logistic Regression",
                    "badge": "★ Fast & Interpretable",
                    "tag": "Linear",
                    "rationale": "Convex linear baseline providing clear log-odds feature importance.",
                },
                {
                    "model_type": "svm",
                    "display_name": "Support Vector Machine (SVC)",
                    "badge": "★ Non-Linear Boundary",
                    "tag": "Kernel",
                    "rationale": "High-margin hyperplane separation with RBF kernel for complex feature spaces.",
                },
            ]
        else:
            top_5 = [
                {
                    "model_type": "hist_gradient_boosting",
                    "display_name": "Hist Gradient Boosting Classifier",
                    "badge": "★ #1 Big Data Pick",
                    "tag": "Histogram Boosting",
                    "rationale": f"Histogram-binned splitting engineered for high throughput on {rows:,} rows.",
                },
                {
                    "model_type": "random_forest",
                    "display_name": "Random Forest Classifier",
                    "badge": "★ Ensemble Power",
                    "tag": "Ensemble",
                    "rationale": "Parallel decision tree bagging with strong resistance to overfitting.",
                },
                {
                    "model_type": "gradient_boosting",
                    "display_name": "Gradient Boosting Classifier",
                    "badge": "★ High Precision",
                    "tag": "Boosting",
                    "rationale": "Deep residual fitting that frequently achieves top leaderboard rank.",
                },
                {
                    "model_type": "extra_trees",
                    "display_name": "Extra Trees Classifier",
                    "badge": "★ High Speed",
                    "tag": "Randomized Forest",
                    "rationale": "Accelerated multi-tree ensembling with extreme split randomization.",
                },
                {
                    "model_type": "logistic_regression",
                    "display_name": "Logistic Regression",
                    "badge": "★ Fast Baseline",
                    "tag": "Linear",
                    "rationale": "Fast convergence on large tabular datasets with clean probability outputs.",
                },
            ]
    else:
        # Regression
        if rows < 3000:
            top_5 = [
                {
                    "model_type": "random_forest",
                    "display_name": "Random Forest Regressor",
                    "badge": "★ #1 Top Pick",
                    "tag": "Ensemble",
                    "rationale": f"Captures complex non-linear curve relationships without overfitting on {rows:,} rows.",
                },
                {
                    "model_type": "gradient_boosting",
                    "display_name": "Gradient Boosting Regressor",
                    "badge": "★ Low RMSE",
                    "tag": "Boosting",
                    "rationale": "Iteratively fits residuals to minimize squared error metrics.",
                },
                {
                    "model_type": "extra_trees",
                    "display_name": "Extra Trees Regressor",
                    "badge": "★ Low Variance",
                    "tag": "Randomized Forest",
                    "rationale": "Averages randomized tree predictions for smooth continuous regressions.",
                },
                {
                    "model_type": "ridge",
                    "display_name": "Ridge Regressor (L2)",
                    "badge": "★ Regularized Linear",
                    "tag": "Regularized",
                    "rationale": "L2 weight shrinkage suppresses collinearity and prevents overfitting.",
                },
                {
                    "model_type": "svm",
                    "display_name": "Support Vector Regressor (SVR)",
                    "badge": "★ Margin Robustness",
                    "tag": "Kernel",
                    "rationale": "Epsilon-insensitive loss function offers immunity to numerical outliers.",
                },
            ]
        else:
            top_5 = [
                {
                    "model_type": "hist_gradient_boosting",
                    "display_name": "Hist Gradient Boosting Regressor",
                    "badge": "★ #1 Big Data Pick",
                    "tag": "Histogram Boosting",
                    "rationale": f"Scalable binned regression algorithm optimized for {rows:,} continuous records.",
                },
                {
                    "model_type": "random_forest",
                    "display_name": "Random Forest Regressor",
                    "badge": "★ Ensemble Power",
                    "tag": "Ensemble",
                    "rationale": "Robust multi-tree averaging for reliable continuous value estimations.",
                },
                {
                    "model_type": "gradient_boosting",
                    "display_name": "Gradient Boosting Regressor",
                    "badge": "★ Low RMSE",
                    "tag": "Boosting",
                    "rationale": "High-precision residual error minimization across continuous target values.",
                },
                {
                    "model_type": "ridge",
                    "display_name": "Ridge Regressor (L2)",
                    "badge": "★ Fast Linear",
                    "tag": "Linear",
                    "rationale": "Closed-form analytical solution offering immediate regression baselines.",
                },
                {
                    "model_type": "extra_trees",
                    "display_name": "Extra Trees Regressor",
                    "badge": "★ High Speed",
                    "tag": "Randomized Forest",
                    "rationale": "High-performance parallel tree ensembling with randomized feature cutoffs.",
                },
            ]

    return {
        "has_dataset": True,
        "problem_type": problem_type,
        "target_column": target,
        "rows": rows,
        "features_count": features_count,
        "top_5_models": top_5,
        "all_categories": get_all_model_categories(problem_type),
    }


def get_all_model_categories(problem_type: str = "classification") -> List[Dict[str, Any]]:
    """Returns the full catalog of supported models organized into categories."""
    is_clf = problem_type.lower() == "classification"
    return [
        {
            "category": "Ensemble & Tree Models",
            "models": [
                {"type": "random_forest", "name": "Random Forest " + ("Classifier" if is_clf else "Regressor")},
                {"type": "gradient_boosting", "name": "Gradient Boosting " + ("Classifier" if is_clf else "Regressor")},
                {"type": "extra_trees", "name": "Extra Trees " + ("Classifier" if is_clf else "Regressor")},
                {"type": "hist_gradient_boosting", "name": "Hist Gradient Boosting " + ("Classifier" if is_clf else "Regressor")},
                {"type": "adaboost", "name": "AdaBoost " + ("Classifier" if is_clf else "Regressor")},
                {"type": "decision_tree", "name": "Decision Tree " + ("Classifier" if is_clf else "Regressor")},
            ],
        },
        {
            "category": "Linear & Regularized Models",
            "models": [
                {"type": "logistic_regression" if is_clf else "linear_regression", "name": "Logistic Regression" if is_clf else "Linear Regression"},
                {"type": "ridge", "name": "Ridge " + ("Classifier" if is_clf else "Regressor (L2)")},
                {"type": "lasso", "name": "Lasso " + ("Logistic (L1)" if is_clf else "Regressor (L1)")},
            ],
        },
        {
            "category": "Distance & Kernel Models",
            "models": [
                {"type": "svm", "name": "Support Vector Machine " + ("(SVC)" if is_clf else "(SVR)")},
                {"type": "knn", "name": "K-Nearest Neighbors (KNN)"},
            ],
        },
        {
            "category": "Neural & Probabilistic Models",
            "models": [
                {"type": "mlp", "name": "MLP Neural Network " + ("Classifier" if is_clf else "Regressor")},
                {"type": "naive_bayes", "name": "Gaussian Naive Bayes" if is_clf else "Bayesian Ridge Regressor"},
            ],
        },
    ]
