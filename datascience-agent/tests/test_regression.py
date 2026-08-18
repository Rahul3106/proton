"""
Phase B -- Regression Support Tests.

Uses a synthetic housing-price-style dataset (continuous numeric target)
to exercise the full regression path: detection, training, comparison,
tuning, explanation, and prediction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proton_agent.context import SessionState
from proton_agent.tools import TrainModelTool, CompareModelsTool, DetectProblemTypeTool, PredictTool
from proton_agent.tools.modeling.explain import ExplainModelTool


# -- Shared fixture ----------------------------------------------------------

@pytest.fixture
def reg_dataset(tmp_path: Path) -> Path:
    """
    Synthetic housing-price dataset with a continuous numeric target (price).
    60 rows, enough for a meaningful train/test split.
    """
    rng = np.random.default_rng(42)
    n = 60
    sqft = rng.integers(500, 3500, size=n).astype(float)
    bedrooms = rng.integers(1, 6, size=n).astype(float)
    age = rng.integers(1, 50, size=n).astype(float)
    distance_km = rng.uniform(0.5, 30.0, size=n)
    price = (
        sqft * 150.0
        + bedrooms * 10_000.0
        - age * 500.0
        - distance_km * 2_000.0
        + rng.normal(0, 5_000, size=n)
    )
    df = pd.DataFrame(
        {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "age": age,
            "distance_km": distance_km,
            "price": price,
        }
    )
    csv_path = tmp_path / "housing.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# -- 1. Task-type detection --------------------------------------------------

def test_detect_regression(reg_dataset: Path) -> None:
    """Continuous numeric target with >20 unique values -> regression."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    tool = DetectProblemTypeTool()
    result = tool.execute(session, target_column="price")

    assert result.success is True
    assert result.data["problem_type"] == "regression", (
        f"Expected 'regression', got '{result.data['problem_type']}'"
    )
    assert result.data["target_column"] == "price"
    assert "continuous" in result.data["detail"].lower()


# -- 2. Auto-train returns RMSE / MAE / R2 -----------------------------------

def test_train_regression_auto(reg_dataset: Path) -> None:
    """train_model on a continuous target -> metrics contain rmse, mae, r2_score."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    tool = TrainModelTool()
    result = tool.execute(session, target_column="price")

    assert result.success is True, result.error
    assert result.data["problem_type"] == "regression"
    metrics = result.data["metrics"]
    for key in ("rmse", "mae", "r2_score"):
        assert key in metrics, f"Missing metric '{key}' in {metrics}"
    assert isinstance(metrics["r2_score"], float)
    assert "accuracy" not in metrics
    assert "f1_score" not in metrics


# -- 3. Ridge alias ----------------------------------------------------------

def test_train_ridge(reg_dataset: Path) -> None:
    """'train ridge' alias routes to Ridge Regressor."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    tool = TrainModelTool()
    result = tool.execute(session, target_column="price", model_type="ridge")

    assert result.success is True, result.error
    assert "Ridge" in result.data["model_name"]
    assert result.data["problem_type"] == "regression"
    assert "r2_score" in result.data["metrics"]


# -- 4. Lasso alias ----------------------------------------------------------

def test_train_lasso(reg_dataset: Path) -> None:
    """'train lasso' alias routes to Lasso Regressor."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    tool = TrainModelTool()
    result = tool.execute(session, target_column="price", model_type="lasso")

    assert result.success is True, result.error
    assert "Lasso" in result.data["model_name"]
    assert result.data["problem_type"] == "regression"
    assert "r2_score" in result.data["metrics"]


# -- 5. Linear Regression alias ----------------------------------------------

def test_train_linear_regression(reg_dataset: Path) -> None:
    """'train linear_regression' routes to LinearRegression."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    tool = TrainModelTool()
    result = tool.execute(session, target_column="price", model_type="linear_regression")

    assert result.success is True, result.error
    assert "Linear Regression" in result.data["model_name"]
    assert result.data["problem_type"] == "regression"


# -- 6. Extra Trees alias ----------------------------------------------------

def test_train_extra_trees_regressor(reg_dataset: Path) -> None:
    """'train extra_trees' routes to ExtraTreesRegressor for regression targets."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    tool = TrainModelTool()
    result = tool.execute(session, target_column="price", model_type="extra_trees")

    assert result.success is True, result.error
    assert "Extra Trees" in result.data["model_name"]
    assert result.data["problem_type"] == "regression"


# -- 7. Compare models uses r2_score -----------------------------------------

def test_compare_regression(reg_dataset: Path) -> None:
    """compare_models on a regression target -> primary_metric is r2_score."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    tool = CompareModelsTool()
    result = tool.execute(session, target_column="price")

    assert result.success is True, result.error
    assert result.data["problem_type"] == "regression"
    assert result.data["candidate_count"] >= 3

    for entry in result.data["comparison_matrix"]:
        assert "r2_score" in entry["metrics"], (
            f"Model '{entry['model_name']}' missing r2_score: {entry['metrics']}"
        )
        assert "f1_score" not in entry["metrics"], (
            f"Classification metric f1_score leaked into regression result for '{entry['model_name']}'"
        )


# -- 8. Tune model reports best_params ---------------------------------------

def test_tune_regression(reg_dataset: Path) -> None:
    """tune=True on Random Forest Regressor runs RandomizedSearchCV and reports best_params."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    tool = TrainModelTool()
    result = tool.execute(
        session, target_column="price", model_type="random_forest", tune=True
    )

    assert result.success is True, result.error
    assert result.data["tuned"] is True
    assert isinstance(result.data["best_params"], dict)
    assert "r2_score" in result.data["metrics"]


# -- 9. Explain model saves residual plot + predicted-vs-actual ---------------

def test_explain_regression(reg_dataset: Path) -> None:
    """ExplainModelTool on a regression model saves two PNG plots."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="price", model_type="random_forest")

    explain_tool = ExplainModelTool()
    result = explain_tool.execute(session)

    assert result.success is True, result.error
    assert result.data["problem_type"] == "regression"

    assert "residuals_summary" in result.data
    rs = result.data["residuals_summary"]
    for key in ("mean_residual", "std_residual", "max_error"):
        assert key in rs, f"Missing key '{key}' in residuals_summary"

    assert "residual_plot" in result.data, (
        f"residual_plot missing; error: {result.data.get('residual_plot_error')}"
    )
    assert Path(result.data["residual_plot"]).exists()

    assert "predicted_vs_actual_plot" in result.data, (
        f"predicted_vs_actual_plot missing; error: {result.data.get('predicted_vs_actual_plot_error')}"
    )
    assert Path(result.data["predicted_vs_actual_plot"]).exists()

    assert "confusion_matrix" not in result.data
    assert "roc_auc" not in result.data


# -- 10. Predict returns numeric (float) values ------------------------------

def test_predict_regression_numeric(reg_dataset: Path) -> None:
    """Predictions on a regression model are float-parseable strings."""
    session = SessionState()
    session.load_dataset(reg_dataset)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="price", model_type="random_forest")

    pred_tool = PredictTool()
    result = pred_tool.execute(session, n_samples=5)

    assert result.success is True, result.error
    assert len(result.data["predictions"]) == 5

    for row in result.data["predictions"]:
        val = row["prediction"]
        try:
            float(val)
        except (ValueError, TypeError):
            pytest.fail(f"Prediction value '{val}' is not float-parseable")
