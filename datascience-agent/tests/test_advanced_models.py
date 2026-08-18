"""
Phase D Tests: XGBoost, LightGBM, CatBoost & Optuna Bayesian Hyperparameter Tuning.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from proton_agent.agent import Agent, RuleLLMClient
from proton_agent.context import SessionState
from proton_agent.tools import (
    CompareModelsTool,
    PredictTool,
    TrainModelTool,
    default_registry,
)


@pytest.fixture
def churn_df(tmp_path: Path) -> Path:
    csv_file = tmp_path / "churn_data.csv"
    rng = np.random.default_rng(42)
    n = 60
    age = rng.integers(20, 70, size=n).astype(float)
    tenure = rng.integers(1, 15, size=n).astype(float)
    balance = rng.uniform(100.0, 5000.0, size=n)
    churn = ((age > 45) & (balance > 1500.0)).astype(int)
    df = pd.DataFrame({"age": age, "tenure": tenure, "balance": balance, "churn": churn})
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def housing_df(tmp_path: Path) -> Path:
    csv_file = tmp_path / "housing_data.csv"
    rng = np.random.default_rng(42)
    n = 60
    sqft = rng.integers(600, 3500, size=n).astype(float)
    bedrooms = rng.integers(1, 6, size=n).astype(float)
    age = rng.integers(1, 50, size=n).astype(float)
    price = sqft * 150.0 + bedrooms * 10000.0 - age * 500.0 + rng.normal(0, 1000, size=n)
    df = pd.DataFrame({"sqft": sqft, "bedrooms": bedrooms, "age": age, "price": price})
    df.to_csv(csv_file, index=False)
    return csv_file


def test_xgboost_classification(churn_df: Path):
    session = SessionState()
    session.load_dataset(churn_df)

    trainer = TrainModelTool()
    res = trainer.execute(session, target_column="churn", model_type="xgboost")
    assert res.success is True, res.error
    assert "XGBoost" in res.data["model_name"]
    assert "f1_score" in res.data["metrics"]

    pred_tool = PredictTool()
    p_res = pred_tool.execute(session, n_samples=3)
    assert p_res.success is True
    assert len(p_res.data["predictions"]) == 3


def test_xgboost_regression(housing_df: Path):
    session = SessionState()
    session.load_dataset(housing_df)

    trainer = TrainModelTool()
    res = trainer.execute(session, target_column="price", model_type="xgboost")
    assert res.success is True, res.error
    assert "XGBoost" in res.data["model_name"]
    assert "r2_score" in res.data["metrics"]


def test_lightgbm_classification(churn_df: Path):
    session = SessionState()
    session.load_dataset(churn_df)

    trainer = TrainModelTool()
    res = trainer.execute(session, target_column="churn", model_type="lightgbm")
    assert res.success is True, res.error
    assert "LightGBM" in res.data["model_name"]
    assert "accuracy" in res.data["metrics"]


def test_lightgbm_regression(housing_df: Path):
    session = SessionState()
    session.load_dataset(housing_df)

    trainer = TrainModelTool()
    res = trainer.execute(session, target_column="price", model_type="lightgbm")
    assert res.success is True, res.error
    assert "LightGBM" in res.data["model_name"]
    assert "rmse" in res.data["metrics"]


def test_catboost_classification(churn_df: Path):
    session = SessionState()
    session.load_dataset(churn_df)

    trainer = TrainModelTool()
    res = trainer.execute(session, target_column="churn", model_type="catboost")
    assert res.success is True, res.error
    assert "CatBoost" in res.data["model_name"]
    assert "f1_score" in res.data["metrics"]


def test_catboost_regression(housing_df: Path):
    session = SessionState()
    session.load_dataset(housing_df)

    trainer = TrainModelTool()
    res = trainer.execute(session, target_column="price", model_type="catboost")
    assert res.success is True, res.error
    assert "CatBoost" in res.data["model_name"]
    assert "r2_score" in res.data["metrics"]


def test_optuna_tuning_classification(churn_df: Path):
    session = SessionState()
    session.load_dataset(churn_df)

    trainer = TrainModelTool()
    res = trainer.execute(session, target_column="churn", model_type="random_forest", tune=True)
    assert res.success is True, res.error
    assert res.data["tuned"] is True
    assert "Optuna" in res.data["tuning_engine"]
    assert res.data["trials_run"] > 0
    assert isinstance(res.data["best_params"], dict)
    assert len(res.data["best_params"]) > 0
    assert "tuning_rationale" in res.data
    assert "trials on" in res.data["tuning_rationale"]


def test_optuna_tuning_regression(housing_df: Path):
    session = SessionState()
    session.load_dataset(housing_df)

    trainer = TrainModelTool()
    res = trainer.execute(session, target_column="price", model_type="xgboost", tune=True)
    assert res.success is True, res.error
    assert res.data["tuned"] is True
    assert "Optuna" in res.data["tuning_engine"]
    assert isinstance(res.data["best_params"], dict)
    assert "tuning_rationale" in res.data


def test_compare_models_7_families(churn_df: Path):
    session = SessionState()
    session.load_dataset(churn_df)

    compare_tool = CompareModelsTool()
    res = compare_tool.execute(session, target_column="churn")
    assert res.success is True, res.error
    assert res.data["candidate_count"] == 7
    model_names = [m["model_name"] for m in res.data["comparison_matrix"]]
    assert any("Random Forest" in n for n in model_names)
    assert any("Gradient Boosting" in n for n in model_names)
    assert any("XGBoost" in n for n in model_names)
    assert any("LightGBM" in n for n in model_names)
    assert any("CatBoost" in n for n in model_names)
    assert any("Decision Tree" in n for n in model_names)
    assert any("Logistic Regression" in n for n in model_names)


def test_planner_advanced_models_routing():
    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)

    p_xgb = agent.planner.create_plan("train xgboost")
    assert p_xgb.steps[0].tool_kwargs["model_type"] == "xgboost"

    p_lgb = agent.planner.create_plan("train lightgbm")
    assert p_lgb.steps[0].tool_kwargs["model_type"] == "lightgbm"

    p_cb = agent.planner.create_plan("train catboost")
    assert p_cb.steps[0].tool_kwargs["model_type"] == "catboost"
