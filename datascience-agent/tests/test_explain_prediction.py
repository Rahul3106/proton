"""
Phase C Tests: Per-Row Prediction Explanations (SHAP values & local feature attributions).
"""
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from proton_agent.agent import Agent, RuleLLMClient
from proton_agent.api.server import app, session_manager
from proton_agent.context import SessionState
from proton_agent.tools import (
    ExplainPredictionTool,
    TrainModelTool,
    default_registry,
)


@pytest.fixture
def churn_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "churn_data.csv"
    df = pd.DataFrame(
        {
            "age": [25, 30, 45, 50, 23, 34, 40, 60, 28, 55, 38, 42],
            "tenure": [1, 5, 10, 12, 2, 4, 8, 15, 3, 11, 7, 9],
            "balance": [100.0, 500.0, 1200.0, 2000.0, 50.0, 400.0, 900.0, 2500.0, 150.0, 1800.0, 800.0, 1000.0],
            "churn": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def housing_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "housing_data.csv"
    import numpy as np
    rng = np.random.default_rng(42)
    n = 40
    sqft = rng.integers(600, 3500, size=n).astype(float)
    bedrooms = rng.integers(1, 6, size=n).astype(float)
    age = rng.integers(1, 50, size=n).astype(float)
    price = sqft * 150.0 + bedrooms * 10000.0 - age * 500.0 + rng.normal(0, 1000, size=n)
    df = pd.DataFrame(
        {
            "sqft": sqft,
            "bedrooms": bedrooms,
            "age": age,
            "price": price,
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file



def test_explain_prediction_classification(churn_csv: Path):
    session = SessionState()
    session.load_dataset(churn_csv)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn", model_type="random_forest")

    explain_tool = ExplainPredictionTool()
    row_features = {"age": 55, "tenure": 12, "balance": 2200.0}
    res = explain_tool.execute(session, features=row_features)

    assert res.success is True
    data = res.data
    assert "prediction" in data
    assert "top_contributions" in data
    assert len(data["top_contributions"]) == 3

    # Contributions must be sorted by absolute impact
    contribs = [abs(c["contribution"]) for c in data["top_contributions"]]
    assert contribs == sorted(contribs, reverse=True)

    # Every item must have structured fields and rationale summary
    for c in data["top_contributions"]:
        assert "feature" in c
        assert "value" in c
        assert "baseline" in c
        assert "contribution" in c
        assert c["direction"] in ("positive", "negative", "neutral")
        assert len(c["summary"]) > 10


def test_explain_prediction_regression(housing_csv: Path):
    session = SessionState()
    session.load_dataset(housing_csv)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="price", model_type="random_forest")

    explain_tool = ExplainPredictionTool()
    row_features = {"sqft": 2500, "bedrooms": 4, "age": 5}
    res = explain_tool.execute(session, features=row_features)

    assert res.success is True
    data = res.data
    assert data["problem_type"] == "regression"
    assert "top_contributions" in data
    assert len(data["top_contributions"]) == 3
    assert float(data["prediction"]) > 0


def test_explain_prediction_determinism(churn_csv: Path):
    session = SessionState()
    session.load_dataset(churn_csv)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn", model_type="decision_tree")

    explain_tool = ExplainPredictionTool()
    row_features = {"age": 40, "tenure": 8, "balance": 900.0}

    res1 = explain_tool.execute(session, features=row_features)
    res2 = explain_tool.execute(session, features=row_features)

    assert res1.success is True and res2.success is True
    c1 = [c["contribution"] for c in res1.data["top_contributions"]]
    c2 = [c["contribution"] for c in res2.data["top_contributions"]]
    assert c1 == c2


def test_explain_prediction_fallback_linear(churn_csv: Path):
    session = SessionState()
    session.load_dataset(churn_csv)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn", model_type="logistic_regression")

    explain_tool = ExplainPredictionTool()
    row_features = {"age": 30, "tenure": 4, "balance": 400.0}
    res = explain_tool.execute(session, features=row_features)

    assert res.success is True
    assert len(res.data["top_contributions"]) == 3


def test_explain_prediction_from_row_index(churn_csv: Path):
    session = SessionState()
    session.load_dataset(churn_csv)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn")

    explain_tool = ExplainPredictionTool()
    res = explain_tool.execute(session, row_index=2)

    assert res.success is True
    assert res.data["features_input"]["age"] == 45


def test_explain_prediction_without_model(churn_csv: Path):
    session = SessionState()
    session.load_dataset(churn_csv)

    explain_tool = ExplainPredictionTool()
    res = explain_tool.execute(session, features={"age": 30})
    assert res.success is False
    assert "No active trained model" in res.error


def test_planner_explain_prediction_intent():
    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)
    plan = agent.planner.create_plan("explain last prediction")

    assert plan is not None
    assert plan.total_steps == 1
    assert plan.steps[0].tool_name == "explain_prediction"


def test_api_explain_prediction_endpoint(churn_csv: Path):
    client = TestClient(app)
    sid = "test_phase_c_session"
    session_manager.create_session(sid)
    session = session_manager.get_session(sid)
    session.load_dataset(churn_csv)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn")

    # 1. Test POST /api/session/{sid}/explain-prediction
    res = client.post(
        f"/api/session/{sid}/explain-prediction",
        json={"features": {"age": 50, "tenure": 10, "balance": 1500.0}},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "top_contributions" in data
    assert len(data["top_contributions"]) > 0

    # 2. Test predict-row with include_explanation=True
    pred_res = client.post(
        f"/api/session/{sid}/predict-row",
        json={
            "features": {"age": 50, "tenure": 10, "balance": 1500.0},
            "include_explanation": True,
        },
    )
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert pred_data["success"] is True
    assert pred_data["explanation"] is not None
    assert "top_contributions" in pred_data["explanation"]
