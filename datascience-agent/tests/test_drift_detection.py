"""
Phase F Tests: Dataset Drift & Distribution Shift Detection Tool.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from proton_agent.agent import Agent, RuleLLMClient
from proton_agent.context import SessionState
from proton_agent.tools import DriftDetectionTool, default_registry


@pytest.fixture
def baseline_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "baseline.csv"
    rng = np.random.default_rng(42)
    n = 100
    age = rng.normal(35, 5, size=n)
    income = rng.normal(60000, 10000, size=n)
    category = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    target = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    df = pd.DataFrame({"age": age, "income": income, "category": category, "target": target})
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def drifted_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "drifted.csv"
    rng = np.random.default_rng(123)
    n = 100
    # Significant shift in age and income distributions
    age = rng.normal(55, 8, size=n)
    income = rng.normal(120000, 25000, size=n)
    # Significant category distribution flip
    category = rng.choice(["A", "B", "C"], size=n, p=[0.1, 0.2, 0.7])
    target = rng.choice([0, 1], size=n, p=[0.2, 0.8])
    df = pd.DataFrame({"age": age, "income": income, "category": category, "target": target})
    df.to_csv(csv_file, index=False)
    return csv_file


def test_drift_detection_identical_splits_no_drift(baseline_csv: Path):
    session = SessionState()
    session.load_dataset(baseline_csv)

    tool = DriftDetectionTool()
    res = tool.execute(session)
    assert res.success is True, res.error
    assert res.data["total_features_analyzed"] >= 2
    # In split of same distribution, most features should be stable
    assert res.data["overall_status"] in ("No Significant Drift (Stable)", "Moderate Drift Detected")
    assert "plot_path" in res.data


def test_drift_detection_covariate_shift(baseline_csv: Path, drifted_csv: Path):
    session = SessionState()
    tool = DriftDetectionTool()
    res = tool.execute(
        session,
        reference_path=str(baseline_csv),
        current_path=str(drifted_csv),
    )
    assert res.success is True, res.error
    assert res.data["drift_detected"] is True
    assert res.data["drifted_features_count"] >= 2

    # Check that age and income have high PSI / significant drift
    metrics_map = {m["feature"]: m for m in res.data["feature_drift_metrics"]}
    assert "age" in metrics_map
    assert metrics_map["age"]["is_drifted"] is True
    assert metrics_map["age"]["psi_score"] > 0.25
    assert metrics_map["age"]["drift_status"] == "Significant Drift"


def test_drift_detection_categorical_shift(baseline_csv: Path, drifted_csv: Path):
    session = SessionState()
    tool = DriftDetectionTool()
    res = tool.execute(
        session,
        reference_path=str(baseline_csv),
        current_path=str(drifted_csv),
    )
    assert res.success is True
    metrics_map = {m["feature"]: m for m in res.data["feature_drift_metrics"]}
    assert "category" in metrics_map
    assert metrics_map["category"]["data_type"] == "categorical"
    assert metrics_map["category"]["is_drifted"] is True


def test_drift_detection_plot_generation(baseline_csv: Path, drifted_csv: Path):
    session = SessionState()
    tool = DriftDetectionTool()
    res = tool.execute(
        session,
        reference_path=str(baseline_csv),
        current_path=str(drifted_csv),
    )
    assert res.success is True
    assert res.data["plot_path"] is not None
    plot_file = Path(res.data["plot_path"])
    assert plot_file.exists()
    assert plot_file.stat().st_size > 1000


def test_planner_drift_routing():
    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)
    plan = agent.planner.create_plan("detect data drift")
    assert len(plan.steps) >= 1
    assert plan.steps[0].tool_name == "detect_drift"

    plan2 = agent.planner.create_plan("check covariate shift")
    assert len(plan2.steps) >= 1
    assert plan2.steps[0].tool_name == "detect_drift"


def test_api_detect_drift_endpoint(baseline_csv: Path, drifted_csv: Path):
    from fastapi.testclient import TestClient
    from proton_agent.api.server import app

    client = TestClient(app)
    # Create session and load
    s_res = client.post("/api/session", json={"session_id": "test_drift_sess"})
    assert s_res.status_code in (200, 201)


    load_res = client.post(
        "/api/session/test_drift_sess/load",
        json={"filepath": str(baseline_csv)},
    )
    assert load_res.status_code == 200

    # Call detect-drift
    drift_res = client.post(
        "/api/session/test_drift_sess/detect-drift",
        params={
            "reference_path": str(baseline_csv),
            "current_path": str(drifted_csv),
        },
    )
    assert drift_res.status_code == 200
    data = drift_res.json()
    assert data["drift_detected"] is True
    assert "feature_drift_metrics" in data
