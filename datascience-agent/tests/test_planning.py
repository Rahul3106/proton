"""
Tests for Agent Planning, multi-step plan execution, and summary findings (Phase 6 & Phase 7).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.agent import Agent, Planner, RuleLLMClient
from proton_agent.context import SessionState
from proton_agent.tools import default_registry


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "age": [25.0, 30.0, None, 40.0],
            "income": [50000.0, 60000.0, 75000.0, 90000.0],
            "city": ["NY", "SF", "LA", "NY"],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_planner_creates_multi_step_plan():
    planner = Planner(llm_client=RuleLLMClient(), registry=default_registry)
    plan = planner.create_plan("analyze this dataset")

    assert plan.total_steps == 7
    assert plan.steps[0].tool_name == "inspect_dataset"
    assert plan.steps[1].tool_name == "audit_data_quality"
    assert plan.steps[2].tool_name == "detect_missing_values"
    assert plan.steps[3].tool_name == "detect_duplicates"
    assert plan.steps[4].tool_name == "analyze_distribution"
    assert plan.steps[5].tool_name == "detect_outliers"
    assert plan.steps[6].tool_name == "calculate_correlation"


def test_planner_creates_single_step_plan():
    planner = Planner(llm_client=RuleLLMClient(), registry=default_registry)
    plan = planner.create_plan("detect missing values")

    assert plan.total_steps == 1
    assert plan.steps[0].tool_name == "detect_missing_values"


def test_agent_executes_multi_step_plan(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)

    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)
    response = agent.process_request("analyze this dataset", session)

    assert response.plan is not None
    assert response.plan.is_complete is True
    assert response.plan.completed_steps == 7
    assert "Dataset Overview" in response.summary
    assert "Missing" in response.summary or "Quality" in response.summary
    assert "Duplicates" in response.summary


def test_advisory_questions_do_not_execute_cleaning(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)

    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)
    response = agent.process_request("should I clean this dataset?", session)

    assert response.selected_tool == "custom_llm_qa"
    assert "Advisory Assessment" in response.explanation


def test_planner_specific_model_training():
    planner = Planner(llm_client=RuleLLMClient(), registry=default_registry)
    plan = planner.create_plan("train decision tree")

    assert plan.total_steps == 2
    assert plan.steps[0].tool_name == "train_model"
    assert plan.steps[0].tool_kwargs == {"model_type": "decision_tree"}
    assert plan.steps[1].tool_name == "predict"


def test_planner_predict_for_specific_model():
    planner = Planner(llm_client=RuleLLMClient(), registry=default_registry)
    plan = planner.create_plan("predict for decision tree")

    assert plan.total_steps == 1
    assert plan.steps[0].tool_name == "predict"
    assert plan.steps[0].tool_kwargs == {"model_name": "decision_tree"}

