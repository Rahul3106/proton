"""
Tests for Agent core, Planner, Executor, and LLM abstraction (Phase 5).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.agent import Agent, Executor, Planner, RuleLLMClient
from proton_agent.context import SessionState
from proton_agent.tools import default_registry


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "age": [25, 30, None, 40],
            "income": [50000, 60000, 75000, 90000],
            "city": ["NY", "SF", "LA", "NY"],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_rule_llm_client_intent_matching():
    client = RuleLLMClient()
    tools = [
        {"name": "detect_missing_values", "description": "Detect missing values"},
        {"name": "detect_duplicates", "description": "Detect duplicates"},
    ]

    assert client.select_tool("what columns have missing values?", tools) == "detect_missing_values"
    assert client.select_tool("check for duplicates in my data", tools) == "detect_duplicates"
    assert client.select_tool("unrelated random text", tools) is None


def test_planner():
    client = RuleLLMClient()
    planner = Planner(llm_client=client, registry=default_registry)

    tool_name = planner.select_tool("find feature correlations")
    assert tool_name == "calculate_correlation"


def test_executor(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    executor = Executor(registry=default_registry)

    result = executor.execute_tool("detect_missing_values", session)
    assert result.success is True
    assert result.data["total_missing"] == 1


def test_agent_end_to_end(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)

    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)

    response = agent.process_request("what columns have missing values?", session)
    assert response.selected_tool == "detect_missing_values"
    assert response.tool_result is not None
    assert response.tool_result.success is True
    assert response.tool_result.data["total_missing"] == 1


def test_agent_custom_question(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)

    response = agent.process_request("why is customer retention declining in Q3?", session)
    assert response.selected_tool == "custom_llm_qa"
    assert response.plan is not None
    assert response.plan.is_complete is True
    assert "Query:" in response.explanation
