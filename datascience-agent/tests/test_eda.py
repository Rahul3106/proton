"""
Tests for Phase 7 EDA Agent tools (outliers, distribution, data quality) and autonomous EDA execution.
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.agent import Agent, RuleLLMClient
from proton_agent.context import SessionState
from proton_agent.tools import (
    AnalyzeDistributionTool,
    AuditDataQualityTool,
    DetectOutliersTool,
    default_registry,
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "eda_data.csv"
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "val": [10.0, 12.0, 11.0, 10.5, 100.0, 11.5, 12.5, 10.8, 11.2, None],
            "category": ["A", "A", "B", "B", "A", "B", "C", "C", "A", "B"],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_detect_outliers_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)

    tool = DetectOutliersTool()
    result = tool.execute(session)

    assert result.success is True
    assert result.data["total_outliers"] >= 1
    assert "val" in result.data["affected_columns"]


def test_analyze_distribution_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)

    tool = AnalyzeDistributionTool()
    result = tool.execute(session)

    assert result.success is True
    assert "val" in result.data["numerical_distributions"]
    assert "category" in result.data["categorical_distributions"]


def test_audit_data_quality_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)

    tool = AuditDataQualityTool()
    result = tool.execute(session)

    assert result.success is True
    assert "quality_score" in result.data
    assert result.data["quality_score"] <= 100.0
    assert len(result.data["warnings"]) >= 1


def test_autonomous_eda_plan_execution(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)

    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)
    response = agent.process_request("perform complete EDA", session)

    assert response.plan is not None
    assert response.plan.total_steps == 7
    assert response.plan.is_complete is True
    assert "Data Quality Score" in response.summary
    assert "Outliers" in response.summary
