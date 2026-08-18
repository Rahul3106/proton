"""
Tests for Phase 8 Data Cleaning Agent tools and transformations.
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.agent import Agent, RuleLLMClient
from proton_agent.context import SessionState
from proton_agent.tools import (
    CleanDatasetTool,
    EncodeFeaturesTool,
    HandleMissingValuesTool,
    ScaleFeaturesTool,
    default_registry,
)


@pytest.fixture
def dirty_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "dirty_data.csv"
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 4, 5, 6],
            "age": [25.0, 30.0, 30.0, None, 50.0, 40.0],
            "gender": ["M", "F", "F", "M", None, "F"],
            "constant": [1, 1, 1, 1, 1, 1],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_clean_dataset_tool(dirty_csv: Path):
    session = SessionState()
    session.load_dataset(dirty_csv)

    tool = CleanDatasetTool()
    result = tool.execute(session)

    assert result.success is True
    assert result.data["duplicate_rows_removed"] == 1
    assert "constant" in result.data["constant_columns_dropped"]

    # Verify session in-memory dataset state updated
    df_clean = session.current_dataset.df
    assert len(df_clean) == 5
    assert "constant" not in df_clean.columns
    assert df_clean["age"].isnull().sum() == 0
    assert df_clean["gender"].isnull().sum() == 0
    assert len(session.transformations) >= 2


def test_handle_missing_values_tool(dirty_csv: Path):
    session = SessionState()
    session.load_dataset(dirty_csv)

    tool = HandleMissingValuesTool()
    result = tool.execute(session, strategy="median")

    assert result.success is True
    assert result.data["total_missing_after"] == 0
    assert session.current_dataset.total_missing == 0


def test_encode_features_tool(dirty_csv: Path):
    session = SessionState()
    session.load_dataset(dirty_csv)

    tool = EncodeFeaturesTool()
    result = tool.execute(session, method="onehot")

    assert result.success is True
    assert "gender" in result.data["original_categorical_columns"]


def test_scale_features_tool(dirty_csv: Path):
    session = SessionState()
    session.load_dataset(dirty_csv)

    tool = ScaleFeaturesTool()
    result = tool.execute(session, method="standard")

    assert result.success is True
    assert "age" in result.data["scaled_columns"]


def test_agent_cleaning_request(dirty_csv: Path):
    session = SessionState()
    session.load_dataset(dirty_csv)

    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)
    response = agent.process_request("clean this dataset", session)

    assert response.selected_tool == "clean_dataset"
    assert response.tool_result is not None
    assert response.tool_result.success is True
