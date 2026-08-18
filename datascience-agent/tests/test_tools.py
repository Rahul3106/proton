"""
Tests for Data Science Tools and ToolRegistry (Phase 4).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.context import SessionState
from proton_agent.tools import (
    CalculateCorrelationTool,
    CalculateStatisticsTool,
    DescribeDatasetTool,
    DetectDuplicatesTool,
    DetectMissingValuesTool,
    InspectDataTypesTool,
    InspectDatasetTool,
    LoadDatasetTool,
    ToolRegistry,
    default_registry,
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "sample_data.csv"
    df = pd.DataFrame(
        {
            "age": [25.0, 30.0, None, 40.0, 25.0],
            "income": [50000.0, 60000.0, 75000.0, 90000.0, 50000.0],
            "city": ["NY", "SF", "LA", None, "NY"],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_tool_registry():
    registry = ToolRegistry()
    tool = DetectMissingValuesTool()
    registry.register(tool)

    assert registry.get("detect_missing_values") is tool
    assert "detect_missing_values" in registry.tool_names()
    assert len(registry.list_tools()) == 1


def test_load_dataset_tool(sample_csv: Path):
    session = SessionState()
    tool = LoadDatasetTool()

    res = tool.execute(session, filepath=str(sample_csv))
    assert res.success is True
    assert res.data["rows"] == 5
    assert res.data["columns"] == 3


def test_inspect_dataset_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    tool = InspectDatasetTool()

    res = tool.execute(session)
    assert res.success is True
    assert res.data["rows"] == 5
    assert len(res.data["preview"]) == 5


def test_missing_values_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    tool = DetectMissingValuesTool()

    res = tool.execute(session)
    assert res.success is True
    assert res.data["total_missing"] == 2
    assert res.data["columns_with_missing_count"] == 2


def test_duplicates_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    tool = DetectDuplicatesTool()

    res = tool.execute(session)
    assert res.success is True
    assert res.data["duplicate_rows"] == 1
    assert res.data["has_duplicates"] is True


def test_dtypes_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    tool = InspectDataTypesTool()

    res = tool.execute(session)
    assert res.success is True
    assert "age" in res.data["numerical_columns"]
    assert "income" in res.data["numerical_columns"]
    assert "city" in res.data["categorical_columns"]


def test_describe_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    tool = DescribeDatasetTool()

    res = tool.execute(session)
    assert res.success is True
    assert "age" in res.data["numerical_summary"]
    assert "income" in res.data["numerical_summary"]


def test_statistics_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    tool = CalculateStatisticsTool()

    res = tool.execute(session)
    assert res.success is True
    assert "age" in res.data["statistics"]
    assert res.data["statistics"]["age"]["mean"] == 31.666666666666668 or res.data["statistics"]["age"]["count"] == 4


def test_correlation_tool(sample_csv: Path):
    session = SessionState()
    session.load_dataset(sample_csv)
    tool = CalculateCorrelationTool()

    res = tool.execute(session)
    assert res.success is True
    assert len(res.data["top_correlations"]) >= 1
    pair = res.data["top_correlations"][0]
    assert "feature_1" in pair and "feature_2" in pair


def test_tools_error_without_dataset():
    session = SessionState()
    for tool in [
        InspectDatasetTool(),
        DetectMissingValuesTool(),
        DetectDuplicatesTool(),
        InspectDataTypesTool(),
        DescribeDatasetTool(),
        CalculateStatisticsTool(),
        CalculateCorrelationTool(),
    ]:
        res = tool.execute(session)
        assert res.success is False
        assert res.error is not None
