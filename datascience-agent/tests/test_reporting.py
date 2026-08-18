"""
Tests for Phase 12 Reporting Agent tools (generate EDA & model reports).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.context import SessionState
from proton_agent.tools import GenerateReportTool, TrainModelTool


@pytest.fixture
def report_dataset(tmp_path: Path) -> Path:
    csv_file = tmp_path / "report_data.csv"
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "target": [0, 1, 0, 1, 0],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_generate_eda_report_tool(report_dataset: Path):
    session = SessionState()
    session.load_dataset(report_dataset)

    tool = GenerateReportTool()
    result = tool.execute(session, report_type="eda")

    assert result.success is True
    report_path = Path(result.data["report_path"])
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Exploratory Data Analysis Report" in content


def test_generate_model_report_tool(report_dataset: Path):
    session = SessionState()
    session.load_dataset(report_dataset)

    # Train model first
    trainer = TrainModelTool()
    trainer.execute(session, target_column="target")

    tool = GenerateReportTool()
    result = tool.execute(session, report_type="model")

    assert result.success is True
    report_path = Path(result.data["report_path"])
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Machine Learning Report" in content
