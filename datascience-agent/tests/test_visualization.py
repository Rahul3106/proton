"""
Tests for Phase 11 Visualization Agent tools (plot distribution, plot correlation, plot feature importance).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.context import SessionState
from proton_agent.tools import (
    PlotCorrelationMatrixTool,
    PlotDistributionTool,
    PlotFeatureImportanceTool,
    TrainModelTool,
)


@pytest.fixture
def viz_dataset(tmp_path: Path) -> Path:
    csv_file = tmp_path / "viz_data.csv"
    df = pd.DataFrame(
        {
            "val1": [10.0, 15.0, 12.0, 18.0, 25.0, 30.0],
            "val2": [100.0, 120.0, 110.0, 150.0, 200.0, 220.0],
            "label": [0, 0, 0, 1, 1, 1],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_plot_distribution_tool(viz_dataset: Path):
    session = SessionState()
    session.load_dataset(viz_dataset)

    tool = PlotDistributionTool()
    result = tool.execute(session, column="val1")

    assert result.success is True
    assert Path(result.data["plot_path"]).exists()


def test_plot_correlation_matrix_tool(viz_dataset: Path):
    session = SessionState()
    session.load_dataset(viz_dataset)

    tool = PlotCorrelationMatrixTool()
    result = tool.execute(session)

    assert result.success is True
    assert Path(result.data["plot_path"]).exists()


def test_plot_feature_importance_tool(viz_dataset: Path):
    session = SessionState()
    session.load_dataset(viz_dataset)

    # Train model first to get feature importances
    trainer = TrainModelTool()
    trainer.execute(session, target_column="label")

    tool = PlotFeatureImportanceTool()
    result = tool.execute(session)

    assert result.success is True
    assert Path(result.data["plot_path"]).exists()


def test_plot_distribution_non_existent_column(viz_dataset: Path):
    session = SessionState()
    session.load_dataset(viz_dataset)

    tool = PlotDistributionTool()
    result = tool.execute(session, user_request="plot distribution of age")

    assert result.success is False
    assert "Column 'age' not found" in result.error
