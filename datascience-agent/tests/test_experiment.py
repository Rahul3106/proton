"""
Tests for Phase 10 Experiment Management (ExperimentTracker, ExperimentRun, logging).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.context import ExperimentTracker, SessionState
from proton_agent.tools import TrainModelTool


@pytest.fixture
def exp_dataset(tmp_path: Path) -> Path:
    csv_file = tmp_path / "exp_data.csv"
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10, 20, 30, 40, 50, 60],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_experiment_tracker_manual_logging():
    tracker = ExperimentTracker()
    run = tracker.log_run(
        dataset_name="test.csv",
        dataset_rows=100,
        dataset_cols=5,
        target_column="target",
        feature_names=["f1", "f2", "f3", "f4"],
        model_name="Random Forest",
        problem_type="classification",
        metrics={"accuracy": 0.95, "f1_score": 0.94},
    )

    assert run.id is not None
    assert len(tracker.list_runs()) == 1
    assert tracker.get_best_run("f1_score").model_name == "Random Forest"


def test_automated_experiment_logging_on_train(exp_dataset: Path):
    session = SessionState()
    session.load_dataset(exp_dataset)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="target", model_type="random_forest")

    assert len(session.experiments.list_runs()) == 1
    run = session.experiments.list_runs()[0]
    assert run.target_column == "target"
    assert "accuracy" in run.metrics


def test_experiment_dataframe_export():
    tracker = ExperimentTracker()
    tracker.log_run(
        dataset_name="data1.csv",
        dataset_rows=50,
        dataset_cols=4,
        target_column="y",
        feature_names=["x1", "x2"],
        model_name="Linear Regression",
        problem_type="regression",
        metrics={"rmse": 2.5, "r2_score": 0.88},
    )

    df_exp = tracker.to_dataframe()
    assert not df_exp.empty
    assert "metric_rmse" in df_exp.columns
