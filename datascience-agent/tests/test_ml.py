"""
Tests for Phase 9 Machine Learning Agent tools (detect problem, train, compare, predict).
"""
from pathlib import Path
import pandas as pd
import pytest

from proton_agent.agent import Agent, RuleLLMClient
from proton_agent.context import SessionState
from proton_agent.tools import (
    CompareModelsTool,
    DetectProblemTypeTool,
    PredictTool,
    TrainModelTool,
    default_registry,
)


@pytest.fixture
def ml_dataset(tmp_path: Path) -> Path:
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


def test_detect_problem_type(ml_dataset: Path):
    session = SessionState()
    session.load_dataset(ml_dataset)

    tool = DetectProblemTypeTool()
    result = tool.execute(session, target_column="churn")

    assert result.success is True
    assert result.data["problem_type"] == "classification"
    assert result.data["target_column"] == "churn"


def test_train_model_tool(ml_dataset: Path):
    session = SessionState()
    session.load_dataset(ml_dataset)

    tool = TrainModelTool()
    result = tool.execute(session, target_column="churn", model_type="random_forest")

    assert result.success is True
    assert "accuracy" in result.data["metrics"]
    assert session.current_model is not None


def test_compare_models_tool(ml_dataset: Path):
    session = SessionState()
    session.load_dataset(ml_dataset)

    tool = CompareModelsTool()
    result = tool.execute(session, target_column="churn")

    assert result.success is True
    assert result.data["candidate_count"] >= 3
    assert "selection_rationale" in result.data
    assert session.current_model is not None


def test_predict_tool(ml_dataset: Path):
    session = SessionState()
    session.load_dataset(ml_dataset)

    # Train model first
    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn")

    pred_tool = PredictTool()
    result = pred_tool.execute(session, n_samples=3)

    assert result.success is True
    assert len(result.data["predictions"]) == 3


def test_agent_ml_plan_execution(ml_dataset: Path):
    session = SessionState()
    session.load_dataset(ml_dataset)

    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)
    response = agent.process_request("train a model to predict churn", session)

    assert response.plan is not None
    assert response.plan.total_steps == 3
    assert response.plan.is_complete is True


def test_agent_specific_model_execution(ml_dataset: Path):
    session = SessionState()
    session.load_dataset(ml_dataset)

    agent = Agent(llm_client=RuleLLMClient(), registry=default_registry)
    response = agent.process_request("train decision tree", session)

    assert response.plan is not None
    assert response.plan.total_steps == 2
    assert response.plan.is_complete is True
    assert session.current_model["name"] == "Decision Tree Classifier"


def test_predict_for_named_model(ml_dataset: Path):
    session = SessionState()
    session.load_dataset(ml_dataset)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn", model_type="decision_tree")
    trainer.execute(session, target_column="churn", model_type="random_forest")

    # Current model is RF, but we ask to predict with decision tree
    pred_tool = PredictTool()
    result = pred_tool.execute(session, n_samples=3, model_name="decision_tree")

    assert result.success is True
    assert "Decision Tree" in result.data["model_name"]


def test_export_and_load_model(ml_dataset: Path, tmp_path: Path):
    from proton_agent.tools.modeling.export_model import ExportModelTool
    from proton_agent.tools.modeling.load_model import LoadModelTool

    session = SessionState()
    session.load_dataset(ml_dataset)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn", model_type="decision_tree")

    export_tool = ExportModelTool()
    exp_res = export_tool.execute(session, model_name="test_dt_model")
    assert exp_res.success is True
    saved_path = exp_res.data["saved_path"]

    # Clear session and reload
    session.current_model = None
    session.models.clear()

    load_tool = LoadModelTool()
    load_res = load_tool.execute(session, filepath=saved_path)
    assert load_res.success is True
    assert session.current_model is not None
    assert "Decision Tree" in session.current_model["name"]


def test_export_predictions(ml_dataset: Path, tmp_path: Path):
    from proton_agent.tools.modeling.export_predictions import ExportPredictionsTool

    session = SessionState()
    session.load_dataset(ml_dataset)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn", model_type="random_forest")

    exp_pred_tool = ExportPredictionsTool()
    res = exp_pred_tool.execute(session, filename="test_preds.csv")
    assert res.success is True
    assert Path(res.data["saved_path"]).exists()
    assert res.data["rows"] == 12


def test_explain_model(ml_dataset: Path):
    from proton_agent.tools.modeling.explain import ExplainModelTool

    session = SessionState()
    session.load_dataset(ml_dataset)

    trainer = TrainModelTool()
    trainer.execute(session, target_column="churn", model_type="random_forest")

    explain_tool = ExplainModelTool()
    res = explain_tool.execute(session)
    assert res.success is True
    assert "confusion_matrix" in res.data


def test_target_url_string_robustness(tmp_path: Path):
    """Verifies that non-numeric columns and URL columns are handled safely without string to float crashes."""
    import pandas as pd
    from proton_agent.tools.modeling.detect import DetectProblemTypeTool

    csv_file = tmp_path / "url_data.csv"
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "income": [50000, 60000, 75000, 90000, 110000],
        "adaptation_label_TARGET": ["Film", "Novel", "Film", "Novel", "Film"],
        "gutenberg_source_url": [
            "https://www.gutenberg.org/ebooks/1",
            "https://www.gutenberg.org/ebooks/2",
            "https://www.gutenberg.org/ebooks/3",
            "https://www.gutenberg.org/ebooks/4",
            "https://www.gutenberg.org/ebooks/5",
        ],
    })
    df.to_csv(csv_file, index=False)

    session = SessionState()
    session.load_dataset(csv_file)

    # 1. Detect target should automatically pick 'adaptation_label_TARGET', ignoring 'gutenberg_source_url'
    detect_tool = DetectProblemTypeTool()
    det_res = detect_tool.execute(session)
    assert det_res.success is True
    assert det_res.data["target_column"] == "adaptation_label_TARGET"
    assert det_res.data["problem_type"] == "classification"

    # 2. Train model executes successfully without string conversion crash
    trainer = TrainModelTool()
    train_res = trainer.execute(session, model_type="decision_tree")
    assert train_res.success is True
    assert train_res.data["problem_type"] == "classification"
