"""
Comprehensive Tests for Phase 1 REST & SSE API endpoints.
Verifies all 20+ Proton tools, session management, file uploads, dynamic tool dispatcher,
agent queries, SSE streaming, and artifact serving.
"""
import io
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from proton_agent.api import app


@pytest.fixture
def api_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "api_data.csv"
    df = pd.DataFrame(
        {
            "age": [25, 30, 45, 50, 23, 34, 40, 60, 28, 55, 38, 42],
            "income": [50000.0, 60000.0, 75000.0, 90000.0, 45000.0, 52000.0, 80000.0, 110000.0, 48000.0, 95000.0, 72000.0, 85000.0],
            "target": [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def test_csv(tmp_path: Path) -> Path:
    csv_file = tmp_path / "test_api_data.csv"
    df = pd.DataFrame(
        {
            "age": [29, 35, 48],
            "income": [58000.0, 64000.0, 82000.0],
            "target": [0, 1, 1],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


def test_api_root_and_health():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["registered_tools"] > 15


def test_web_shell_html_and_static_serving():
    client = TestClient(app)

    # HTML Shell for browser
    html_resp = client.get("/", headers={"Accept": "text/html"})
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers["content-type"]
    assert "PROTON" in html_resp.text
    assert "System Status & Engine Capabilities" in html_resp.text

    # Static CSS
    css_resp = client.get("/static/index.css")
    assert css_resp.status_code == 200
    assert "text/css" in css_resp.headers["content-type"]

    # Static JS
    js_resp = client.get("/static/app.js")
    assert js_resp.status_code == 200



def test_api_config():
    client = TestClient(app)
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "llm_provider" in data
    assert "api_key_status" in data


def test_session_lifecycle():
    client = TestClient(app)

    # Create session
    create_resp = client.post("/api/session", json={"session_id": "test-sess-1"})
    assert create_resp.status_code == 201
    assert create_resp.json()["session_id"] == "test-sess-1"

    # Get session summary
    get_resp = client.get("/api/session/test-sess-1")
    assert get_resp.status_code == 200
    assert get_resp.json()["has_dataset"] is False

    # List sessions
    list_resp = client.get("/api/sessions")
    assert list_resp.status_code == 200
    assert "test-sess-1" in list_resp.json()["sessions"]

    # Delete session
    del_resp = client.delete("/api/session/test-sess-1")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True


def test_load_dataset_filepath(api_csv: Path, test_csv: Path):
    client = TestClient(app)
    sid = "load-test-session"

    # Load training data
    load_train = client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv), "dataset_type": "train"})
    assert load_train.status_code == 200
    assert load_train.json()["rows"] == 12
    assert load_train.json()["is_test"] is False

    # Load test data
    load_test = client.post(f"/api/session/{sid}/load", json={"filepath": str(test_csv), "dataset_type": "test"})
    assert load_test.status_code == 200
    assert load_test.json()["rows"] == 3
    assert load_test.json()["is_test"] is True

    # Check session state
    sess_resp = client.get(f"/api/session/{sid}")
    assert sess_resp.json()["has_dataset"] is True
    assert sess_resp.json()["has_test_dataset"] is True


def test_load_dataset_multipart_upload():
    client = TestClient(app)
    sid = "upload-multipart-session"

    csv_content = b"col1,col2,target\n1,10.5,0\n2,20.5,1\n3,30.5,0\n"
    files = {"file": ("uploaded_data.csv", io.BytesIO(csv_content), "text/csv")}

    resp = client.post(f"/api/session/{sid}/load", files=files, data={"dataset_type": "train"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["rows"] == 3
    assert data["columns"] == 3
    assert "encoding_used" in data


def test_list_tools():
    client = TestClient(app)
    resp = client.get("/api/tools")
    assert resp.status_code == 200
    tools = resp.json()["tools"]
    tool_names = [t["name"] for t in tools]

    assert "inspect_dataset" in tool_names
    assert "train_model" in tool_names
    assert "compare_models" in tool_names
    assert "explain_model" in tool_names
    assert "export_model" in tool_names
    assert "clean_dataset" in tool_names


def test_dynamic_tool_dispatcher(api_csv: Path):
    client = TestClient(app)
    sid = "dispatcher-test-session"

    # Ingest data
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    # Test tool 1: inspect_dataset
    res1 = client.post(f"/api/session/{sid}/tool/inspect_dataset", json={})
    assert res1.status_code == 200
    assert res1.json()["success"] is True
    assert res1.json()["data"]["rows"] == 12

    # Test tool 2: calculate_statistics
    res2 = client.post(f"/api/session/{sid}/tool/calculate_statistics", json={})
    assert res2.status_code == 200
    assert res2.json()["success"] is True

    # Test tool 3: detect_missing_values
    res3 = client.post(f"/api/session/{sid}/tool/detect_missing_values", json={})
    assert res3.status_code == 200
    assert res3.json()["success"] is True
    assert res3.json()["data"]["total_missing"] == 0

    # Test tool 4: train_model (Decision Tree)
    res4 = client.post(f"/api/session/{sid}/tool/train_model", json={"target_column": "target", "model_type": "decision_tree"})
    assert res4.status_code == 200
    assert res4.json()["success"] is True
    assert "accuracy" in res4.json()["data"]["metrics"]

    # Test tool 5: predict
    res5 = client.post(f"/api/session/{sid}/tool/predict", json={"n_samples": 3})
    assert res5.status_code == 200
    assert res5.json()["success"] is True
    assert len(res5.json()["data"]["predictions"]) == 3

    # Test tool 6: explain_model
    res6 = client.post(f"/api/session/{sid}/tool/explain_model", json={})
    assert res6.status_code == 200
    assert res6.json()["success"] is True
    assert "confusion_matrix" in res6.json()["data"]

    # Test tool 7: export_model
    res7 = client.post(f"/api/session/{sid}/tool/export_model", json={"model_name": "api_test_model"})
    assert res7.status_code == 200
    assert res7.json()["success"] is True
    assert "saved_path" in res7.json()["data"]

    # Test tool 8: export_predictions
    res8 = client.post(f"/api/session/{sid}/tool/export_predictions", json={"filename": "api_preds.csv"})
    assert res8.status_code == 200
    assert res8.json()["success"] is True

    # Test unknown tool -> 404
    res_err = client.post(f"/api/session/{sid}/tool/non_existent_tool", json={})
    assert res_err.status_code == 404


def test_agent_query_endpoint(api_csv: Path):
    client = TestClient(app)
    sid = "query-test-session"
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    # Natural language query
    resp = client.post(f"/api/session/{sid}/query", json={"query": "analyze this dataset"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["plan"] is not None
    assert data["plan"]["is_complete"] is True
    assert data["plan"]["total_steps"] == 7
    assert "Dataset Overview" in data["summary"]


def test_sse_streaming_endpoint(api_csv: Path):
    client = TestClient(app)
    sid = "stream-test-session"
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    # Request SSE stream
    response = client.get(f"/api/session/{sid}/stream?query=detect%20missing%20values")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    event_text = response.text
    assert "plan_created" in event_text
    assert "step_start" in event_text
    assert "step_complete" in event_text
    assert "plan_complete" in event_text



def test_artifact_serving(tmp_path: Path):
    client = TestClient(app)

    # Create dummy artifact in outputs/plots/
    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    test_plot = plots_dir / "test_artifact_plot.png"
    test_plot.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    resp = client.get("/api/artifacts/plots/test_artifact_plot.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"

    # Test invalid path traversal -> 400 or 404
    bad_resp = client.get("/api/artifacts/../../etc/passwd")
    assert bad_resp.status_code in (400, 404)


def test_dataset_sample_endpoint(api_csv: Path):
    client = TestClient(app)
    sid = "sample-test-session"
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    resp = client.get(f"/api/session/{sid}/sample?page=1&page_size=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_rows"] == 12
    assert len(data["records"]) == 5
    assert data["page"] == 1
    assert data["total_pages"] == 3


def test_dataset_revert_endpoint(api_csv: Path):
    client = TestClient(app)
    sid = "revert-test-session"
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    # Mutate data via clean_dataset
    client.post(f"/api/session/{sid}/tool/clean_dataset", json={"params": {"dry_run": False}})


    # Revert dataset
    rev_resp = client.post(f"/api/session/{sid}/revert")
    assert rev_resp.status_code == 200
    assert rev_resp.json()["success"] is True
    assert "reverted" in rev_resp.json()["message"].lower()


def test_single_row_prediction_endpoint(api_csv: Path):
    client = TestClient(app)
    sid = "singlerow-test-session"
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    # Train model
    client.post(f"/api/session/{sid}/tool/train_model", json={"params": {"target_column": "target", "model_type": "decision_tree"}})

    # Predict single row
    pred_resp = client.post(f"/api/session/{sid}/predict-row", json={"features": {"age": 35, "income": 65000.0}})
    assert pred_resp.status_code == 200
    data = pred_resp.json()
    assert data["success"] is True
    assert data["target_column"] == "target"
    assert data["prediction"] in ("0", "1")


def test_session_state_export_and_import(api_csv: Path):
    client = TestClient(app)
    sid = "persist-test-session"
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    # Export state
    exp_resp = client.get(f"/api/session/{sid}/export-state")
    assert exp_resp.status_code == 200
    state_data = exp_resp.json()
    assert state_data["has_dataset"] is True

    # Import into another session
    sid2 = "restored-session"
    imp_resp = client.post(f"/api/session/{sid2}/import-state", json=state_data)
    assert imp_resp.status_code == 200
    assert imp_resp.json()["success"] is True


def test_set_target_column_endpoint(api_csv: Path):
    client = TestClient(app)
    sid = "target-test-session"
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    # Set target column
    resp = client.post(f"/api/session/{sid}/target-column", json={"target_column": "target"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["target_column"] == "target"


def test_suggest_models_endpoint(api_csv: Path):
    client = TestClient(app)
    sid = "suggest-models-session"
    client.post(f"/api/session/{sid}/load", json={"filepath": str(api_csv)})

    # Request model recommendations
    resp = client.get(f"/api/session/{sid}/suggest-models?target_column=target")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["problem_type"] == "classification"
    assert len(data["top_5_models"]) == 5
    assert data["top_5_models"][0]["model_type"] == "random_forest"
    assert "all_categories" in data




