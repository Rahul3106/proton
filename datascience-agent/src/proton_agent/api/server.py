"""
FastAPI Server: RESTful & Server-Sent Events (SSE) API interface for Proton-Agent.
Exposes tool registry, agent multi-step planning, dataset ingestion, and session state over HTTP.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from fastapi.staticfiles import StaticFiles

import numpy as np
from proton_agent.api.manager import session_manager
from proton_agent.config.settings import get_settings
from proton_agent.context.dataset import (
    DatasetContext,
    DatasetNotFoundError,
    InvalidDatasetError,
    NoDatasetLoadedError,
)
from proton_agent.tools import default_registry
from proton_agent.tools.modeling.recommend import get_model_recommendations


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively converts NumPy, Pandas, NaN, and complex types to valid JSON primitives."""
    if obj is None:
        return None
    if isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj) or np.isneginf(obj) or np.isposinf(obj):
            return None
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize_for_json(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _sanitize_for_json(obj.to_dict())
        except Exception:
            return str(obj)
    return str(obj)


def safe_json_dumps(obj: Any) -> str:
    """Safe wrapper around json.dumps that never fails on NumPy/Pandas objects."""
    sanitized = _sanitize_for_json(obj)
    return json.dumps(sanitized, default=str)


app = FastAPI(
    title="Proton Data Science Agent API",
    description="REST & SSE HTTP service exposing Proton Agent tools, planning, and session state.",
    version="1.0.0",
)

# Enable CORS for browser frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web assets
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── REQUEST / RESPONSE MODELS ──────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Optional custom session ID")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language data science query or command")


class SetTargetRequest(BaseModel):
    target_column: Optional[str] = Field(None, description="Name of the selected target column")



# ── HEALTH, WEB APP & CONFIG ENDPOINTS ─────────────────────────────────────────

@app.get("/favicon.ico")
def serve_favicon() -> FileResponse:
    """Serves the vector SVG favicon for browser tabs."""
    fav_file = STATIC_DIR / "favicon.svg"
    if fav_file.exists():
        return FileResponse(fav_file, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found.")


@app.get("/")
def serve_index_or_health(request: Request) -> Any:
    """Serves the interactive terminal web shell for browsers or JSON health check for API clients."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept or "application/xhtml+xml" in accept:
        index_file = STATIC_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file, media_type="text/html")
    return health_check()



@app.get("/health")
def health_check() -> dict[str, Any]:

    settings = get_settings()
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    key_masked = f"...{api_key[-6:]}" if len(api_key) > 6 else ("Active" if api_key else "Not Configured")
    return {
        "service": "Proton Data Science Agent API",
        "status": "online",
        "version": "1.0.0",
        "llm_provider": settings.llm_provider,
        "llm_model": settings.gemini_model,
        "api_key_status": key_masked,
        "active_sessions": len(session_manager.list_sessions()),
        "registered_tools": len(default_registry.list_tools()),
    }


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    settings = get_settings()
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    key_masked = f"...{api_key[-6:]}" if len(api_key) > 6 else ("Active" if api_key else "Not Configured")
    return {
        "llm_provider": settings.llm_provider,
        "gemini_model": settings.gemini_model,
        "temperature": settings.temperature,
        "api_key_status": key_masked,
        "output_dir": str(settings.output_dir),
        "plots_dir": str(settings.plots_dir),
        "reports_dir": str(settings.reports_dir),
        "models_dir": str(settings.models_dir),
    }


# ── SESSION LIFECYCLE ──────────────────────────────────────────────────────────

@app.post("/api/session", status_code=status.HTTP_201_CREATED)
def create_session(req: Optional[CreateSessionRequest] = None) -> dict[str, Any]:
    sid = req.session_id if req else None
    created_id = session_manager.create_session(sid)
    return {
        "success": True,
        "session_id": created_id,
        "message": f"Session '{created_id}' initialized successfully.",
    }


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    sids = session_manager.list_sessions()
    return {
        "count": len(sids),
        "sessions": sids,
    }


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    return session_manager.get_session_summary(session_id)


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    deleted = session_manager.delete_session(session_id)
    return {
        "success": deleted,
        "session_id": session_id,
        "message": f"Session '{session_id}' cleared." if deleted else "Session not found.",
    }


# ── DATA INGESTION & UPLOAD ───────────────────────────────────────────────────

def _load_dataset_file(filepath: Path) -> tuple[pd.DataFrame, str]:
    """Helper to load dataframe with multi-format and encoding detection."""
    ext = filepath.suffix.lower()
    encoding_used = "utf-8"

    if ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
        df = pd.read_excel(filepath)
    elif ext == ".parquet":
        df = pd.read_parquet(filepath)
    elif ext == ".json":
        df = pd.read_json(filepath)
    elif ext in (".tsv", ".tab"):
        df = pd.read_csv(filepath, sep="\t")
    else:
        try:
            df = pd.read_csv(filepath, encoding="utf-8")
            encoding_used = "utf-8"
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, encoding="latin-1")
                encoding_used = "latin-1 (fallback)"
            except Exception:
                df = pd.read_csv(filepath, encoding="cp1252")
                encoding_used = "cp1252 (fallback)"

    return df, encoding_used


@app.post("/api/session/{session_id}/load")
async def load_dataset(
    session_id: str,
    request: Request,
) -> dict[str, Any]:
    """
    Ingests a dataset into the specified session.
    Accepts multipart file uploads (browser drag & drop) or JSON filepath bodies.
    """
    session = session_manager.get_session(session_id)
    content_type = request.headers.get("content-type", "")

    resolved_path: Optional[Path] = None
    is_test: bool = False

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        filepath = form.get("filepath")
        dtype = str(form.get("dataset_type", "train")).lower()
        is_test = dtype == "test"

        if file and hasattr(file, "filename") and file.filename:
            upload_dir = Path(f"outputs/uploads/{session_id}")
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest_path = upload_dir / file.filename
            content = await file.read()
            dest_path.write_bytes(content)
            resolved_path = dest_path.resolve()
        elif filepath:
            resolved_path = Path(str(filepath)).expanduser().resolve()
    else:
        try:
            body = await request.json()
        except Exception:
            body = {}
        filepath = body.get("filepath") or request.query_params.get("filepath")
        dtype = str(body.get("dataset_type") or request.query_params.get("dataset_type", "train")).lower()
        is_test = dtype == "test"
        if filepath:
            resolved_path = Path(str(filepath)).expanduser().resolve()

    if not resolved_path:
        raise HTTPException(status_code=400, detail="Must provide either a file upload or a filepath.")

    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: '{resolved_path}'")

    try:
        df, encoding_used = _load_dataset_file(resolved_path)
        ctx = DatasetContext(filepath=resolved_path, df=df)

        if is_test:
            session.test_dataset = ctx
            session.history.append(f"load test {resolved_path.name}")
        else:
            session.current_dataset = ctx
            session.history.append(f"load {resolved_path.name}")
            session.transformations.clear()
            session.target_column = None
            session.models.clear()
            session.current_model = None

        sample_preview = df.head(5).fillna("").to_dict(orient="records")

        return {
            "success": True,
            "name": ctx.name,
            "rows": ctx.rows,
            "columns": ctx.columns_count,
            "columns_list": ctx.columns,
            "dtypes": ctx.dtypes,
            "total_missing": ctx.total_missing,
            "filepath": str(ctx.filepath),
            "is_test": is_test,
            "encoding_used": encoding_used,
            "sample_preview": sample_preview,
            "message": f"Successfully loaded {'test' if is_test else 'train'} dataset '{ctx.name}' ({ctx.rows:,} rows × {ctx.columns_count} cols).",
        }
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {err}")


# ── TOOL REGISTRY & DYNAMIC DISPATCHER ─────────────────────────────────────────

@app.get("/api/tools")
def list_registered_tools() -> dict[str, Any]:
    """Returns all tools registered in default_registry with their descriptions."""
    tools = default_registry.list_tools()
    return {
        "count": len(tools),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
            }
            for t in tools
        ],
    }


@app.post("/api/session/{session_id}/tool/{tool_name}")
async def execute_tool(
    session_id: str,
    tool_name: str,
    request: Request,
) -> dict[str, Any]:
    """
    Dynamic generic dispatcher: executes any registered tool against the session.
    Accepts arbitrary JSON kwargs in request body.
    """
    session = session_manager.get_session(session_id)
    tool = default_registry.get(tool_name)

    if not tool:
        all_tools = [t.name for t in default_registry.list_tools()]
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found. Available tools: {', '.join(all_tools)}",
        )

    # Parse body kwargs if provided
    kwargs: dict[str, Any] = {}
    try:
        body = await request.json()
        if isinstance(body, dict):
            kwargs = body.get("params", body) if "params" in body and isinstance(body["params"], dict) else body
    except Exception:
        pass

    try:
        result = tool.execute(session, **kwargs)
        return {
            "success": result.success,
            "tool_name": tool_name,
            "data": result.data,
            "error": result.error,
        }
    except Exception as err:
        return {
            "success": False,
            "tool_name": tool_name,
            "data": {},
            "error": str(err),
        }


# ── AGENT NATURAL LANGUAGE QUERY / CHAT ────────────────────────────────────────

@app.post("/api/session/{session_id}/query")
def process_agent_query(session_id: str, req: QueryRequest) -> dict[str, Any]:
    """
    Executes an autonomous Agent turn for the session.
    Constructs multi-step plan, executes steps, and returns full structured response.
    """
    session = session_manager.get_session(session_id)
    agent = session_manager.get_agent(session_id)

    try:
        response = agent.process_request(req.query, session)

        plan_data = None
        if response.plan:
            plan_data = {
                "goal": response.plan.goal,
                "total_steps": response.plan.total_steps,
                "completed_steps": response.plan.completed_steps,
                "is_complete": response.plan.is_complete,
                "steps": [
                    {
                        "step_number": s.step_number,
                        "description": s.description,
                        "tool_name": s.tool_name,
                        "completed": s.completed,
                        "result": {
                            "success": s.result.success,
                            "data": s.result.data,
                            "error": s.result.error,
                        } if s.result else None,
                    }
                    for s in response.plan.steps
                ],
            }

        return {
            "user_request": response.user_request,
            "selected_tool": response.selected_tool,
            "explanation": response.explanation,
            "summary": response.summary,
            "plan": plan_data,
            "tool_result": {
                "success": response.tool_result.success,
                "data": response.tool_result.data,
                "error": response.tool_result.error,
            } if response.tool_result else None,
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))


# ── SSE LIVE STREAMING FOR PROGRESS CHECKLISTS ─────────────────────────────────

@app.get("/api/session/{session_id}/stream")
def stream_agent_query(session_id: str, query: str = Query(..., description="Query to execute with live streaming")) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint: streams real-time step execution events
    so the web UI can render the live interactive checklist (✓ Step 1, ✓ Step 2...).
    """
    session = session_manager.get_session(session_id)
    agent = session_manager.get_agent(session_id)

    def event_generator():
        try:
            plan = agent.planner.create_plan(query)

            if not plan.steps:
                custom_resp = agent._handle_custom_question(query, session)
                yield f"event: qa_response\ndata: {safe_json_dumps({'explanation': custom_resp.explanation})}\n\n"
                return

            yield f"event: plan_created\ndata: {safe_json_dumps({'goal': plan.goal, 'total_steps': plan.total_steps, 'steps': [{'step_number': s.step_number, 'description': s.description, 'tool_name': s.tool_name} for s in plan.steps]})}\n\n"

            for step in plan.steps:
                yield f"event: step_start\ndata: {safe_json_dumps({'step_number': step.step_number, 'description': step.description, 'tool_name': step.tool_name})}\n\n"

                try:
                    step.result = agent.executor.execute_tool(step.tool_name, session, user_request=query, **step.tool_kwargs)
                    if step.result.success:
                        step.completed = True
                        yield f"event: step_complete\ndata: {safe_json_dumps({'step_number': step.step_number, 'tool_name': step.tool_name, 'success': True, 'data': step.result.data})}\n\n"
                    else:
                        yield f"event: step_complete\ndata: {safe_json_dumps({'step_number': step.step_number, 'tool_name': step.tool_name, 'success': False, 'error': step.result.error})}\n\n"
                        break
                except Exception as step_err:
                    yield f"event: step_complete\ndata: {safe_json_dumps({'step_number': step.step_number, 'tool_name': step.tool_name, 'success': False, 'error': str(step_err)})}\n\n"
                    break

            plan.is_complete = all(s.completed for s in plan.steps)
            summary = agent._summarize_plan_findings(plan)
            failed_step = next((s for s in plan.steps if s.result and not s.result.success), None)
            if failed_step:
                diagnosis = agent._diagnose_failure(failed_step, session, query)
                if diagnosis:
                    summary = (summary + f"\n\n🤖 AI Diagnostic & Recommended Action:\n{diagnosis}").strip()

            yield f"event: plan_complete\ndata: {safe_json_dumps({'is_complete': plan.is_complete, 'completed_steps': plan.completed_steps, 'total_steps': plan.total_steps, 'summary': summary})}\n\n"
        except Exception as plan_err:
            yield f"event: plan_complete\ndata: {safe_json_dumps({'is_complete': False, 'completed_steps': 0, 'total_steps': 0, 'summary': f'Execution error: {plan_err}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",

        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── ARTIFACT SERVING ──────────────────────────────────────────────────────────

@app.get("/api/artifacts/{path:path}")
def serve_artifact(path: str) -> FileResponse:
    """
    Serves generated plots, reports, models, and predictions safely from the outputs directory.
    """
    clean_path = Path(path).as_posix()
    if ".." in clean_path or clean_path.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid artifact path.")

    candidate_paths = [
        Path(clean_path),
        Path("outputs") / clean_path,
        Path("outputs/plots") / clean_path,
        Path("outputs/reports") / clean_path,
        Path("outputs/models") / clean_path,
    ]

    found_file = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            found_file = p.resolve()
            break

    if not found_file:
        raise HTTPException(status_code=404, detail=f"Artifact not found: '{path}'")

    ext = found_file.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".html": "text/html",
        ".pkl": "application/octet-stream",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        found_file,
        media_type=media_type,
        filename=found_file.name,
    )


# ── DATASET SAMPLE EXPLORATION (PHASE 4) ──────────────────────────────────────

@app.get("/api/session/{session_id}/sample")
def get_dataset_sample(
    session_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    is_test: bool = False,
) -> dict[str, Any]:
    """Returns paginated dataset records for the interactive data viewer."""
    session = session_manager.get_session(session_id)
    dataset = session.test_dataset if is_test else session.current_dataset

    if not dataset or dataset.df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded in this session.")

    df = dataset.df
    total_rows = len(df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)

    subset = df.iloc[start_idx:end_idx].fillna("")
    records = subset.to_dict(orient="records")

    return {
        "dataset_name": dataset.name,
        "is_test": is_test,
        "total_rows": total_rows,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "records": records,
    }


@app.post("/api/session/{session_id}/target-column")
def set_target_column(session_id: str, req: SetTargetRequest) -> dict[str, Any]:
    """Updates the active target column in the session state."""
    session = session_manager.get_session(session_id)
    session.target_column = req.target_column.strip() if req.target_column else None
    return {
        "success": True,
        "target_column": session.target_column,
        "message": f"Active target column set to '{session.target_column}'." if session.target_column else "Target column cleared.",
    }


@app.get("/api/session/{session_id}/suggest-models")
def get_suggested_models(
    session_id: str,
    target_column: Optional[str] = Query(None, description="Optional target column name"),
) -> dict[str, Any]:
    """
    Intelligently analyzes dataset size, feature distribution, and target variable
    to return the Top 5 recommended models and the full categorized algorithm catalog.
    """
    session = session_manager.get_session(session_id)
    recommendations = get_model_recommendations(session, target_column=target_column)
    return {
        "success": True,
        **recommendations,
    }



# ── DATASET REVERT (PHASE 5 GUARDRAIL) ────────────────────────────────────────


@app.post("/api/session/{session_id}/revert")
def revert_session_dataset(session_id: str) -> dict[str, Any]:
    """Reverts the session's active dataset to its pristine state upon initial upload."""
    session = session_manager.get_session(session_id)
    try:
        reverted_ds = session.revert_dataset()
        return {
            "success": True,
            "name": reverted_ds.name,
            "rows": reverted_ds.rows,
            "columns": reverted_ds.columns_count,
            "message": f"Successfully reverted '{reverted_ds.name}' to original unmodified state.",
        }
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


# ── SINGLE-ROW INTERACTIVE PREDICTION & EXPLANATION (PHASE 6 & PHASE C) ───────

class PredictRowRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Dictionary of feature names and input values")
    include_explanation: bool = Field(False, description="Whether to include per-row feature explanation")


class ExplainPredictionRequest(BaseModel):
    features: Optional[Dict[str, Any]] = Field(None, description="Dictionary of feature names and input values")
    row_index: Optional[int] = Field(None, description="Optional zero-based row index from active dataset")


@app.post("/api/session/{session_id}/predict-row")
def predict_single_row(session_id: str, req: PredictRowRequest) -> dict[str, Any]:
    """Executes single-row inference against the active trained model, with optional explanation."""
    session = session_manager.get_session(session_id)

    if not session.current_model:
        raise HTTPException(
            status_code=400,
            detail="No active trained model found in session. Train a model first via 'Train Model' or 'Compare Models'.",
        )

    model_record = session.current_model
    model = model_record["model"]
    feature_names = model_record.get("feature_names", [])
    target_col = model_record.get("target_column", "Prediction")

    try:
        input_df = pd.DataFrame([req.features])

        # Handle categorical encoding for input
        cat_cols = input_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if cat_cols:
            input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True, dtype=float)
        else:
            input_encoded = input_df.copy()

        # Align with model trained feature schema
        aligned_df = input_encoded.reindex(columns=feature_names, fill_value=0.0)

        # Fill any NaNs
        for col in aligned_df.columns:
            aligned_df[col] = aligned_df[col].fillna(0.0)

        prediction_val = model.predict(aligned_df)[0]

        # Probabilities if classification
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probs = model.predict_proba(aligned_df)[0]
                classes = getattr(model, "classes_", [str(i) for i in range(len(probs))])
                probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
            except Exception:
                pass

        explanation_data = None
        if req.include_explanation:
            from proton_agent.tools.modeling.explain_prediction import ExplainPredictionTool
            tool = ExplainPredictionTool()
            res = tool.execute(session, features=req.features)
            if res.success:
                explanation_data = res.data

        return {
            "success": True,
            "model_name": model_record.get("name", "Model"),
            "target_column": target_col,
            "prediction": str(prediction_val),
            "probabilities": probabilities,
            "features_used": req.features,
            "explanation": explanation_data,
            "message": f"Predicted '{prediction_val}' for target '{target_col}' using {model_record.get('name')}.",
        }
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Prediction error: {err}")


@app.post("/api/session/{session_id}/explain-prediction")
def explain_single_row_prediction(session_id: str, req: Optional[ExplainPredictionRequest] = None) -> dict[str, Any]:
    """
    Computes a structured instance-level explanation (local SHAP values and feature contributions)
    for a specific prediction row.
    """
    session = session_manager.get_session(session_id)
    if not session.current_model:
        raise HTTPException(
            status_code=400,
            detail="No active trained model found in session. Train a model first via 'Train Model' or 'Compare Models'.",
        )

    from proton_agent.tools.modeling.explain_prediction import ExplainPredictionTool

    tool = ExplainPredictionTool()
    features = req.features if req else None
    row_index = req.row_index if req else None

    result = tool.execute(session, features=features, row_index=row_index)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)

    return {
        "success": True,
        **result.data,
    }



# ── SESSION STATE EXPORT & IMPORT (PHASE 7 PERSISTENCE) ────────────────────────

@app.get("/api/session/{session_id}/export-state")
def export_session_state(session_id: str) -> dict[str, Any]:
    """Exports full session state, history, transformations, and experiment runs to JSON."""
    summary = session_manager.get_session_summary(session_id)
    session = session_manager.get_session(session_id)
    summary["experiments"] = [r.to_dict() for r in session.experiments.list_runs()]
    summary["version"] = "1.0.0"
    return summary


@app.post("/api/session/{session_id}/import-state")
def import_session_state(session_id: str, state: Dict[str, Any]) -> dict[str, Any]:
    """Restores session state from exported JSON."""
    session = session_manager.get_session(session_id)
    if "transformations" in state and isinstance(state["transformations"], list):
        session.transformations = state["transformations"]
    if "history" in state and isinstance(state["history"], list):
        session.history = state["history"]
    if "target_column" in state:
        session.target_column = state["target_column"]

    return {
        "success": True,
        "session_id": session_id,
        "message": f"Successfully restored session '{session_id}' state.",
    }


# ── EXPERIMENT TRACKING ───────────────────────────────────────────────────────

@app.get("/api/session/{session_id}/experiments")
def get_session_experiments(session_id: str) -> dict[str, Any]:
    session = session_manager.get_session(session_id)
    runs = session.experiments.list_runs()
    return {
        "count": len(runs),
        "experiments": [r.to_dict() for r in runs],
    }


@app.post("/api/session/{session_id}/detect-drift")
def detect_session_drift(
    session_id: str,
    reference_path: str = "",
    current_path: str = "",
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """Detects statistical data drift and covariate shift across session features."""
    session = session_manager.get_session(session_id)
    tool = default_registry.get("detect_drift")
    if not tool:
        raise HTTPException(status_code=500, detail="DriftDetectionTool not registered.")
    res = tool.execute(
        session,
        reference_path=reference_path,
        current_path=current_path,
        significance_level=significance_level,
    )
    if not res.success:
        raise HTTPException(status_code=400, detail=res.error)
    return res.data




# Backwards compatibility legacy endpoints
@app.post("/api/load")
async def legacy_load(request: Request) -> dict[str, Any]:
    return await load_dataset("default", request)


@app.get("/api/inspect")
def legacy_inspect() -> dict[str, Any]:
    session = session_manager.get_session("default")
    try:
        dataset = session.inspect_dataset()
        return {
            "name": dataset.name,
            "rows": dataset.rows,
            "columns_count": dataset.columns_count,
            "columns": dataset.columns,
            "dtypes": dataset.dtypes,
            "total_missing": dataset.total_missing,
        }
    except Exception as err:
        raise HTTPException(status_code=404, detail=str(err))


@app.post("/api/query")
def legacy_query(req: QueryRequest) -> dict[str, Any]:
    return process_agent_query("default", req)


@app.get("/api/experiments")
def legacy_experiments() -> dict[str, Any]:
    return get_session_experiments("default")
