"""
LoadModelTool: loads a saved model specification from outputs/models/ into the active session.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from proton_agent.context.session import SessionState
from proton_agent.tools.base import BaseTool, ToolResult


class LoadModelTool(BaseTool):
    """Loads a previously exported model file (.txt or .pkl) from disk into the active session."""

    name: str = "load_model"
    description: str = "Load a saved model file from disk into the active session for predictions."

    def execute(self, session: SessionState, filepath: str = "", **kwargs: Any) -> ToolResult:
        if not filepath:
            models_dir = Path("outputs/models")
            if models_dir.exists():
                model_files = sorted(
                    list(models_dir.glob("*.txt")) + list(models_dir.glob("*.pkl")),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if model_files:
                    filepath = str(model_files[0])
                else:
                    return ToolResult(
                        success=False,
                        data={},
                        error="No model files found in 'outputs/models/'. Export a model first.",
                    )
            else:
                return ToolResult(
                    success=False,
                    data={},
                    error="No model save directory found. Train and export a model first.",
                )

        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            return ToolResult(
                success=False,
                data={},
                error=f"Model file not found: '{filepath}'",
            )

        if path.suffix == ".pkl":
            try:
                import joblib
                model_record = joblib.load(path)
            except Exception as err:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Failed to load binary model from '{filepath}': {err}",
                )
        else:
            # Parse model metadata from .txt specification
            try:
                txt_content = path.read_text(encoding="utf-8")
                model_name = path.stem.replace("_", " ").title()
                target_col = None
                problem_type = None

                for line in txt_content.splitlines():
                    if "Model Name" in line and ":" in line:
                        model_name = line.split(":", 1)[1].strip()
                    elif "Target Column" in line and ":" in line:
                        target_col = line.split(":", 1)[1].strip()
                    elif "Problem Type" in line and ":" in line:
                        problem_type = line.split(":", 1)[1].strip().lower()

                model_record = {
                    "name": model_name,
                    "target_column": target_col,
                    "problem_type": problem_type,
                    "feature_names": [],
                    "metrics": {},
                }
            except Exception as err:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Failed to load model file '{filepath}': {err}",
                )

        session.current_model = model_record
        if isinstance(model_record, dict):
            model_name = model_record.get("name", path.stem)
            session.models[model_name] = model_record
            session.target_column = model_record.get("target_column")
        else:
            model_name = path.stem

        return ToolResult(
            success=True,
            data={
                "loaded_path": str(path),
                "model_name": model_name,
                "target_column": model_record.get("target_column") if isinstance(model_record, dict) else None,
                "problem_type": model_record.get("problem_type") if isinstance(model_record, dict) else None,
                "message": f"✓ Model '{model_name}' loaded from '{path.name}'.",
            },
        )
