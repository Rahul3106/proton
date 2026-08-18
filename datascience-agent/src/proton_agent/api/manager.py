"""
SessionManager: Manages in-memory SessionState and Agent instances across HTTP sessions.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from proton_agent.agent import Agent
from proton_agent.context.session import SessionState
from proton_agent.tools import default_registry


class SessionManager:
    """Manages active sessions for the REST & SSE API."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._agents: Dict[str, Agent] = {}

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Creates a new session and returns its ID."""
        sid = session_id.strip() if session_id and session_id.strip() else str(uuid.uuid4())[:8]
        self._sessions[sid] = SessionState()
        self._agents[sid] = Agent(registry=default_registry)
        return sid

    def get_session(self, session_id: str) -> SessionState:
        """Retrieves or auto-creates a session state for the given session ID."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
            self._agents[session_id] = Agent(registry=default_registry)
        return self._sessions[session_id]

    def get_agent(self, session_id: str) -> Agent:
        """Retrieves or auto-creates an Agent instance for the given session ID."""
        if session_id not in self._agents:
            self._sessions[session_id] = SessionState()
            self._agents[session_id] = Agent(registry=default_registry)
        return self._agents[session_id]

    def delete_session(self, session_id: str) -> bool:
        """Cleans up and removes a session."""
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            del self._sessions[session_id]
            if session_id in self._agents:
                del self._agents[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """Lists all active session IDs."""
        return list(self._sessions.keys())

    def get_session_summary(self, session_id: str) -> dict:
        """Produces a JSON-serializable snapshot of session status."""
        session = self.get_session(session_id)
        current_ds = None
        if session.current_dataset:
            ds = session.current_dataset
            current_ds = {
                "name": ds.name,
                "rows": ds.rows,
                "columns_count": ds.columns_count,
                "columns": ds.columns,
                "dtypes": ds.dtypes,
                "total_missing": ds.total_missing,
                "filepath": str(ds.filepath),
            }

        test_ds = None
        if session.test_dataset:
            tds = session.test_dataset
            test_ds = {
                "name": tds.name,
                "rows": tds.rows,
                "columns_count": tds.columns_count,
                "columns": tds.columns,
                "dtypes": tds.dtypes,
                "total_missing": tds.total_missing,
                "filepath": str(tds.filepath),
            }

        curr_model = None
        if session.current_model and isinstance(session.current_model, dict):
            curr_model = {
                "name": session.current_model.get("name"),
                "problem_type": session.current_model.get("problem_type"),
                "target_column": session.current_model.get("target_column"),
                "metrics": session.current_model.get("metrics", {}),
                "feature_names": session.current_model.get("feature_names", []),
            }

        return {
            "session_id": session_id,
            "has_dataset": session.current_dataset is not None,
            "has_test_dataset": session.test_dataset is not None,
            "current_dataset": current_ds,
            "test_dataset": test_ds,
            "target_column": session.target_column,
            "current_model": curr_model,
            "models_count": len(session.models),
            "transformations": list(session.transformations),
            "history": list(session.history),
            "experiments_count": len(session.experiments.list_runs()),
        }


# Global session manager instance
session_manager = SessionManager()
