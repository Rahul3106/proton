"""
Optional REST API package powered by FastAPI and Uvicorn.
Wraps core Agent and SessionState without coupling core logic.
"""
from proton_agent.api.server import app

__all__ = ["app"]
