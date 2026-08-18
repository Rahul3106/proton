"""
Base abstractions for Data Science tools.
Per Rule 9 & Rule 10, tools are deterministic, independent of the LLM/UI,
and return structured machine-readable outputs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from proton_agent.context.session import SessionState


@dataclass
class ToolResult:
    """Structured output returned by all Data Science tools."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }


class BaseTool(ABC):
    """Abstract base class for deterministic Data Science tools."""

    name: str
    description: str

    @abstractmethod
    def execute(self, session: SessionState, **kwargs: Any) -> ToolResult:
        """Executes tool logic against session state and returns a structured ToolResult."""
        pass
