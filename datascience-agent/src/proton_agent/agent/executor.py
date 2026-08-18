"""
Executor: executes chosen tool against SessionState via ToolRegistry.
"""
from __future__ import annotations

from typing import Any, Optional

from proton_agent.context.session import SessionState
from proton_agent.tools import ToolRegistry, ToolResult, default_registry


class Executor:
    """Runs a specified tool using the current session context."""

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or default_registry

    def execute_tool(self, tool_name: str, session: SessionState, **kwargs: Any) -> ToolResult:
        """Executes tool by name against session state."""
        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' is not registered in ToolRegistry.",
            )

        return tool.execute(session, **kwargs)
