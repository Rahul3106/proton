"""
ToolRegistry: central registry for discovering and accessing Data Science tools.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from proton_agent.tools.base import BaseTool


class ToolRegistry:
    """Registry managing available Data Science tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance by its name."""
        self._tools[tool.name.lower()] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name.lower())

    def list_tools(self) -> List[BaseTool]:
        """Return list of all registered tools."""
        return list(self._tools.values())

    def tool_names(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())


# Process-wide default registry
default_registry = ToolRegistry()
