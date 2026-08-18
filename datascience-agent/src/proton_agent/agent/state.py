"""
Agent state dataclasses for encapsulating planning and execution outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from proton_agent.tools.base import ToolResult


@dataclass
class PlanStep:
    """Represents a single step in an Agent Plan."""

    step_number: int
    description: str
    tool_name: str
    completed: bool = False
    result: Optional[ToolResult] = None
    tool_kwargs: dict = field(default_factory=dict)


@dataclass
class Plan:
    """Represents a multi-step execution plan for achieving a Data Science goal."""

    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    is_complete: bool = False

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for step in self.steps if step.completed)


@dataclass
class AgentResponse:
    """Represents the outcome of an Agent turn."""

    user_request: str
    selected_tool: Optional[str]
    tool_result: Optional[ToolResult]
    explanation: str
    plan: Optional[Plan] = None
    summary: str = ""
