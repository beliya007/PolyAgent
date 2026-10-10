"""
agents package — four agent paradigms.

Public surface
--------------
- BaseAgent
- SimpleAgent
- ReActAgent
- PlanAndSolveAgent
- ReflectionAgent

All agents are domain-neutral. The default ``system_prompt`` is a
generic helpful-assistant prompt; pass a custom ``system_prompt`` to
specialise any agent for a particular application.
"""
from agents.base import BaseAgent
from agents.message import Message, ToolCall, ToolResult
from agents.plan_solve_agent import PlanAndSolveAgent
from agents.react_agent import ReActAgent
from agents.reflection_agent import ReflectionAgent
from agents.simple_agent import SimpleAgent

__all__ = [
    "BaseAgent",
    "Message",
    "ToolCall",
    "ToolResult",
    "SimpleAgent",
    "ReActAgent",
    "PlanAndSolveAgent",
    "ReflectionAgent",
]
