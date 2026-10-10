"""
ReActAgent — Thought / Action / Observation loop on top of native function calling.

Use when the agent needs to flexibly decide which tools to call and when to stop.
"""
from __future__ import annotations

import json as _json
from typing import Iterator, Optional

from agents.base import BaseAgent
from core.message import Message, ToolResult
from llm_client import HelloAgentsLLM
from tools.registry import ToolRegistry


DEFAULT_REACT_PROMPT = """You are a tool-using AI assistant operating in a
Thought / Action / Observation loop.

Rules:
1. Think first (Thought), then decide whether a tool is needed (Action).
2. Use the native function-calling interface for tool calls — never hand-write
   a tool invocation in the response text.
3. After receiving an Observation, decide on the next Thought.
4. When you have enough information, return the final answer to the user
   directly (no more tool calls).
5. Be concise and professional; reply in the user's language.
"""


def _format_args(args: dict) -> str:
    if not args:
        return ""
    return ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())


class ReActAgent(BaseAgent):
    def __init__(
        self,
        name: str = "ReActAgent",
        llm: Optional[HelloAgentsLLM] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_steps: int = 6,
        system_prompt: Optional[str] = None,
    ):
        if llm is None:
            from llm_client import build_llm
            llm = build_llm()
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or DEFAULT_REACT_PROMPT,
            tool_registry=tool_registry,
        )
        self.max_steps = max_steps
        self._trajectory: list[dict] = []

    def _build_user_prompt(self, question: str) -> str:
        tools_desc = (
            self.tool_registry.get_tools_description() if self.has_tools() else "(no tools)"
        )
        return (
            f"Available tools:\n{tools_desc}\n\n"
            f"Solve the following question using Thought -> Action -> Observation.\n"
            f"Use the native function-calling interface (no hand-written tool code).\n\n"
            f"Question: {question}"
        )

    def run(self, input_text: str, **kwargs) -> str:
        self._trajectory = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_prompt(input_text)},
        ]
        tools_schema = self.get_tools_schema() if self.has_tools() else None

        for step in range(1, self.max_steps + 1):
            print(f"\n--- ReAct step {step} ---")
            response = self.call_llm_with_tools(messages, tools=tools_schema, **kwargs)

            thought = response.content or ""
            if thought:
                print(f"Thought: {thought[:200]}{'...' if len(thought) > 200 else ''}")
                self._trajectory.append({"step": step, "thought": thought})

            if not response.tool_calls:
                self.add_message(Message.user(input_text))
                self.add_message(Message.assistant(thought))
                return thought

            results: list[ToolResult] = []
            for tc in response.tool_calls:
                print(f"Action: {tc.name}({_format_args(tc.arguments)})")
                result = self.execute_tool_call(tc)
                results.append(result)
                preview = result.content[:200] + ("..." if len(result.content) > 200 else "")
                print(f"Observation: {preview}")
                self._trajectory.append({
                    "step": step,
                    "action": tc.name,
                    "arguments": tc.arguments,
                    "observation": result.content,
                })

            messages.append({
                "role": "assistant",
                "content": thought,
                "tool_calls": [tc.to_openai() for tc in response.tool_calls],
            })
            for r in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": r.tool_call_id,
                    "name": r.name,
                    "content": r.content,
                })

        print(f"Reached max steps {self.max_steps}; summarising.")
        summary = self.llm.invoke(messages, **kwargs) or "no answer produced"
        return summary

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        self._trajectory = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_user_prompt(input_text)},
        ]
        tools_schema = self.get_tools_schema() if self.has_tools() else None

        final_text = ""
        for step in range(1, self.max_steps + 1):
            yield f"\n[Step {step}]\n"
            response = self.call_llm_with_tools(messages, tools=tools_schema, **kwargs)
            thought = response.content or ""
            if thought:
                yield f"Thought: {thought}\n"
                final_text = thought

            if not response.tool_calls:
                self.add_message(Message.user(input_text))
                self.add_message(Message.assistant(final_text))
                return

            messages.append({
                "role": "assistant",
                "content": thought,
                "tool_calls": [tc.to_openai() for tc in response.tool_calls],
            })
            for tc in response.tool_calls:
                yield f"Action: {tc.name}({_format_args(tc.arguments)})\n"
                result = self.execute_tool_call(tc)
                preview = result.content[:300] + ("..." if len(result.content) > 300 else "")
                yield f"Observation: {preview}\n"
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "name": result.name,
                    "content": result.content,
                })

        summary = self.llm.invoke(messages, **kwargs) or "no answer produced"
        yield f"\nSummary: {summary}\n"
        self.add_message(Message.user(input_text))
        self.add_message(Message.assistant(summary))
