"""
SimpleAgent — single-turn LLM with native function calling.

Use for direct Q&A, chitchat and trivial single-tool invocations.
"""
from __future__ import annotations

from typing import Iterator, Optional

from agents.base import BaseAgent
from core.message import Message
from llm_client import PolyAgentLLM
from tools.registry import ToolRegistry


DEFAULT_SIMPLE_PROMPT = """You are a helpful AI assistant. Answer the user's
question using your own knowledge and any registered tools.

Guidelines:
- If a tool can supply the needed information, call it via the native
  function-calling interface (do not hand-write the call).
- Be concise and professional; prefer concrete numbers / facts over
  hand-waving.
- Reply in the user's language.
"""


class SimpleAgent(BaseAgent):
    def __init__(
        self,
        name: str = "SimpleAgent",
        llm: Optional[PolyAgentLLM] = None,
        system_prompt: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        if llm is None:
            from llm_client import build_llm
            llm = build_llm()
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or DEFAULT_SIMPLE_PROMPT,
            tool_registry=tool_registry,
        )

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        messages = self.to_openai_messages(input_text)
        tools_schema = self.get_tools_schema() if self.has_tools() else None

        for _ in range(max_tool_iterations + 1):
            response = self.call_llm_with_tools(
                messages,
                tools=tools_schema,
                **kwargs,
            )

            if not response.tool_calls:
                content = response.content or ""
                self.add_message(Message.user(input_text))
                self.add_message(Message.assistant(content))
                return content

            print(f"tool calls detected: {len(response.tool_calls)}")
            results = []
            for tc in response.tool_calls:
                result = self.execute_tool_call(tc)
                results.append(result)
                print(f"  {tc.name}({tc.arguments}) -> {result.content[:80]}")

            self.add_message(
                Message.assistant(response.content, tool_calls=response.tool_calls)
            )
            for r in results:
                self.add_message(Message.tool(r.content, r.tool_call_id, r.name))
            messages = [m.to_openai() for m in self.history]

        # Out of iterations — fall back to a plain completion.
        final = self.llm.invoke(messages, **kwargs) or ""
        return final

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        messages = self.to_openai_messages(input_text)
        tools_schema = self.get_tools_schema() if self.has_tools() else None

        full = ""
        for chunk in self.llm.stream_with_tools(
            messages, tools=tools_schema, **kwargs
        ):
            if chunk["type"] == "content":
                piece = chunk["delta"]
                full += piece
                yield piece

        self.add_message(Message.user(input_text))
        self.add_message(Message.assistant(full))
