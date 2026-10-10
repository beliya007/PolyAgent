"""
PlanAndSolveAgent — Planner first, Executor second.

Use for structured multi-step tasks where you want explicit, inspectable steps.
"""
from __future__ import annotations

import ast
from typing import Iterator, List, Optional

from agents.base import BaseAgent
from core.message import Message
from llm_client import PolyAgentLLM


PLANNER_PROMPT = """You are an expert planner. Decompose the question into a
small list of concrete steps.

Question: {question}

Output strict Python list literal in a fenced code block:
```python
["step 1", "step 2", ...]
```
Keep it to 3-7 steps. Each step should be a single, actionable sentence.
"""


EXECUTOR_SYSTEM_PROMPT = """You are an execution specialist. Follow the plan
step by step. You may use the available tools via the native function-calling
interface to obtain information. Be concise and stick to the current step.
"""


EXECUTOR_PROMPT = """Original question: {question}

Full plan:
{plan}

Completed steps so far:
{history}

Current step: {current_step}

Execute the current step. Call tools if you need more information, then
state the result of this step.
"""


class PlanAndSolveAgent(BaseAgent):
    def __init__(
        self,
        name: str = "PlanAndSolveAgent",
        llm: Optional[PolyAgentLLM] = None,
        tool_registry=None,
        max_steps: int = 6,
        system_prompt: Optional[str] = None,
    ):
        if llm is None:
            from llm_client import build_llm
            llm = build_llm()
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or EXECUTOR_SYSTEM_PROMPT,
            tool_registry=tool_registry,
        )
        self.max_steps = max_steps

    def _plan(self, question: str) -> List[str]:
        prompt = PLANNER_PROMPT.format(question=question)
        raw = self.llm.invoke([{"role": "user", "content": prompt}]) or ""
        try:
            plan_str = raw.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except Exception as e:  # noqa: BLE001
            print(f"plan parse failed: {e}")
            return []

    def _execute_step(
        self,
        question: str,
        plan: List[str],
        history: str,
        step: str,
    ) -> str:
        prompt = EXECUTOR_PROMPT.format(
            question=question,
            plan=plan,
            history=history or "(none)",
            current_step=step,
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        tools_schema = self.get_tools_schema() if self.has_tools() else None
        response = self.call_llm_with_tools(messages, tools=tools_schema)
        if not response.tool_calls:
            return response.content or ""

        messages.append({
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [tc.to_openai() for tc in response.tool_calls],
        })
        for tc in response.tool_calls:
            result = self.execute_tool_call(tc)
            messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "content": result.content,
            })
        messages.append({
            "role": "user",
            "content": "Based on the tool results above, state the conclusion for this step.",
        })
        return self.llm.invoke(messages) or response.content or ""

    def run(self, input_text: str, **kwargs) -> str:
        plan = self._plan(input_text)
        if not plan:
            return "Could not generate a valid plan."
        history = ""
        final = ""
        for i, step in enumerate(plan, 1):
            print(f"\n-> step {i}/{len(plan)}: {step}")
            result = self._execute_step(input_text, plan, history, step)
            history += f"step {i}: {step}\nresult: {result}\n\n"
            final = result
            print(f"   {result[:200]}")
        self.add_message(Message.user(input_text))
        self.add_message(Message.assistant(final))
        return final

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        yield "Generating plan...\n"
        plan = self._plan(input_text)
        if not plan:
            yield "Plan generation failed."
            return
        yield "Plan:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan)) + "\n\n"
        history = ""
        final = ""
        for i, step in enumerate(plan, 1):
            yield f"-> step {i}/{len(plan)}: {step}\n"
            result = self._execute_step(input_text, plan, history, step)
            history += f"step {i}: {step}\nresult: {result}\n\n"
            final = result
            yield f"   {result}\n"
        yield f"\nFinal answer:\n{final}"
        self.add_message(Message.user(input_text))
        self.add_message(Message.assistant(final))
