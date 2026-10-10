"""
Executor — run a :class:`Plan` step by step.

Designed to be domain-neutral. By default, the executor knows a small set
of generic ``action`` names and maps each to a tool by name. If your
application needs different mappings, pass a custom ``action_to_tool``
dictionary or call :meth:`Executor.register_action_map` at runtime.

Parallel batches
----------------
When ``plan.metadata['parallel']`` is True, the executor groups steps
whose ``action`` is in :data:`PARALLEL_OK_ACTIONS` into batches and
executes each batch concurrently with :class:`AsyncToolExecutor`.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm_client import HelloAgentsLLM, tool_to_openai_schema
from memory.manager import MemoryManager
from orchestrator.planner import Plan, PlanStep
from rag.rag_tool import RAGTool
from tools.async_executor import AsyncToolExecutor
from tools.registry import ToolRegistry


# ============================================================
#  Data classes
# ============================================================
@dataclass
class StepResult:
    step: int
    action: str
    output: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    plan: Plan
    steps: List[StepResult] = field(default_factory=list)
    final_answer: str = ""
    elapsed: float = 0.0
    paradigm: str = "unknown"
    parallel_steps: int = 0

    def summary(self) -> str:
        return "\n".join(
            f"[step {r.step}] {r.action}: {'OK' if r.success else 'FAIL'} {str(r.output)[:200]}"
            for r in self.steps
        )


# ============================================================
#  Default action → tool name mapping
# ============================================================
# When a step's action is one of these keys, the executor will look up the
# same-named tool in the registry and call it. Override or extend via
# ``action_to_tool`` in the constructor.
DEFAULT_ACTION_TOOL: Dict[str, str] = {
    "RETRIEVE_KNOWLEDGE": "query_knowledge",
    "LOOKUP": "lookup",
    "QUERY": "query",
    "COMPUTE": "calculator",
    "VALIDATE": "validate",
    "SAVE": "save",
}

# Generic "exploratory" actions that are safe to run in parallel batches
# because they have no data dependency on each other.
DEFAULT_PARALLEL_OK_ACTIONS: frozenset = frozenset({
    "RETRIEVE_KNOWLEDGE",
    "LOOKUP",
    "QUERY",
    "COMPUTE",
})


# ============================================================
#  Executor
# ============================================================
class Executor:
    """Execute a :class:`Plan` with optional parallel batches."""

    def __init__(
        self,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        rag_tool: Optional[RAGTool] = None,
        memory: Optional[MemoryManager] = None,
        max_async_workers: int = 4,
        action_to_tool: Optional[Dict[str, str]] = None,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.rag = rag_tool
        self.memory = memory
        self._async = AsyncToolExecutor(tool_registry, max_workers=max_async_workers)
        self._action_to_tool: Dict[str, str] = dict(action_to_tool or DEFAULT_ACTION_TOOL)
        self._parallel_ok: frozenset = DEFAULT_PARALLEL_OK_ACTIONS

    def shutdown(self, wait: bool = False) -> None:
        self._async.shutdown(wait=wait)

    # -------- public configuration --------
    def register_action_map(self, action: str, tool_name: str) -> None:
        """Bind an action name to a tool name at runtime."""
        self._action_to_tool[action] = tool_name

    def set_parallel_actions(self, actions: Sequence[str]) -> None:
        self._parallel_ok = frozenset(actions)

    # -------- main entry --------
    def execute(self, plan: Plan, query: str, session_id: str = "") -> ExecutionResult:
        return asyncio.run(self.aexecute(plan, query, session_id))

    async def aexecute(self, plan: Plan, query: str, session_id: str = "") -> ExecutionResult:
        start = time.time()
        result = ExecutionResult(plan=plan)

        enable_parallel = bool(getattr(plan, "metadata", {}).get("parallel", False))
        batches = (
            self._split_into_parallel_batches(plan.steps)
            if enable_parallel
            else [[s] for s in plan.steps]
        )

        collected: List[str] = []
        for batch in batches:
            if len(batch) > 1 and enable_parallel:
                outputs = await self._run_batch_parallel(batch, query, collected)
                result.parallel_steps += len(outputs)
            else:
                outputs = [self._run_step(batch[0], query, collected)]

            for step, out in zip(batch, outputs):
                success = not (out.startswith("ERROR:") or out.startswith("FAILED:"))
                result.steps.append(
                    StepResult(
                        step=step.step,
                        action=step.action,
                        output=out,
                        success=success,
                        metadata={"batch_parallel": len(batch) > 1} if enable_parallel else {},
                    )
                )
                collected.append(f"step {step.step} ({step.action}): {out}")

        result.final_answer = self._direct_answer(query, "\n".join(collected))
        result.elapsed = time.time() - start
        return result

    # -------- parallel batching --------
    def _split_into_parallel_batches(
        self,
        steps: Sequence[PlanStep],
    ) -> List[List[PlanStep]]:
        batches: List[List[PlanStep]] = []
        cur: List[PlanStep] = []
        for s in steps:
            ok_to_merge = (
                s.action in self._parallel_ok
                and all(c.action in self._parallel_ok for c in cur)
                and cur
            )
            if ok_to_merge:
                cur.append(s)
            else:
                if cur:
                    batches.append(cur)
                cur = [s]
        if cur:
            batches.append(cur)
        return batches

    async def _run_batch_parallel(
        self,
        batch: Sequence[PlanStep],
        query: str,
        history: List[str],
    ) -> List[str]:
        tasks = []
        for step in batch:
            tool_name = self._action_to_tool.get(step.action)
            if not tool_name:
                return [self._run_step(step, query, history) for step in batch]
            params = self._fill_params(tool_name, dict(step.params or {}), query)
            tasks.append({"tool_name": tool_name, "input_data": params})
        print(f"Running {len(tasks)} tools in parallel")
        try:
            results = await self._async.execute_tools_parallel(tasks)
            return results
        except Exception as e:  # noqa: BLE001
            return [f"ERROR: parallel batch failed: {e}"] * len(tasks)

    # -------- single-step execution --------
    def _run_step(self, step: PlanStep, query: str, history: List[str]) -> str:
        tool_name = self._action_to_tool.get(step.action)
        if tool_name:
            return self._execute_tool_with_fc(tool_name, step.params, query)
        if step.action == "GENERATE":
            return self._generate(query, history, step.params)
        if step.action == "DIRECT_ANSWER":
            return self._direct_answer(query, history, step.params)
        return f"ERROR: unknown action: {step.action}"

    def _execute_tool_with_fc(
        self, tool_name: str, params: Dict[str, Any], query: str
    ) -> str:
        """Native function-calling path."""
        tool = self.tools.get_tool(tool_name)
        if not tool:
            return f"ERROR: tool '{tool_name}' not registered"

        tools_schema = [tool_to_openai_schema(tool)]
        filled_params = self._fill_params(tool_name, params, query)

        system_prompt = (
            "You are a helpful AI assistant. When you need information, "
            "call the available tools using their function-calling schema. "
            f"User query: {query}\n"
            f"Known info so far: {json.dumps(params, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Please call the '{tool_name}' tool with parameters: "
                    f"{json.dumps(filled_params, ensure_ascii=False)}"
                ),
            },
        ]

        response = self.llm.invoke_with_tools(messages, tools=tools_schema)
        if not response.tool_calls:
            return response.content or "tool was not invoked"

        tc = response.tool_calls[0]
        return self.tools.execute_tool(tc.name, tc.arguments)

    def _fill_params(
        self, tool_name: str, params: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Fill missing required parameters with sensible defaults.

        Generic fallback: copy the user query into the first string-shaped
        required parameter. Override in a subclass for domain-specific
        routing.
        """
        tool = self.tools.get_tool(tool_name)
        if not tool:
            return params
        required = [p for p in tool.parameters if p.required]
        result = dict(params)
        used = {k.lower() for k in result.keys()}

        # If the only required parameter is a free-form text field, mirror the query.
        free_form_keys = {"query", "text", "input", "expression", "json_string", "url"}
        for p in required:
            if p.name.lower() in used:
                continue
            if p.name in free_form_keys:
                result[p.name] = query
            else:
                result[p.name] = query if p.type == "string" else None
        return result

    def _generate(self, query: str, history: List[str], params: Dict[str, Any]) -> str:
        """LLM-based generation step."""
        prompt = (
            "Based on the information collected so far, compose the final answer.\n\n"
            f"# User query\n{query}\n\n"
            f"# Collected info\n{chr(10).join(history) if history else '(none)'}\n\n"
            f"# Constraints\n{json.dumps(params, ensure_ascii=False) if params else '(none)'}\n\n"
            "Write a clear, professional response. Stay within any specified constraints."
        )
        return self.llm.invoke([{"role": "user", "content": prompt}]) or "generation failed"

    def _direct_answer(
        self, query: str, history: List[str], params: Optional[Dict[str, Any]] = None
    ) -> str:
        prompt = (
            "Answer the user's question using the context below.\n\n"
            f"# User query\n{query}\n\n"
            f"# Context\n{chr(10).join(history) if history else '(none)'}\n"
        )
        return self.llm.invoke([{"role": "user", "content": prompt}]) or "no answer"
