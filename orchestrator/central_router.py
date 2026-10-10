"""
Central multi-agent router.

Pipeline:
    user input
      ↓
    1. Intent recognition (IntentRecognizer)
      ↓
    2. Planning         (Planner)     ──► Plan
      ↓
    3. Execution        (Executor)    ──► StepResults (auto-parallel batches)
      ↓
    4. Reflection       (Reflector)   ──► score + retry if below threshold
      ↓
    5. Result           (intent / plan / execution / reflection / final_answer)

Four agent paradigms are available for the execution phase:
  - ReAct             flexible tool use (default for tool_use / data_query)
  - Simple            single-shot answer (default for chitchat / knowledge_query)
  - Plan-and-Solve    structured multi-step
  - Reflection        iterative refine

Domain-neutral by design: the router itself owns no business logic. All
specific behaviour comes from the ``ToolRegistry`` it is constructed with
and the LLM behind it.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents import (
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
)
from context.builder import ContextBuilder
from llm_client import PolyAgentLLM
from memory.manager import MemoryManager
from orchestrator.executor import Executor, ExecutionResult, Plan, PlanStep
from orchestrator.intent_recognizer import IntentRecognizer, IntentResult
from orchestrator.planner import Planner
from orchestrator.reflector import Reflector, ReflectionResult
from rag.rag_tool import RAGTool
from tools.async_executor import AsyncToolExecutor
from tools.chain import ToolChainManager
from tools.registry import ToolRegistry


# ============================================================
#  结果数据结构
# ============================================================
@dataclass
class CentralRouterResult:
    """End-to-end routing result."""
    query: str
    intent: IntentResult
    plan: Plan
    execution: ExecutionResult
    reflection: ReflectionResult
    final_answer: str
    used_paradigm: str = ""
    elapsed_time: float = 0.0
    session_id: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "intent": {
                "intent": self.intent.intent,
                "confidence": self.intent.confidence,
                "mode": self.intent.mode,
                "entities": self.intent.entities,
                "reasoning": self.intent.reasoning,
            },
            "plan": self.plan.to_dict(),
            "execution": {
                "paradigm": getattr(self.execution, "paradigm", "unknown"),
                "steps": [
                    {
                        "step": s.step,
                        "action": s.action,
                        "success": s.success,
                        "output": str(s.output)[:500] if s.output else "",
                        "batch_parallel": s.metadata.get("batch_parallel", False),
                    }
                    for s in getattr(self.execution, "steps", [])
                ],
                "final_answer": getattr(self.execution, "final_answer", ""),
                "elapsed": getattr(self.execution, "elapsed", 0.0),
                "parallel_steps": getattr(self.execution, "parallel_steps", 0),
            },
            "reflection": {
                "score": self.reflection.score,
                "is_satisfactory": self.reflection.is_satisfactory,
                "issues": self.reflection.issues,
                "suggestions": self.reflection.suggestions,
            },
            "final_answer": self.final_answer,
            "used_paradigm": self.used_paradigm,
            "elapsed_time": self.elapsed_time,
            "metadata": self.metadata,
        }


# ============================================================
#  中心路由
# ============================================================
class CentralRouter:
    """Centralised multi-agent orchestrator.

    All capabilities come from the injected ``tool_registry``, ``memory_manager``
    and ``rag_tool`` — the router itself contains no domain knowledge.
    """

    def __init__(
        self,
        llm: PolyAgentLLM,
        tool_registry: ToolRegistry,
        memory_manager: Optional[MemoryManager] = None,
        rag_tool: Optional[RAGTool] = None,
        context_builder: Optional[ContextBuilder] = None,
        enable_reflection: bool = True,
        max_retry: int = 1,
        enable_parallel: bool = True,
        max_async_workers: int = 4,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.memory = memory_manager
        self.rag = rag_tool
        self.context = context_builder or ContextBuilder(
            memory_manager=memory_manager,
            rag_tool=rag_tool,
        )
        self.enable_reflection = enable_reflection
        self.max_retry = max_retry
        self.enable_parallel = enable_parallel

        # Singletons
        self.intent_recognizer = IntentRecognizer(llm)
        self.planner = Planner(llm)
        self.executor = Executor(
            llm,
            tool_registry,
            rag_tool=rag_tool,
            memory=memory_manager,
            max_async_workers=max_async_workers,
        )
        self.reflector = Reflector(llm)

        # Chain + parallel executors (polyagent 7.5.4)
        self.chain_manager = ToolChainManager(tool_registry)
        self.async_executor = AsyncToolExecutor(
            tool_registry, max_workers=max_async_workers
        )

        # Lazy paradigm instances
        self._simple: Optional[SimpleAgent] = None
        self._react: Optional[ReActAgent] = None
        self._plan_solve: Optional[PlanAndSolveAgent] = None
        self._reflection: Optional[ReflectionAgent] = None

        print(
            f"CentralRouter ready "
            f"(reflection={'on' if enable_reflection else 'off'}"
            f", parallel={'on' if enable_parallel else 'off'})"
        )

    # -------- paradigm lazy properties --------
    @property
    def simple_agent(self) -> SimpleAgent:
        if not self._simple:
            self._simple = SimpleAgent(llm=self.llm, tool_registry=self.tools)
        return self._simple

    @property
    def react_agent(self) -> ReActAgent:
        if not self._react:
            self._react = ReActAgent(llm=self.llm, tool_registry=self.tools)
        return self._react

    @property
    def plan_solve_agent(self) -> PlanAndSolveAgent:
        if not self._plan_solve:
            self._plan_solve = PlanAndSolveAgent(llm=self.llm, tool_registry=self.tools)
        return self._plan_solve

    @property
    def reflection_agent(self) -> ReflectionAgent:
        if not self._reflection:
            self._reflection = ReflectionAgent(llm=self.llm)
        return self._reflection

    def shutdown(self, wait: bool = False) -> None:
        """Shut down the async thread pool. Call before process exit."""
        self.async_executor.shutdown(wait=wait)
        if hasattr(self.executor, "shutdown"):
            self.executor.shutdown(wait=wait)

    # ============================================================
    #  Core routing method
    # ============================================================
    def route(
        self,
        query: str,
        mode: str = "auto",
        session_id: Optional[str] = None,
        parallel: Optional[bool] = None,
    ) -> CentralRouterResult:
        """Full pipeline: intent → plan → execute → reflect → return."""
        start = time.time()
        session_id = session_id or str(uuid.uuid4())[:8]
        metadata: Dict[str, Any] = {"session_id": session_id}

        # 1. Intent
        print("\n[1/5] Intent recognition...")
        intent = self.intent_recognizer.recognize(query)
        print(
            f"  -> intent: {intent.intent} | confidence: {intent.confidence:.2f} | "
            f"recommended mode: {intent.mode}"
        )
        metadata["intent_type"] = intent.intent
        metadata["entities"] = intent.entities

        if self.memory:
            self.memory.add_working(
                f"user: {query[:50]} -> intent: {intent.intent}",
                importance=0.6,
                session_id=session_id,
            )

        # 2. Paradigm choice
        chosen = mode if mode != "auto" else intent.mode
        metadata["chosen_paradigm"] = chosen
        print(f"\n[2/5] Paradigm: {chosen}")

        # 3. Build context (RAG + memory)
        context_prompt = self._build_context(query, session_id)
        print(f"  -> context: {'built' if context_prompt else 'empty'}")

        # 4. Execute
        print(f"\n[3/5] Executing ({chosen})...")
        execution: ExecutionResult
        used_paradigm = chosen

        if chosen == "react":
            execution, used_paradigm = self._execute_via_react(query, intent, session_id, parallel)
        elif chosen == "plan_solve":
            execution, used_paradigm = self._execute_via_plan_solve(query, intent, session_id)
        elif chosen == "reflection":
            execution, used_paradigm = self._execute_via_reflection(query, intent, session_id)
        else:
            execution, used_paradigm = self._execute_via_simple(query, intent, session_id)

        final_answer = execution.final_answer

        # 5. Reflect
        print("\n[4/5] Reflecting...")
        reflection = self._reflect(query, execution, final_answer)

        retry_count = 0
        while (
            not reflection.is_satisfactory
            and reflection.score < 0.6
            and retry_count < self.max_retry
        ):
            retry_count += 1
            print(f"  -> score {reflection.score:.2f} too low, retry #{retry_count}...")
            refined = self.reflector.refine(query, execution, reflection)
            final_answer = refined
            execution.final_answer = refined
            metadata["retry"] = retry_count
            reflection = self._reflect(query, execution, final_answer)

        print(
            f"  -> score: {reflection.score:.2f} | "
            f"satisfactory: {reflection.is_satisfactory}"
        )
        if reflection.issues:
            print(f"  -> issues: {', '.join(reflection.issues[:2])}")

        if self.memory:
            self.memory.add_episodic(
                f"[{session_id}] {query[:40]} | intent:{intent.intent} | "
                f"paradigm:{used_paradigm} | score:{reflection.score:.2f}",
                importance=0.7,
                session_id=session_id,
                event_type="agent_completion",
            )

        elapsed = time.time() - start
        return CentralRouterResult(
            query=query,
            intent=intent,
            plan=Plan(steps=[]),        # plan already executed
            execution=execution,
            reflection=reflection,
            final_answer=final_answer,
            used_paradigm=used_paradigm,
            elapsed_time=elapsed,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
        )

    # ============================================================
    #  Context building
    # ============================================================
    def _build_context(self, query: str, session_id: str) -> str:
        """Build enhanced context from RAG + memory."""
        parts: List[str] = []
        if self.rag:
            ctx = self.rag.build_context(query)
            if ctx:
                parts.append(ctx)
        if self.memory:
            items = self.memory.search(query, limit=3)
            if items:
                lines = ["[relevant memory]"]
                for it in items:
                    lines.append(
                        f"[{it.memory_type}|{it.importance:.1f}] {it.content[:80]}"
                    )
                parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # ============================================================
    #  Reflection
    # ============================================================
    def _reflect(
        self, query: str, execution: ExecutionResult, answer: str
    ) -> ReflectionResult:
        if not self.enable_reflection:
            return ReflectionResult(
                score=0.8, is_satisfactory=True, issues=[], suggestions=[]
            )
        return self.reflector.reflect(query, execution, answer)

    # ============================================================
    #  Paradigm adapters
    # ============================================================
    def _execute_via_react(
        self,
        query: str,
        intent: IntentResult,
        session_id: str,
        parallel: Optional[bool] = None,
    ) -> tuple:
        plan = self.planner.plan(intent, query)
        if parallel is None:
            parallel = self.enable_parallel
        plan.metadata["parallel"] = parallel
        print(f"  -> {len(plan.steps)} steps, parallel={parallel}")
        exec_result = self.executor.execute(plan, query, session_id=session_id)
        exec_result.paradigm = "react"
        return exec_result, "react"

    def _execute_via_simple(
        self, query: str, intent: IntentResult, session_id: str
    ) -> tuple:
        answer = self.simple_agent.run(query)
        plan = Plan(steps=[PlanStep(1, "DIRECT_ANSWER", {})])
        return (
            ExecutionResult(plan=plan, final_answer=answer, paradigm="simple"),
            "simple",
        )

    def _execute_via_plan_solve(
        self, query: str, intent: IntentResult, session_id: str
    ) -> tuple:
        answer = self.plan_solve_agent.run(query)
        plan = Plan(steps=[PlanStep(1, "PLAN_AND_EXECUTE", {})])
        return (
            ExecutionResult(plan=plan, final_answer=answer, paradigm="plan_solve"),
            "plan_solve",
        )

    def _execute_via_reflection(
        self, query: str, intent: IntentResult, session_id: str
    ) -> tuple:
        answer = self.reflection_agent.run(query)
        plan = Plan(steps=[PlanStep(1, "REFLECTION_LOOP", {})])
        return (
            ExecutionResult(plan=plan, final_answer=answer, paradigm="reflection"),
            "reflection",
        )

    # ============================================================
    #  Chain / parallel public API
    # ============================================================
    def run_chain(
        self,
        chain_name: str,
        initial_input: Any = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Run a registered tool chain synchronously."""
        from tools.base import ToolResult
        return self.chain_manager.execute(
            chain_name, initial_input=initial_input, context=context
        )

    async def arun_chain(
        self,
        chain_name: str,
        initial_input: Any = "",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Run a registered tool chain asynchronously."""
        return await self.async_executor.execute_chain_async(
            chain_name,
            self.chain_manager,
            initial_input=initial_input,
            context=context,
        )

    async def arun_tools_parallel(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Run multiple tools in parallel."""
        return await self.async_executor.execute_tools_parallel(tasks)

    def list_chains(self) -> List[str]:
        """List all registered tool chains."""
        return self.chain_manager.list()

    # ============================================================
    #  Stats
    # ============================================================
    def stats(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.stats(),
            "memory": self.memory.stats() if self.memory else None,
            "rag": self.rag.stats() if self.rag else None,
            "tools": self.tools.list_tools(),
            "chains": self.list_chains(),
            "parallel_enabled": self.enable_parallel,
        }


# ============================================================
#  Convenience singleton
# ============================================================
_router: Optional[CentralRouter] = None


def get_router(
    enable_persistence: bool = False,
    enable_reflection: bool = True,
    enable_reranker: bool = False,
    enable_parallel: bool = True,
    tool_registry: Optional[ToolRegistry] = None,
) -> CentralRouter:
    """Get (or create) the global CentralRouter singleton.

    Pass a custom ``tool_registry`` to use your own domain-specific tools.
    Without one, the framework's built-in example tool registry is used
    (echo, calculator, current_time, http_get, text_stats, uuid_generator,
    json_format).
    """
    global _router
    if _router is None:
        from llm_client import build_llm
        from rag import RAGTool

        llm = build_llm()
        if tool_registry is None:
            from tools import create_default_tool_registry
            tool_registry = create_default_tool_registry()

        try:
            rag = RAGTool(
                llm,
                namespace="polyagent",
                enable_hybrid=True,
                enable_reranker=enable_reranker,
            )
        except Exception:
            rag = None

        try:
            memory = MemoryManager(
                user_id="polyagent_user",
                enable_persistence=enable_persistence,
            )
        except Exception:
            memory = MemoryManager()

        _router = CentralRouter(
            llm=llm,
            tool_registry=tool_registry,
            memory_manager=memory,
            rag_tool=rag,
            enable_reflection=enable_reflection,
            enable_parallel=enable_parallel,
        )
        print("Global CentralRouter singleton created")
    return _router


def chat(
    query: str,
    mode: str = "auto",
    session_id: Optional[str] = None,
    parallel: Optional[bool] = None,
) -> Dict[str, Any]:
    """One-line chat entry point."""
    router = get_router()
    result = router.route(query, mode=mode, session_id=session_id, parallel=parallel)
    return result.to_dict()
