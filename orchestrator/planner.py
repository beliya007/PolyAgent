"""
Planner — turn a recognised intent into an executable plan.

Domain-neutral by design: the default plan templates and the available
``action`` vocabulary are deliberately generic. To specialise the planner
for a particular application, subclass it and override
:meth:`Planner._default_plan` or pass a custom ``prompt_template``.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_client import PolyAgentLLM
from orchestrator.intent_recognizer import IntentResult


# Default available actions. A planner is allowed to emit any string,
# but these are the ones the default Executor understands out of the box.
DEFAULT_ACTIONS: List[str] = [
    "RETRIEVE_KNOWLEDGE",   # pull information from a knowledge base / RAG
    "LOOKUP",               # fetch a record by id / key
    "QUERY",                # call an external query-style tool
    "COMPUTE",              # run an arithmetic / symbolic tool
    "GENERATE",             # LLM generates an artefact from the collected info
    "VALIDATE",             # validate an artefact against rules
    "SAVE",                 # persist the result
    "DIRECT_ANSWER",        # no further tool use; answer from context
]


PLANNER_PROMPT = """You are a planning expert. Decompose the user request into a small
sequence of executable steps that together satisfy the goal.

Intent: {intent}
Entities: {entities}
User input: {query}

Available action types (use exactly one per step):
- RETRIEVE_KNOWLEDGE - pull information from a knowledge base / RAG
- LOOKUP             - fetch a record by id / key
- QUERY              - call an external query-style tool
- COMPUTE            - run an arithmetic / symbolic tool
- GENERATE           - LLM composes an artefact from the collected info
- VALIDATE           - validate an artefact against rules
- SAVE               - persist the result
- DIRECT_ANSWER      - answer directly from context, no further tool use

Output strict JSON only (no prose, no markdown fences):
[
  {{"step": 1, "action": "<one of the actions above>", "params": {{}}, "purpose": "<short reason>"}},
  ...
]

Guidelines:
- Steps should have a clear dependency order; later steps may reference earlier results.
- 3~7 steps is usually enough; pick the smallest plan that still works.
- For generation tasks, prefer to RETRIEVE_KNOWLEDGE / QUERY first, then GENERATE, then VALIDATE.
- Use DIRECT_ANSWER as the final step when no further tool is required.

Output JSON array only.
"""


@dataclass
class PlanStep:
    step: int
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    purpose: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "params": self.params,
            "purpose": self.purpose,
        }


@dataclass
class Plan:
    steps: List[PlanStep] = field(default_factory=list)
    raw: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)  # plan↔executor instruction channel

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps], "metadata": self.metadata}


class Planner:
    """Generate an execution plan for a recognised intent."""

    PROMPT = PLANNER_PROMPT
    DEFAULT_ACTIONS = DEFAULT_ACTIONS

    def __init__(
        self,
        llm: PolyAgentLLM,
        prompt_template: Optional[str] = None,
        actions: Optional[List[str]] = None,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template or self.PROMPT
        self.actions = list(actions or self.DEFAULT_ACTIONS)

    def plan(self, intent: IntentResult, query: str) -> Plan:
        prompt = self.prompt_template.format(
            intent=intent.intent,
            entities=json.dumps(intent.entities, ensure_ascii=False),
            query=query,
        )
        raw = self.llm.invoke([{"role": "user", "content": prompt}]) or ""

        steps = self._parse_steps(raw)
        if not steps:
            steps = self._default_plan(intent)
        return Plan(steps=steps, raw=raw)

    def _parse_steps(self, raw: str) -> List[PlanStep]:
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(m.group(0))
            except Exception:
                return []
        if not isinstance(data, list):
            return []
        steps: List[PlanStep] = []
        for i, item in enumerate(data, 1):
            if not isinstance(item, dict):
                continue
            steps.append(
                PlanStep(
                    step=item.get("step", i),
                    action=item.get("action", "DIRECT_ANSWER"),
                    params=item.get("params", {}) or {},
                    purpose=item.get("purpose", ""),
                )
            )
        return steps

    @staticmethod
    def _default_plan(intent: IntentResult) -> List[PlanStep]:
        """Generic fallback plan for the built-in intent types.

        Override this in a subclass for domain-specific behaviour.
        """
        ents = dict(intent.entities or {})
        if intent.intent == "task_planning":
            return [
                PlanStep(1, "RETRIEVE_KNOWLEDGE", ents, "Retrieve relevant background knowledge"),
                PlanStep(2, "LOOKUP", ents, "Look up related records"),
                PlanStep(3, "GENERATE", {}, "Compose the final artefact"),
                PlanStep(4, "VALIDATE", {}, "Validate against the rules"),
            ]
        if intent.intent == "data_query":
            return [
                PlanStep(1, "QUERY", ents, "Run the data query"),
                PlanStep(2, "DIRECT_ANSWER", {}, "Summarise the result"),
            ]
        if intent.intent == "calculation":
            return [
                PlanStep(1, "COMPUTE", {"expression": ents.get("query", "")}, "Run the calculation"),
            ]
        if intent.intent == "knowledge_query":
            return [
                PlanStep(1, "RETRIEVE_KNOWLEDGE", {"query": ents.get("query", "")}, "Retrieve from KB"),
                PlanStep(2, "DIRECT_ANSWER", {}, "Summarise the result"),
            ]
        if intent.intent == "tool_use":
            return [
                PlanStep(1, "QUERY", ents, "Invoke the requested tool"),
                PlanStep(2, "DIRECT_ANSWER", {}, "Surface the tool result"),
            ]
        # chitchat / unknown / anything else
        return [PlanStep(1, "DIRECT_ANSWER", {}, "Reply directly from the model")]
