"""
Reflector — quality scoring and (optional) refinement of an agent's answer.

The default prompt is generic. Override :attr:`Reflector.PROMPT` or pass
a custom ``prompt_template`` to the constructor to specialise it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_client import HelloAgentsLLM
from orchestrator.executor import ExecutionResult


REFLECT_PROMPT = """You are a strict quality reviewer. Evaluate whether the agent's
answer below actually satisfies the user's question.

# User question
{query}

# Agent answer
{answer}

# Execution trace
{timeline}

Score the answer on three dimensions and output strict JSON only:
1. Relevance:   does it actually answer the question?
2. Completeness: are key details present?
3. Accuracy:    are the claims internally consistent and free of obvious errors?

Output JSON (no markdown fences):
{{
  "score": <0.0~1.0>,
  "is_satisfactory": <true|false>,
  "issues": ["problem 1", ...],
  "suggestions": ["improvement 1", ...]
}}
"""


REFINE_PROMPT = """You are the original author. Improve the answer using the reviewer's
feedback. Keep what is already good, fix the listed issues, and apply the
suggestions where they make sense.

# User question
{query}

# Original answer
{answer}

# Reviewer feedback
- Score: {score}
- Issues: {issues}
- Suggestions: {suggestions}

Write the improved final answer. Reply with the answer only.
"""


@dataclass
class ReflectionResult:
    score: float
    is_satisfactory: bool
    issues: List[str]
    suggestions: List[str]
    raw: str = ""


class Reflector:
    """Score an :class:`ExecutionResult` and optionally refine the answer."""

    PROMPT = REFLECT_PROMPT
    REFINE = REFINE_PROMPT

    def __init__(
        self,
        llm: HelloAgentsLLM,
        prompt_template: Optional[str] = None,
        refine_template: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template or self.PROMPT
        self.refine_template = refine_template or self.REFINE

    def reflect(
        self, query: str, result: ExecutionResult, answer: str
    ) -> ReflectionResult:
        prompt = self.prompt_template.format(
            query=query,
            answer=answer,
            timeline=result.summary(),
        )
        raw = self.llm.invoke([{"role": "user", "content": prompt}]) or ""

        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
                return ReflectionResult(
                    score=float(data.get("score", 0.7)),
                    is_satisfactory=bool(data.get("is_satisfactory", True)),
                    issues=list(data.get("issues", [])),
                    suggestions=list(data.get("suggestions", [])),
                    raw=raw,
                )
            except Exception:  # noqa: BLE001
                pass

        return ReflectionResult(
            score=0.7,
            is_satisfactory=True,
            issues=[],
            suggestions=[],
            raw=raw,
        )

    def refine(
        self,
        query: str,
        result: ExecutionResult,
        reflection: ReflectionResult,
    ) -> str:
        prompt = self.refine_template.format(
            query=query,
            answer=result.final_answer,
            score=reflection.score,
            issues=", ".join(reflection.issues) or "none",
            suggestions=", ".join(reflection.suggestions) or "none",
        )
        return self.llm.invoke([{"role": "user", "content": prompt}]) or result.final_answer
