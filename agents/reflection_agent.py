"""
ReflectionAgent — produce -> critique -> refine, repeated.

Use when the output needs iterative polishing (writing, design, code).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from agents.base import BaseAgent
from core.message import Message
from llm_client import PolyAgentLLM


DEFAULT_INITIAL_PROMPT = """You are an expert in the requested domain. Produce a
high-quality initial answer to the task below.

Task: {task}

Write the answer directly. No preamble.
"""


DEFAULT_REFLECT_PROMPT = """You are a strict reviewer. Evaluate the following
draft answer to the original task.

# Original task
{task}

# Draft
{code}

Evaluate on:
1. Does it address the task?
2. Are the key elements present?
3. Are there obvious errors or missing pieces?

If the draft is already strong, reply exactly: NO_IMPROVEMENT
Otherwise, list concrete issues and suggestions.
"""


DEFAULT_REFINE_PROMPT = """You are the original author. Apply the reviewer's
feedback to produce a refined answer.

# Original task
{task}

# Previous draft
{last_code_attempt}

# Reviewer feedback
{feedback}

Write the refined answer directly. No preamble.
"""


@dataclass
class MemoryRecord:
    type: str  # 'execution' / 'reflection'
    content: str


class Memory:
    """Tiny in-memory trail of execution and reflection rounds."""

    def __init__(self):
        self.records: List[MemoryRecord] = []

    def add(self, record_type: str, content: str) -> None:
        self.records.append(MemoryRecord(type=record_type, content=content))

    def trajectory(self) -> str:
        out = []
        for r in self.records:
            if r.type == "execution":
                out.append(f"--- previous draft ---\n{r.content}\n")
            else:
                out.append(f"--- reviewer feedback ---\n{r.content}\n")
        return "\n".join(out).strip()

    def last_execution(self) -> Optional[str]:
        for r in reversed(self.records):
            if r.type == "execution":
                return r.content
        return None


class ReflectionAgent(BaseAgent):
    def __init__(
        self,
        name: str = "ReflectionAgent",
        llm: Optional[PolyAgentLLM] = None,
        tool_registry=None,
        max_iterations: int = 3,
        system_prompt: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        reflect_prompt: Optional[str] = None,
        refine_prompt: Optional[str] = None,
    ):
        if llm is None:
            from llm_client import build_llm
            llm = build_llm()
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or "You are an iterative refinement expert.",
            tool_registry=tool_registry,
        )
        self.memory = Memory()
        self.max_iterations = max_iterations
        self.initial_prompt = initial_prompt or DEFAULT_INITIAL_PROMPT
        self.reflect_prompt = reflect_prompt or DEFAULT_REFLECT_PROMPT
        self.refine_prompt = refine_prompt or DEFAULT_REFINE_PROMPT

    def _call(self, prompt: str) -> str:
        return self.llm.invoke([{"role": "user", "content": prompt}]) or ""

    def run(self, input_text: str, **kwargs) -> str:
        print("\nProducing initial draft...")
        initial = self._call(self.initial_prompt.format(task=input_text))
        self.memory.add("execution", initial)

        for i in range(1, self.max_iterations + 1):
            print(f"\n--- round {i}/{self.max_iterations} ---")
            feedback = self._call(
                self.reflect_prompt.format(
                    task=input_text, code=self.memory.last_execution()
                )
            )
            self.memory.add("reflection", feedback)
            print(f"Feedback: {feedback[:150]}...")

            upper = feedback.upper()
            if "NO_IMPROVEMENT" in upper or "NO IMPROVEMENT" in upper:
                print("Reviewer is satisfied.")
                break

            refined = self._call(
                self.refine_prompt.format(
                    task=input_text,
                    last_code_attempt=self.memory.last_execution(),
                    feedback=feedback,
                )
            )
            self.memory.add("execution", refined)
            print(f"Refined: {refined[:150]}...")

        final = self.memory.last_execution() or ""
        self.add_message(Message.user(input_text))
        self.add_message(Message.assistant(final))
        return final

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        yield "Producing initial draft...\n"
        initial = self._call(self.initial_prompt.format(task=input_text))
        self.memory.add("execution", initial)
        yield f"Initial:\n{initial}\n"

        final = initial
        for i in range(1, self.max_iterations + 1):
            yield f"\n--- round {i} ---\n"
            feedback = self._call(
                self.reflect_prompt.format(
                    task=input_text, code=self.memory.last_execution()
                )
            )
            self.memory.add("reflection", feedback)
            yield f"Feedback: {feedback}\n"
            upper = feedback.upper()
            if "NO_IMPROVEMENT" in upper or "NO IMPROVEMENT" in upper:
                yield "Reviewer is satisfied.\n"
                final = self.memory.last_execution() or ""
                break
            refined = self._call(
                self.refine_prompt.format(
                    task=input_text,
                    last_code_attempt=self.memory.last_execution(),
                    feedback=feedback,
                )
            )
            self.memory.add("execution", refined)
            final = refined
            yield f"Refined:\n{refined}\n"

        self.add_message(Message.user(input_text))
        self.add_message(Message.assistant(final))
