"""
Intent recognizer (IntentRecognizer).

LLM-first classification with a rule-based fallback. Domain-neutral — the
default intent taxonomy is deliberately generic; subclasses or downstream
applications are expected to add their own intents or override the prompt
for their particular use case.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm_client import PolyAgentLLM


# Default intent taxonomy. Keep it small and generic.
INTENT_TYPES: List[str] = [
    "task_planning",     # user wants a plan / breakdown / multi-step solution
    "data_query",        # user wants to retrieve a specific fact
    "calculation",       # user wants a numeric calculation
    "knowledge_query",   # user wants conceptual / explanatory information
    "tool_use",          # user wants the agent to invoke a tool
    "chitchat",          # greetings / small talk
    "unknown",           # nothing matched
]

# Intent → recommended agent paradigm
INTENT_TO_MODE: Dict[str, str] = {
    "task_planning": "plan_solve",
    "data_query": "react",
    "calculation": "react",
    "knowledge_query": "simple",
    "tool_use": "react",
    "chitchat": "simple",
    "unknown": "simple",
}

# Rule-based fallback (deliberately language-agnostic and conservative).
_RULES: List[tuple] = [
    (re.compile(r"(plan|break\s*down|step\s*by\s*step|how\s+to|how\s+can\s+i)"), "task_planning"),
    (re.compile(r"(calculate|compute|how\s+much|how\s+many|sum|total|average)"), "calculation"),
    (re.compile(r"(what|who|when|where|which|define|explain)"), "knowledge_query"),
    (re.compile(r"(find|search|lookup|fetch|get\s+me|show\s+me)"), "data_query"),
    (re.compile(r"(hi|hello|hey|good\s+(morning|afternoon|evening)|thanks|thank\s+you)"), "chitchat"),
]


@dataclass
class IntentResult:
    intent: str
    confidence: float
    entities: Dict[str, str]
    mode: str
    reasoning: str = ""
    raw_response: str = ""


class IntentRecognizer:
    """Recognize the intent of a user query.

    The default implementation is intentionally domain-neutral. Pass a
    custom ``prompt_template`` and ``intent_types`` to specialize the
    recognizer for a particular application.
    """

    PROMPT = """You are an intent-classification expert. Classify the user's
input into exactly one of the following intents:

{intent_types}

Intent descriptions:
- task_planning:   the user wants a plan, breakdown or step-by-step solution
- data_query:      the user wants a specific piece of data retrieved
- calculation:     the user wants a numeric or symbolic calculation
- knowledge_query: the user wants conceptual / explanatory information
- tool_use:        the user wants a tool or function to be invoked
- chitchat:        greetings or small talk
- unknown:         nothing else fits

Respond with strict JSON only (no prose, no markdown fences):
{{
  "intent": "<one of the intents above>",
  "confidence": <number between 0.0 and 1.0>,
  "entities": {{"<entity>": "<value>"}},
  "reasoning": "<short justification>"
}}

User input: {query}
"""

    def __init__(
        self,
        llm: PolyAgentLLM,
        intent_types: Optional[List[str]] = None,
        intent_to_mode: Optional[Dict[str, str]] = None,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.intent_types = list(intent_types or INTENT_TYPES)
        self.intent_to_mode = dict(intent_to_mode or INTENT_TO_MODE)
        self.prompt_template = prompt_template or self.PROMPT

    def recognize(self, query: str) -> IntentResult:
        # 1) Try rule-based fast path.
        rule_intent = self._rule_match(query)
        if rule_intent:
            return IntentResult(
                intent=rule_intent,
                confidence=0.7,
                entities=self._extract_entities(query),
                mode=self.intent_to_mode.get(rule_intent, "simple"),
                reasoning="[rule match]",
            )

        # 2) Ask the LLM.
        prompt = self.prompt_template.format(
            intent_types="\n".join(f"- {i}" for i in self.intent_types),
            query=query,
        )
        raw = self.llm.invoke([{"role": "user", "content": prompt}]) or ""
        parsed = self._parse_llm_response(raw)
        if parsed:
            intent = parsed.get("intent", "unknown")
            if intent not in self.intent_types:
                intent = "unknown"
            return IntentResult(
                intent=intent,
                confidence=float(parsed.get("confidence", 0.5)),
                entities=parsed.get("entities", {}),
                mode=self.intent_to_mode.get(intent, "simple"),
                reasoning=parsed.get("reasoning", ""),
                raw_response=raw,
            )

        # 3) Fallback.
        return IntentResult(
            intent="unknown",
            confidence=0.0,
            entities={},
            mode="simple",
            reasoning="classification failed, using default",
        )

    def _rule_match(self, query: str) -> Optional[str]:
        q = query.lower()
        for pattern, intent in _RULES:
            if pattern.search(q):
                return intent
        return None

    @staticmethod
    def _parse_llm_response(raw: str) -> Optional[Dict]:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_entities(query: str) -> Dict[str, str]:
        """Lightweight entity extraction.

        Pulls out quoted phrases and tokens that look like identifiers
        (CamelCase, snake_case, kebab-case). Override in a subclass for
        domain-specific needs.
        """
        entities: Dict[str, str] = {}
        quoted = re.findall(r"[\"']([^\"']{2,})[\"']", query)
        if quoted:
            entities["quoted"] = quoted[0]
        for match in re.findall(r"\b[A-Z][A-Za-z0-9]+\b|\b[a-z]+_[a-z_]+\b|\b[a-z]+-[a-z-]+\b", query):
            entities.setdefault("tokens", match)
        return entities
