"""
Example / default tools.

A small set of domain-neutral tools included out of the box so that the
framework is usable immediately after install. They serve as reference
implementations of :class:`tools.base.BaseTool` and as the default registry
returned by :func:`create_default_tool_registry`.

Available tools
---------------
- echo              Echo back the provided text.
- calculator        Safely evaluate a simple arithmetic expression.
- current_time      Return the current server time (ISO 8601).
- http_get          Perform a simple HTTP GET and return the response body.
- text_stats        Compute basic statistics (length / line count / words) for a text.
- uuid_generator    Generate one or more UUIDs.
- json_format       Pretty-print a JSON string.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re
import uuid
from typing import Any

from tools.base import BaseTool, ToolParameter
from tools.registry import ToolRegistry


# ============================================================
#  echo — simplest possible tool, useful for sanity checks
# ============================================================
class EchoTool(BaseTool):
    name = "echo"
    description = "Echo back the input text. Useful for debugging and sanity checks."
    parameters = [
        ToolParameter("text", "string", "Text to echo back", required=True),
    ]

    def execute(self, text: str) -> str:
        return text


# ============================================================
#  calculator — sandboxed arithmetic
# ============================================================
class CalculatorTool(BaseTool):
    """Safe arithmetic calculator.

    Only digits, whitespace and ``+ - * / ( ) . %`` are allowed.
    Names from ``__builtins__`` are stripped; ``math`` is exposed read-only.
    """

    name = "calculator"
    description = (
        "Evaluate a simple arithmetic expression and return the numeric result. "
        "Supports + - * /, parentheses, and percent. Example: '(8 + 4) * 2 - 5'"
    )
    parameters = [
        ToolParameter("expression", "string", "Arithmetic expression", required=True),
    ]
    _SAFE = re.compile(r"^[\d\s\+\-\*\/\.\(\)\%]+$")

    def execute(self, expression: str) -> str:
        if not self._SAFE.match(expression):
            return json.dumps(
                {"error": "Expression contains illegal characters"},
                ensure_ascii=False,
            )
        try:
            result = eval(expression, {"__builtins__": {}}, {"math": math})  # noqa: S307
            return json.dumps(
                {"expression": expression, "result": result}, ensure_ascii=False
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {"error": f"Calculation failed: {exc}"}, ensure_ascii=False
            )


# ============================================================
#  current_time
# ============================================================
class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = "Return the current server time in ISO 8601 format (UTC + local)."
    parameters = [
        ToolParameter(
            "format",
            "string",
            "Output format: 'iso' (default) or 'unix' (epoch seconds).",
            required=False,
            default="iso",
            enum=["iso", "unix"],
        ),
    ]

    def execute(self, format: str = "iso") -> str:  # noqa: A002 — keyword name
        now = _dt.datetime.now()
        utc = _dt.datetime.now(_dt.timezone.utc)
        payload = {
            "utc": utc.isoformat(),
            "local": now.isoformat(),
            "timezone": _dt.datetime.now().astimezone().tzname(),
        }
        if format == "unix":
            payload["epoch"] = int(now.timestamp())
        return json.dumps(payload, ensure_ascii=False)


# ============================================================
#  http_get — best-effort HTTP fetcher
# ============================================================
class HttpGetTool(BaseTool):
    name = "http_get"
    description = (
        "Perform an HTTP GET request and return a short preview of the response body. "
        "Use only against trusted endpoints; large responses are truncated."
    )
    parameters = [
        ToolParameter("url", "string", "Absolute URL to fetch (http or https).", required=True),
        ToolParameter(
            "max_chars",
            "number",
            "Maximum number of characters to return (default 2000).",
            required=False,
            default=2000,
        ),
        ToolParameter(
            "timeout",
            "number",
            "Request timeout in seconds (default 10).",
            required=False,
            default=10.0,
        ),
    ]

    def execute(self, url: str, max_chars: int = 2000, timeout: float = 10.0) -> str:
        try:
            import httpx
        except ImportError:
            return json.dumps(
                {"error": "httpx is not installed. Run `pip install httpx`."},
                ensure_ascii=False,
            )
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url)
            body = resp.text or ""
            truncated = len(body) > max_chars
            return json.dumps(
                {
                    "url": url,
                    "status_code": resp.status_code,
                    "content_type": resp.headers.get("content-type", ""),
                    "body_preview": body[:max_chars],
                    "truncated": truncated,
                    "body_length": len(body),
                },
                ensure_ascii=False,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {"error": f"HTTP request failed: {exc}"},
                ensure_ascii=False,
            )


# ============================================================
#  text_stats
# ============================================================
class TextStatsTool(BaseTool):
    name = "text_stats"
    description = (
        "Compute basic statistics for a text: character count, word count, "
        "line count, and approximate CJK + Latin character breakdown."
    )
    parameters = [
        ToolParameter("text", "string", "Text to analyze", required=True),
    ]

    def execute(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        lines = text.splitlines() or [""]
        words = re.findall(r"\b\w+\b", text)
        return json.dumps(
            {
                "characters": len(text),
                "cjk_characters": cjk,
                "words": len(words),
                "lines": len(lines),
                "non_empty_lines": sum(1 for line in lines if line.strip()),
            },
            ensure_ascii=False,
        )


# ============================================================
#  uuid_generator
# ============================================================
class UuidGeneratorTool(BaseTool):
    name = "uuid_generator"
    description = "Generate one or more random UUIDs (version 4)."
    parameters = [
        ToolParameter(
            "count",
            "integer",
            "How many UUIDs to generate (1-100, default 1).",
            required=False,
            default=1,
        ),
    ]

    def execute(self, count: int = 1) -> str:
        count = max(1, min(int(count), 100))
        ids = [str(uuid.uuid4()) for _ in range(count)]
        return json.dumps({"count": count, "uuids": ids}, ensure_ascii=False)


# ============================================================
#  json_format
# ============================================================
class JsonFormatTool(BaseTool):
    name = "json_format"
    description = "Pretty-print a JSON string with a configurable indent."
    parameters = [
        ToolParameter("json_string", "string", "JSON string to format", required=True),
        ToolParameter(
            "indent",
            "integer",
            "Indent width (default 2, 0 for compact).",
            required=False,
            default=2,
        ),
    ]

    def execute(self, json_string: str, indent: int = 2) -> str:
        try:
            data: Any = json.loads(json_string)
        except json.JSONDecodeError as exc:
            return json.dumps(
                {"error": f"Invalid JSON: {exc}"},
                ensure_ascii=False,
            )
        if indent <= 0:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(data, ensure_ascii=False, indent=indent)


# ============================================================
#  factory — returns a ToolRegistry pre-populated with all default tools
# ============================================================
def create_default_tool_registry() -> ToolRegistry:
    """Return a :class:`ToolRegistry` populated with the built-in tools.

    The returned registry is fully self-contained and works without any
    external services. Use it as a quick start or as a reference for
    building your own domain-specific registry::

        from polyagent import create_default_tool_registry
        from agents import SimpleAgent
        from llm_client import build_llm

        registry = create_default_tool_registry()
        agent = SimpleAgent(llm=build_llm(), tool_registry=registry)
        print(agent.run("What time is it?"))
    """
    registry = ToolRegistry()
    for tool in (
        EchoTool(),
        CalculatorTool(),
        CurrentTimeTool(),
        HttpGetTool(),
        TextStatsTool(),
        UuidGeneratorTool(),
        JsonFormatTool(),
    ):
        registry.register_tool(tool)
    return registry


# Backwards-compatible alias (used by older imports / tests).
def create_example_tool_registry() -> ToolRegistry:  # pragma: no cover
    return create_default_tool_registry()
