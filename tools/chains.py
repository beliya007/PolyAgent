"""
Pre-built tool chains and parallel task templates.

These serve as worked examples of how to compose multiple :class:`BaseTool`
implementations into reusable pipelines. They are intentionally generic and
do not assume any particular business domain.

Chains
------
- summarize      text_stats → echo (quick reporting loop)
- lookup         uuid_generator → echo (idempotent id generation loop)
- format_report  json_format → text_stats (validate + summarize a JSON document)

Parallel templates
------------------
- parallel_stats     run text_stats for several texts in one batch
- parallel_echo      run echo for several inputs in one batch
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from tools.base import BaseTool
from tools.chain import ToolChain, ToolChainManager
from tools.registry import ToolRegistry


# ============================================================
#  Synchronous chains
# ============================================================
def build_summarize_chain() -> ToolChain:
    """text_stats → echo.

    Useful for getting a quick numeric summary of a text and echoing it back
    so an LLM can reason about the result.
    """
    return (
        ToolChain(name="summarize", description="Compute text statistics and echo them back.")
        .add_step(
            "text_stats",
            "{input}",
            output_key="stats",
            name="compute_stats",
            description="Compute character/word/line statistics.",
        )
        .add_step(
            "echo",
            "Stats: {stats}",
            output_key="report",
            name="report",
            description="Echo the report so the LLM can see it.",
        )
    )


def build_lookup_chain() -> ToolChain:
    """uuid_generator → echo.

    Demonstrates how to take a generated value and feed it into a downstream
    step.
    """
    return (
        ToolChain(name="lookup", description="Generate a fresh UUID and echo it.")
        .add_step(
            "uuid_generator",
            "1",
            output_key="new_id",
            name="make_id",
            description="Generate a UUID.",
        )
        .add_step(
            "echo",
            "Generated: {new_id}",
            output_key="report",
            name="report",
            description="Echo the new ID back.",
        )
    )


def build_format_report_chain() -> ToolChain:
    """json_format → text_stats.

    Validates and pretty-prints a JSON document, then reports its size.
    """
    return (
        ToolChain(
            name="format_report",
            description="Pretty-print a JSON document and report its size.",
        )
        .add_step(
            "json_format",
            "{input}",
            output_key="pretty",
            name="pretty",
            description="Re-format JSON for readability.",
        )
        .add_step(
            "text_stats",
            "{pretty}",
            output_key="report",
            name="size",
            description="Report character/word/line counts.",
        )
    )


def build_all_chains() -> List[ToolChain]:
    return [
        build_summarize_chain(),
        build_lookup_chain(),
        build_format_report_chain(),
    ]


def register_default_chains(registry: ToolRegistry) -> ToolChainManager:
    """Register the built-in chains into a registry and return its manager."""
    mgr = ToolChainManager(registry)
    for chain in build_all_chains():
        mgr.register(chain)
    return mgr


# ============================================================
#  Parallel task templates
# ============================================================
def parallel_stats(texts: Sequence[str]) -> List[Dict[str, Any]]:
    """Build a parallel task list that runs :class:`TextStatsTool` on each text."""
    return [{"tool_name": "text_stats", "input_data": {"text": t}} for t in texts]


def parallel_echo(texts: Sequence[str]) -> List[Dict[str, Any]]:
    """Build a parallel task list that runs :class:`EchoTool` on each text."""
    return [{"tool_name": "echo", "input_data": t} for t in texts]


def parallel_format(json_strings: Sequence[str]) -> List[Dict[str, Any]]:
    """Build a parallel task list that runs :class:`JsonFormatTool` on each string."""
    return [
        {"tool_name": "json_format", "input_data": {"json_string": s}}
        for s in json_strings
    ]
