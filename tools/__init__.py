"""tools package — pluggable tools, registry, chains and async execution.

Public surface:

    Base classes
    ------------
    BaseTool            Abstract base class for all tools.
    FunctionTool        Wrap a plain function as a tool.
    ToolParameter       Single parameter descriptor.
    ToolResult          Structured result of a tool execution.

    Registry
    --------
    ToolRegistry        Centralised strategy / registry.
    create_default_tool_registry()
                        Convenience factory that registers the built-in tools.

    Chain / async
    -------------
    ChainStep           A single step in a tool chain.
    ToolChain           Ordered execution of multiple tools.
    ToolChainManager    Registry of named chains.
    AsyncToolExecutor   Async / parallel execution of tools and chains.

    Built-in tools
    --------------
    The ``example_tools`` module ships a small, domain-neutral set:
    echo, calculator, current_time, http_get, text_stats,
    uuid_generator, json_format. Use them as a reference or as a quick
    starting point for your own domain-specific tools.
"""
from tools.async_executor import AsyncToolExecutor
from tools.base import (
    BaseTool,
    FunctionTool,
    ToolParameter,
    ToolResult,
    ToolSchema,
)
from tools.chain import ChainStep, ToolChain, ToolChainManager
from tools.chains import (
    build_all_chains,
    build_format_report_chain,
    build_lookup_chain,
    build_summarize_chain,
    parallel_echo,
    parallel_format,
    parallel_stats,
    register_default_chains,
)
from tools.example_tools import (
    CalculatorTool,
    CurrentTimeTool,
    EchoTool,
    HttpGetTool,
    JsonFormatTool,
    TextStatsTool,
    UuidGeneratorTool,
    create_default_tool_registry,
)
from tools.memory_tool import MemoryTool
from tools.note_tool import NoteTool
from tools.registry import ToolRegistry
from tools.terminal_tool import TerminalTool

# MCP 协议工具（第十章）
try:
    from protocols.mcp import MCPTool, MCP_AVAILABLE
except ImportError:  # pragma: no cover
    MCPTool = None
    MCP_AVAILABLE = False


__all__ = [
    # Base abstractions
    "BaseTool",
    "FunctionTool",
    "ToolParameter",
    "ToolSchema",
    "ToolResult",
    "ToolRegistry",
    # Chains (sync)
    "ChainStep",
    "ToolChain",
    "ToolChainManager",
    # Async execution
    "AsyncToolExecutor",
    # Built-in example tools
    "EchoTool",
    "CalculatorTool",
    "CurrentTimeTool",
    "HttpGetTool",
    "TextStatsTool",
    "UuidGeneratorTool",
    "JsonFormatTool",
    "create_default_tool_registry",
    # Pre-built chains and parallel templates
    "build_summarize_chain",
    "build_lookup_chain",
    "build_format_report_chain",
    "build_all_chains",
    "register_default_chains",
    "parallel_stats",
    "parallel_echo",
    "parallel_format",
    # Memory tool
    "MemoryTool",
    # Context-engineering tools
    "NoteTool",
    "TerminalTool",
    # MCP protocol tools
    "MCPTool",
    "MCP_AVAILABLE",
]
