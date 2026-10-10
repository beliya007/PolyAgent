"""
PolyAgent — A lightweight multi-agent framework.

A pure-Python, modular agent framework built around:
- 4 classic agent paradigms (Simple / ReAct / Plan-and-Solve / Reflection)
- Native OpenAI function calling (no LangChain dependency)
- A central multi-agent orchestrator (Intent → Plan → Execute → Reflect)
- 4-type memory system with multiple backends (memory / SQLite / Qdrant / Neo4j)
- RAG with hybrid retrieval (BM25 + vector) and LLM reranking
- MCP / A2A / ANP communication protocols
- FastAPI server with REST + SSE streaming endpoints
- Built-in evaluation and tracing

The package is intentionally domain-neutral. Domain-specific capabilities
should be added as custom Tools (subclass :class:`tools.base.BaseTool`) and
registered into a :class:`tools.registry.ToolRegistry`.
"""
from __future__ import annotations

__version__ = "1.0.0"

# ----------------------------------------------------------------------
# LLM client
# ----------------------------------------------------------------------
from llm_client import PolyAgentLLM, build_llm, tool_to_openai_schema

# ----------------------------------------------------------------------
# Core message types
# ----------------------------------------------------------------------
from core.message import Message, ToolCall, ToolResult

# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
from tools import (
    AsyncToolExecutor,
    BaseTool,
    FunctionTool,
    ToolParameter,
    ToolRegistry,
    create_default_tool_registry,
)

# ----------------------------------------------------------------------
# Agents (4 paradigms)
# ----------------------------------------------------------------------
from agents import (
    BaseAgent,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
)

# ----------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------
from memory import (
    BaseMemory,
    EpisodicMemory,
    MemoryItem,
    MemoryManager,
    PerceptualMemory,
    SemanticMemory,
    WorkingMemory,
)

# ----------------------------------------------------------------------
# RAG
# ----------------------------------------------------------------------
from rag import (
    Chunk,
    Document,
    DocumentLoader,
    Embedder,
    HybridSearch,
    LLMReranker,
    RAGTool,
    TextChunker,
    VectorRecord,
    VectorStore,
)

# ----------------------------------------------------------------------
# Context engineering
# ----------------------------------------------------------------------
from context import ContextBuilder, ContextCompressor, ContextConfig

# ----------------------------------------------------------------------
# Orchestrator (central multi-agent routing)
# ----------------------------------------------------------------------
from orchestrator import (
    CentralRouter,
    CentralRouterResult,
    Executor,
    ExecutionResult,
    IntentRecognizer,
    IntentResult,
    Plan,
    Planner,
    PlanStep,
    Reflector,
    ReflectionResult,
    StepResult,
    chat,
    get_router,
)

# ----------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------
try:
    from protocols import A2AClient, A2AServer, A2ATool, ANPDiscovery, ANPRouter, MCPClient, MCPServer, MCPTool
    _PROTOCOLS_AVAILABLE = True
except Exception:  # pragma: no cover
    _PROTOCOLS_AVAILABLE = False

# ----------------------------------------------------------------------
# Evaluation & observability
# ----------------------------------------------------------------------
from evaluation import Evaluator, EvaluationResult
from observability import Tracer, TraceEvent

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
__all__ = [
    "__version__",
    # LLM
    "PolyAgentLLM",
    "build_llm",
    "tool_to_openai_schema",
    # Core types
    "Message",
    "ToolCall",
    "ToolResult",
    # Tools
    "BaseTool",
    "FunctionTool",
    "ToolParameter",
    "ToolRegistry",
    "create_default_tool_registry",
    "AsyncToolExecutor",
    # Agents
    "BaseAgent",
    "SimpleAgent",
    "ReActAgent",
    "PlanAndSolveAgent",
    "ReflectionAgent",
    # Memory
    "BaseMemory",
    "MemoryItem",
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "PerceptualMemory",
    "MemoryManager",
    # RAG
    "Document",
    "DocumentLoader",
    "Chunk",
    "TextChunker",
    "Embedder",
    "VectorRecord",
    "VectorStore",
    "HybridSearch",
    "LLMReranker",
    "RAGTool",
    # Context
    "ContextBuilder",
    "ContextCompressor",
    "ContextConfig",
    # Orchestrator
    "IntentRecognizer",
    "IntentResult",
    "Planner",
    "Plan",
    "PlanStep",
    "Executor",
    "ExecutionResult",
    "StepResult",
    "Reflector",
    "ReflectionResult",
    "CentralRouter",
    "CentralRouterResult",
    "get_router",
    "chat",
    # Evaluation
    "Evaluator",
    "EvaluationResult",
    # Observability
    "Tracer",
    "TraceEvent",
    # Protocols
    "MCPClient",
    "MCPServer",
    "MCPTool",
    "A2AClient",
    "A2AServer",
    "A2ATool",
    "ANPDiscovery",
    "ANPRouter",
]
