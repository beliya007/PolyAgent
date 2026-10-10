"""
FastAPI router — centralised multi-agent service.

Endpoints
---------
- POST /chat                       full centralised routing pipeline
- POST /chat/{simple|react|plan-solve|reflection}   paradigm-specific shortcuts
- GET  /stream/chat                SSE streaming of the full pipeline
- GET  /stream/chat/reasoning      SSE streaming of the model's reasoning trace
- POST /intent                     intent recognition only
- POST /plan                       plan generation only
- GET  /agents                     list supported agent paradigms
- GET  /tools                      list registered tools
- Memory, RAG, system stats endpoints
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llm_client import build_llm
from orchestrator import get_router


router = APIRouter()


# ============================================================
#  Request / response models
# ============================================================
class ChatRequest(BaseModel):
    query: str = Field(..., description="User question")
    mode: str = Field(
        "auto",
        description="auto | simple | react | plan_solve | reflection",
    )
    session_id: Optional[str] = Field(None, description="Conversation session id")
    enable_reflection: bool = Field(True, description="Run reflection on the answer")


class IntentRequest(BaseModel):
    query: str


class PlanRequest(BaseModel):
    query: str
    mode: str = "auto"


class MemoryAddRequest(BaseModel):
    content: str
    memory_type: str = "episodic"
    importance: float = 0.5
    metadata: Optional[Dict[str, Any]] = None


class MemorySearchRequest(BaseModel):
    query: str
    memory_type: Optional[str] = None
    limit: int = 5


class RAGIngestRequest(BaseModel):
    content: Optional[str] = None
    file_path: Optional[str] = None
    doc_id: Optional[str] = "raw"


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


# ============================================================
#  Core: centralised routing
# ============================================================
@router.post("/chat", summary="Centralised multi-agent routing (full pipeline)")
async def central_chat(req: ChatRequest):
    """Full pipeline: intent -> plan -> execute -> reflect -> return."""
    try:
        r = get_router()
        result = r.route(req.query, mode=req.mode, session_id=req.session_id)
        return result.to_dict()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"routing failed: {e}")


# ============================================================
#  Paradigm shortcuts
# ============================================================
@router.post("/chat/simple", summary="Simple Agent")
async def chat_simple(req: ChatRequest):
    return {"mode": "simple", "answer": get_router().simple_agent.run(req.query)}


@router.post("/chat/react", summary="ReAct Agent")
async def chat_react(req: ChatRequest):
    return {"mode": "react", "answer": get_router().react_agent.run(req.query)}


@router.post("/chat/plan-solve", summary="Plan-and-Solve Agent")
async def chat_plan_solve(req: ChatRequest):
    return {
        "mode": "plan_solve",
        "answer": get_router().plan_solve_agent.run(req.query),
    }


@router.post("/chat/reflection", summary="Reflection Agent")
async def chat_reflection(req: ChatRequest):
    return {
        "mode": "reflection",
        "answer": get_router().reflection_agent.run(req.query),
    }


# ============================================================
#  Stage-wise endpoints
# ============================================================
@router.post("/intent", summary="Intent recognition only")
async def recognize_intent(req: IntentRequest):
    r = get_router()
    intent = r.intent_recognizer.recognize(req.query)
    return {
        "intent": intent.intent,
        "confidence": intent.confidence,
        "entities": intent.entities,
        "recommended_mode": intent.mode,
        "reasoning": intent.reasoning,
    }


@router.post("/plan", summary="Generate plan from query")
async def plan_steps(req: PlanRequest):
    r = get_router()
    intent = r.intent_recognizer.recognize(req.query)
    plan = r.planner.plan(intent, req.query)
    return {
        "intent": intent.intent,
        "mode": intent.mode,
        "plan": plan.to_dict(),
    }


# ============================================================
#  Paradigm catalog
# ============================================================
@router.get("/agents", summary="Available agent paradigms")
async def list_agents():
    return {
        "paradigms": [
            {
                "name": "Simple",
                "file": "agents/simple_agent.py",
                "desc": "Single-turn LLM + native function calling.",
            },
            {
                "name": "ReAct",
                "file": "agents/react_agent.py",
                "desc": "Thought -> Action -> Observation loop.",
            },
            {
                "name": "Plan-and-Solve",
                "file": "agents/plan_solve_agent.py",
                "desc": "Planner -> Executor two-stage pipeline.",
            },
            {
                "name": "Reflection",
                "file": "agents/reflection_agent.py",
                "desc": "Produce -> critique -> refine, up to N rounds.",
            },
            {
                "name": "Central (full pipeline)",
                "file": "orchestrator/central_router.py",
                "desc": "Intent -> plan -> execute -> reflect, fully automatic.",
            },
        ],
    }


# ============================================================
#  Tools
# ============================================================
@router.get("/tools", summary="Registered tools")
async def list_tools():
    r = get_router()
    return {
        "tools": r.tools.list_tools(),
        "descriptions": r.tools.get_tools_description(),
    }


# ============================================================
#  Memory
# ============================================================
@router.post("/memory/add", summary="Add a memory item")
async def memory_add(req: MemoryAddRequest):
    r = get_router()
    rid = r.memory.add(
        req.content,
        memory_type=req.memory_type,
        importance=req.importance,
        metadata=req.metadata,
    )
    return {"id": rid, "type": req.memory_type}


@router.post("/memory/search", summary="Search memory")
async def memory_search(req: MemorySearchRequest):
    r = get_router()
    items = r.memory.search(
        req.query, memory_type=req.memory_type, limit=req.limit
    )
    return {
        "query": req.query,
        "results": [
            {**item.to_dict(), "age_min": item.age_min()} for item in items
        ],
    }


@router.get("/memory/stats", summary="Memory statistics")
async def memory_stats():
    return get_router().memory.stats()


@router.post(
    "/memory/consolidate", summary="Consolidate working memory into episodic"
)
async def memory_consolidate(threshold: float = 0.6):
    r = get_router()
    n = r.memory.consolidate(
        from_type="working", to_type="episodic", importance_threshold=threshold
    )
    return {"promoted": n}


@router.post("/memory/forget", summary="Forget low-importance memories")
async def memory_forget(threshold: float = 0.2):
    r = get_router()
    n = r.memory.forget(strategy="importance_based", threshold=threshold)
    return {"forgotten": n}


@router.post("/memory/save", summary="Persist memory to disk")
async def memory_save():
    r = get_router()
    result = r.memory.save_all()
    return {"saved": result}


# ============================================================
#  RAG
# ============================================================
@router.post("/rag/ingest", summary="Ingest a document into RAG")
async def rag_ingest(req: RAGIngestRequest):
    r = get_router()
    if not r.rag:
        raise HTTPException(400, "RAG is not enabled")
    if req.file_path:
        n = r.rag.ingest_file(req.file_path)
    elif req.content:
        n = r.rag.ingest_text(req.content, doc_id=req.doc_id or "raw")
    else:
        raise HTTPException(400, "either `content` or `file_path` is required")
    return {"ingested_chunks": n}


@router.post("/rag/search", summary="Run a RAG retrieval")
async def rag_search(req: RAGSearchRequest):
    r = get_router()
    if not r.rag:
        raise HTTPException(400, "RAG is not enabled")
    results = r.rag.retrieve(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}


@router.get("/rag/stats", summary="RAG statistics")
async def rag_stats():
    r = get_router()
    return r.rag.stats() if r.rag else {"enabled": False}


# ============================================================
#  System
# ============================================================
@router.get("/llm/stats", summary="LLM call statistics")
async def llm_stats():
    return build_llm().stats()


@router.get("/health", summary="Health check")
async def health():
    return {"status": "healthy", "service": "hello-agents"}


@router.get("/stats", summary="System overview")
async def system_stats():
    return get_router().stats()
