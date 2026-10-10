"""
FastAPI entry point for the PolyAgent service.

A pure-Python, modular multi-agent framework:
  - 4 agent paradigms (Simple / ReAct / Plan-and-Solve / Reflection)
  - Centralised orchestrator (intent -> plan -> execute -> reflect)
  - 4-type memory system with multiple backends
  - Hybrid-search RAG with optional LLM reranking
  - MCP / A2A / ANP communication protocols
  - REST + SSE streaming endpoints
  - Built-in evaluation and tracing

Run with:
    python main.py
or:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.agent import router as agent_router
from routers.stream import router as stream_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("PolyAgent service starting")
    logger.info("Pipeline: intent -> plan -> execute -> reflect")
    logger.info(
        "Capabilities: 4 paradigms, 4 memory types (persistent), "
        "RAG (hybrid search), context engineering, SSE streaming"
    )
    logger.info("=" * 60)
    try:
        from orchestrator import get_router
        get_router()
        logger.info("CentralRouter warm-up OK")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"CentralRouter warm-up failed: {e}")
    yield
    logger.info("PolyAgent service shutting down")


app = FastAPI(
    title="PolyAgent",
    version="1.0.0",
    description="""
A modular multi-agent framework that ships with:

- 4 agent paradigms: Simple, ReAct, Plan-and-Solve, Reflection
- 4 memory types: working, episodic, semantic, perceptual (JSONL persistence)
- Hybrid-search RAG: vector + BM25 + optional LLM reranker
- Communication protocols: MCP, A2A, ANP
- Centralised orchestrator: intent -> plan -> execute -> reflect
- SSE streaming of the full pipeline
- Built-in evaluation and tracing
""",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router, prefix="/api/agent", tags=["agent"])
app.include_router(stream_router, prefix="/api/agent", tags=["stream"])


@app.get("/")
async def root():
    return {
        "service": "polyagent",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/agent/health",
        "stats": "/api/agent/stats",
        "sse_chat": "/api/agent/stream/chat",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
