# 🤖 PolyAgent

> A lightweight, modular multi-agent framework in pure Python.
> Pure-native OpenAI function calling + FastAPI. **No LangChain dependency.**
> Built around four classic agent paradigms, a central multi-agent orchestrator,
> pluggable memory / RAG / protocols / tools.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)]()
[![License](https://img.shields.io/badge/license-MIT-orange)]()

---

## 🎯 What is PolyAgent?

A **domain-neutral** agent framework. The framework itself knows nothing about
any specific business problem — you build your own application by writing
`BaseTool` subclasses and registering them into a `ToolRegistry`. The
provided tools, prompts and knowledge base are merely *examples* to get you
running in seconds.

### Highlights

| Capability | Module | Notes |
|---|---|---|
| **4 agent paradigms** | `agents/` | Simple / ReAct / Plan-and-Solve / Reflection |
| **Central orchestrator** | `orchestrator/` | intent → plan → execute → reflect |
| **Pluggable tool system** | `tools/` | Strategy + Registry, native function calling |
| **Memory** | `memory/` | 4 types (working / episodic / semantic / perceptual) + JSONL persistence |
| **RAG** | `rag/` | hybrid (vector × 0.6 + BM25 × 0.4) + LLM reranker |
| **Context engineering** | `context/` | Builder + compressor with token budget |
| **Communication protocols** | `protocols/` | MCP / A2A / ANP |
| **Tool chains & async** | `tools/chain.py` `tools/async_executor.py` | multi-step pipelines + parallel execution |
| **FastAPI server** | `routers/` | REST + SSE streaming |
| **Evaluation & tracing** | `evaluation/` `observability/` | quality scoring + trace persistence |
| **Structured output** | `orchestrator/` | JSON-typed plans & intent results, with parse fallbacks |
| **Multi-layer fault tolerance** | `orchestrator/` | reflection + auto retry + tool-level fallbacks |

---

## 🏗️ Central multi-agent pipeline

```
                          ┌───────────────────────────────┐
                          │     user input (any text)     │
                          └──────────────┬────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────┐
        │  [1] Intent recognition — IntentRecognizer (LLM + rule)│
        │      → intent + entities + recommended mode            │
        └────────────────────────────────┬───────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────┐
        │  [2] Planning — Planner (JSON-typed plan)              │
        │      → 3–7 executable steps with dependencies          │
        └────────────────────────────────┬───────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────┐
        │  [3] Execution — Executor (parallel batches when safe) │
        │      → StepResults + collected context                 │
        └────────────────────────────────┬───────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────┐
        │  [4] Reflection — Reflector (LLM scoring)              │
        │      → score, issues, suggestions                      │
        │      → auto-retry on low score                         │
        └────────────────────────────────┬───────────────────────┘
                                         ▼
        ┌────────────────────────────────────────────────────────┐
        │  [5] Result — CentralRouterResult                      │
        │      → intent / plan / execution / reflection / answer │
        └────────────────────────────────────────────────────────┘
```

---

## 🧩 Design patterns at a glance

### 1. Strategy + Registry

Tools are independent strategies that share a common interface (`BaseTool`).
You can register your own without touching the agent core.

```python
from tools.base import BaseTool, ToolParameter
from tools.registry import ToolRegistry

class MyTool(BaseTool):
    name = "my_tool"
    description = "do something domain-specific"
    parameters = [ToolParameter("query", "string", "input", required=True)]

    def execute(self, query: str) -> str:
        return f"handled: {query}"

registry = ToolRegistry()
registry.register_tool(MyTool())
```

The default registry includes a small, domain-neutral set you can use as
a starting point: `echo`, `calculator`, `current_time`, `http_get`,
`text_stats`, `uuid_generator`, `json_format`.

### 2. Structured Output

The Planner, IntentRecognizer and Reflector all request strict JSON from
the LLM and parse with `re.search` → `json.loads` → `ast.literal_eval`
fallback chain. The Executor then walks the typed plan — no string
splicing downstream.

### 3. Multi-layer fault tolerance

- L1 — tool-level fallbacks: missing backend → mock data, unsafe expression → reject.
- L2 — reflection: low score triggers `refine()` and re-scores.
- L3 — automatic retry with a configurable `max_retry` cap to prevent loops.

### 4. Strategy + Chain + Parallel

`ToolChain` runs multiple tools in order with `{var}`-style templates and
an `on_step` hook. `AsyncToolExecutor` runs tools (or whole chains) in
parallel via a thread pool. The Executor uses both to batch independent
exploratory steps.

---

## 📦 Module layout

```
polyagent/
├── main.py                       FastAPI entry
├── llm_client.py                 LLM client (OpenAI compatible)
├── config.py                     .env-driven config
├── demo.py                       full feature tour
│
├── core/                         core types (Message, ToolCall, ...)
├── agents/                       4 paradigms
│   ├── base.py
│   ├── simple_agent.py
│   ├── react_agent.py
│   ├── plan_solve_agent.py
│   └── reflection_agent.py
├── memory/                       4 memory types + persistence
├── rag/                          loader / chunker / embedder / vector store / hybrid / reranker
├── context/                      ContextBuilder + Compressor
├── orchestrator/                 central router + planner + executor + reflector
├── routers/                      FastAPI REST + SSE
├── tools/                        BaseTool + Registry + Chain + Async
│   ├── base.py                   BaseTool, FunctionTool, ToolResult
│   ├── registry.py
│   ├── chain.py
│   ├── async_executor.py
│   ├── example_tools.py          built-in neutral tools
│   ├── chains.py                 pre-built chains
│   ├── memory_tool.py            Memory-as-a-Tool
│   ├── note_tool.py              scratchpad notes
│   └── terminal_tool.py          shell helper
├── protocols/                    MCP / A2A / ANP
├── evaluation/                   Evaluator
├── observability/                Tracer
│
├── data/                         memory + knowledge base
├── tests/                        unit + e2e
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick start

### 1. Install

```bash
cd polyagent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# edit .env: set LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_ID)
```

### 3. Run the demo

```bash
python demo.py             # built-in tools only (no LLM needed)
python demo.py --full      # full tour
python demo.py --central   # central router only
python demo.py --memory    # memory system
python demo.py --rag       # RAG
python demo.py --eval      # evaluation
```

### 4. Run the server

```bash
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open <http://localhost:8000/docs> for the Swagger UI.

### 5. Use it as a library

```python
from polyagent import (
    PolyAgentLLM, build_llm,
    ToolRegistry, create_default_tool_registry,
    ReActAgent,
)

llm = build_llm()
registry = create_default_tool_registry()

# Add your own tools
from tools.base import BaseTool, ToolParameter

class GreetTool(BaseTool):
    name = "greet"
    description = "say hello in the user's language"
    parameters = [ToolParameter("name", "string", "user name", required=True)]
    def execute(self, name: str) -> str:
        return f"hello, {name}!"

registry.register_tool(GreetTool())

agent = ReActAgent(llm=llm, tool_registry=registry)
print(agent.run("Greet Alice in English."))
```

### 6. Or use the central router

```python
from polyagent import chat

result = chat("What is 12 * 8 + 5?")
print(result["final_answer"])
print(result["intent"], result["used_paradigm"], result["reflection"]["score"])
```

---

## 🧪 Tests

```bash
python -m pytest tests/test_all.py -v --tb=short
```

E2E tests require a running server (`python main.py`) and a valid LLM key.

```bash
python main.py &
python tests/e2e/test_api.py
python tests/e2e/test_central.py
```

---

## 🧠 Memory

```python
from memory import MemoryManager

mm = MemoryManager(user_id="alice", enable_persistence=True, persist_dir="./data/memory")
mm.add_working("current task: refactor module X", importance=0.8)
mm.add_episodic("user said they prefer concise answers", importance=0.7, event_type="preference")
mm.add_semantic("project uses Python 3.11", importance=0.9, concept="env", domain="dev")
mm.add_perceptual("uploaded architecture.png", importance=0.6, modality="image")

results = mm.search("module", limit=5)
mm.consolidate(from_type="working", to_type="episodic", importance_threshold=0.6)
mm.forget(threshold=0.2)
```

---

## 📚 RAG

```python
from rag import RAGTool
from llm_client import build_llm

rag = RAGTool(build_llm(), namespace="my_kb", enable_hybrid=True, enable_reranker=True)
rag.ingest_text(open("notes.md").read(), doc_id="notes")
rag.ingest_file("manual.pdf")
rag.ingest_directory("./docs")

results = rag.retrieve("nightly deployment", top_k=5, use_rerank=True)
context = rag.build_context("nightly deployment")
```

Pipeline: `loader → chunker → embedder → vector store → hybrid search (vector × 0.6 + BM25 × 0.4) → optional LLM reranker`.

---

## 🔌 Protocols

| Protocol | What it gives you |
|---|---|
| **MCP** | standardized tool/resource exchange with external MCP servers |
| **A2A** | direct agent-to-agent skill invocation with auth + signing |
| **ANP** | service discovery, capability-based routing, identity + mesh |

All three are domain-neutral and ship as ready-to-use modules under `protocols/`.

---

## 📜 License

MIT

