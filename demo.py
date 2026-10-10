"""
PolyAgent demo — full pipeline tour.

Demonstrates (all in a domain-neutral way):

  1. Tools           — built-in default tool registry
  2. Agents          — 4 paradigms (Simple / ReAct / Plan-and-Solve / Reflection)
  3. Memory          — 4 memory types + JSONL persistence
  4. RAG             — ingest + hybrid retrieval
  5. Context         — ContextBuilder + compressor
  6. Evaluation      — Evaluator
  7. Central router  — full intent -> plan -> execute -> reflect pipeline

Run modes:

  python demo.py                # tools only (no LLM required)
  python demo.py --full         # everything (LLM required)
  python demo.py --central      # only the central router
  python demo.py --memory       # only memory
  python demo.py --rag          # only RAG
  python demo.py --eval         # only evaluation
"""
from __future__ import annotations

import sys
import time


def banner(title: str, char: str = "="):
    line = char * 70
    print(f"\n{line}\n{title}\n{line}")


# ============================================================
#  1. Tools
# ============================================================
def demo_tools():
    banner("1. Built-in tools (no LLM required)")
    from tools import create_default_tool_registry

    registry = create_default_tool_registry()
    print(f"\nRegistered tools ({len(registry)}):")
    print(registry.get_tools_description())

    print("\n--- calculator ---")
    print(registry.execute_tool("calculator", {"expression": "(8 + 4) * 2 - 5"}))

    print("\n--- text_stats ---")
    print(registry.execute_tool("text_stats", {"text": "Hello world!\nThis is a test."}))

    print("\n--- uuid_generator ---")
    print(registry.execute_tool("uuid_generator", {"count": 3}))

    print("\n--- json_format ---")
    print(registry.execute_tool(
        "json_format",
        {"json_string": '{"a":1,"b":[2,3]}', "indent": 2},
    ))

    print("\n--- current_time ---")
    print(registry.execute_tool("current_time", {"format": "iso"}))


# ============================================================
#  2. Agent paradigms
# ============================================================
def demo_agents():
    banner("2. Agent paradigms (LLM required)")
    from llm_client import build_llm
    from tools import create_default_tool_registry

    llm = build_llm()
    registry = create_default_tool_registry()

    print("\n--- Simple Agent ---")
    from agents import SimpleAgent
    simple = SimpleAgent(llm=llm, tool_registry=registry)
    print("Q: What is 12 * 8?")
    print("A:", simple.run("What is 12 * 8?")[:200])

    print("\n--- ReAct Agent ---")
    from agents import ReActAgent
    react = ReActAgent(llm=llm, tool_registry=registry)
    print("Q: How many words are in the sentence 'The quick brown fox jumps'?")
    print("A:", react.run(
        "How many words are in the sentence 'The quick brown fox jumps'?"
    )[:200])

    print("\n--- Plan-and-Solve Agent ---")
    from agents import PlanAndSolveAgent
    ps = PlanAndSolveAgent(llm=llm, tool_registry=registry)
    print("Q: Generate two UUIDs and tell me their total character count.")
    print("A:", ps.run(
        "Generate two UUIDs and tell me their total character count."
    )[:200])

    print("\n--- Reflection Agent ---")
    from agents import ReflectionAgent
    ref = ReflectionAgent(llm=llm, max_iterations=2)
    print("Q: Write a 2-sentence product tagline for an open-source agent framework.")
    print("A:", ref.run(
        "Write a 2-sentence product tagline for an open-source agent framework."
    )[:200])


# ============================================================
#  3. Memory
# ============================================================
def demo_memory():
    banner("3. Memory system (with JSONL persistence)")
    from memory import MemoryManager

    print("\n--- in-memory mode ---")
    mm = MemoryManager(user_id="demo_user")
    print(mm.stats())

    mm.add_working("current task: ingest sample document", importance=0.8)
    mm.add_episodic(
        "sample interaction: user asked about knowledge base",
        importance=0.7,
        event_type="interaction",
    )
    mm.add_semantic(
        "RAG = retrieval-augmented generation",
        importance=0.9,
        concept="rag",
        domain="ai",
    )
    mm.add_perceptual(
        "image: architecture diagram",
        importance=0.6,
        modality="image",
        source="architecture.png",
    )

    print("\n--- after adding 4 items ---")
    print(mm.stats())

    print("\n--- search 'knowledge' ---")
    for item in mm.search("knowledge")[:5]:
        print(f"[{item.memory_type}|imp={item.importance:.2f}] {item.content}")

    print("\n--- consolidate (working -> episodic) ---")
    mm.consolidate(from_type="working", to_type="episodic", importance_threshold=0.6)
    print(mm.stats())

    print("\n--- with persistence ---")
    mm2 = MemoryManager(
        user_id="demo_user",
        enable_persistence=True,
        persist_dir="./data/memory",
    )
    print(mm2.stats())
    mm2.add_semantic("persistence test: this memory should survive restart", importance=0.8)

    print("\n--- re-create MM (should restore memories) ---")
    mm3 = MemoryManager(
        user_id="demo_user",
        enable_persistence=True,
        persist_dir="./data/memory",
    )
    print(mm3.stats())


# ============================================================
#  4. RAG
# ============================================================
def demo_rag():
    banner("4. RAG system (hybrid retrieval)")
    from llm_client import build_llm
    from rag import RAGTool

    llm = build_llm()
    rag = RAGTool(llm, namespace="demo_kb")

    print("\n--- ingest sample knowledge ---")
    sample = """
# Engineering principles

## Basics
1. **Proportional**: scale resources with workload.
2. **Hot-spot amplification**: double capacity for critical paths.
3. **Time-of-day weighting**: peak hours get 1.5x capacity.
4. **Risk-aware**: pre-allocate for high-risk events.

## Common patterns
- Nightly batch: 6-10 workers per queue, 8h shifts.
- Peak traffic: 10-15 workers per route, 3h cover.
- Release day: 20-30 workers per service, 6h cover.
- Incident response: 8-12 workers, 12h dedicated.

## Risk indicators
- Coverage blind spots
- Under-monitored regions
- Historical incident hotspots
"""
    n = rag.ingest_text(sample, doc_id="deployment_principles")
    print(f"chunks ingested: {n}")

    print("\n--- retrieval ---")
    results = rag.retrieve("nightly batch sizing")
    for r in results:
        print(f"[{r['score']:.2f}] {r['content'][:150]}...")

    print("\n--- hybrid search enabled ---")
    print(f"hybrid: {rag.hybrid is not None}")

    print("\n--- RAG stats ---")
    print(rag.stats())


# ============================================================
#  5. Context engineering
# ============================================================
def demo_context():
    banner("5. Context engineering (builder + compressor)")
    from context import ContextBuilder, ContextCompressor, ContextConfig
    from core.message import Message
    from llm_client import build_llm
    from memory import MemoryManager
    from rag import RAGTool

    llm = build_llm()
    rag = RAGTool(llm, namespace="demo_ctx")
    rag.ingest_text("# Principles\nScale resources with workload.", "ctx_doc")
    memory = MemoryManager(user_id="ctx_user")
    memory.add_semantic("user is interested in scaling patterns", importance=0.8)

    builder = ContextBuilder(
        config=ContextConfig(max_tokens=2000, reserve_ratio=0.1),
        memory_manager=memory,
        rag_tool=rag,
    )

    history = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hello! How can I help?"),
        Message(role="user", content="I want to understand scaling patterns"),
    ]

    ctx = builder.build(
        user_query="How should I size things for peak hours?",
        conversation_history=history,
        system_instructions="You are a helpful assistant. Reply concisely.",
    )
    print("Context preview:")
    print(ctx[:500])

    print("\n--- compression ---")
    compressor = ContextCompressor(llm, max_tokens=200)
    long_history = [
        {"role": "system", "content": "system prompt"},
        *[
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"history message {i}: " + "content " * 50,
            }
            for i in range(10)
        ],
    ]
    print(f"original chars: {sum(len(m.get('content', '')) for m in long_history)}")
    compressed = compressor.compress(long_history)
    print(f"compressed messages: {len(compressed)}")
    for m in compressed:
        print(f"  [{m['role']}] {m.get('content', '')[:80]}")


# ============================================================
#  6. Evaluation & tracing
# ============================================================
def demo_evaluation():
    banner("6. Evaluation & observability")
    from evaluation import Evaluator
    from llm_client import build_llm
    from observability import Tracer

    llm = build_llm()

    print("\n--- tracer ---")
    tracer = Tracer(persist_path="./data/traces/demo_trace.jsonl")

    tracer.start_trace("trace-001")
    t0 = time.time()
    result = llm.invoke([{"role": "user", "content": "Hello"}])
    tracer.llm_call(
        model=llm.model,
        prompt_preview="Hello",
        response_preview=(result[:100] if result else ""),
        duration_ms=(time.time() - t0) * 1000,
    )

    tracer.tool_call(
        tool_name="calculator",
        args={"expression": "2+2"},
        result="4",
        success=True,
        duration_ms=2.0,
    )
    tracer.end_trace()
    tracer.flush()

    print("trace events:")
    for e in tracer.get_recent(5):
        print(f"  [{e.event_type}] {e.data}")

    print("\n--- evaluator ---")
    evaluator = Evaluator(llm)
    test_cases = [
        {
            "query": "What is 12 * 8?",
            "answer": "12 * 8 = 96",
            "intent": "calculation",
            "paradigm": "react",
            "tool_calls": [{"name": "calculator", "success": True}],
            "execution_time": 3.5,
        },
        {
            "query": "Generate a UUID",
            "answer": "Here is one: 3f2c5a4b-...-...",
            "intent": "tool_use",
            "paradigm": "react",
            "tool_calls": [{"name": "uuid_generator", "success": True}],
            "execution_time": 2.1,
        },
    ]

    results = evaluator.batch_evaluate(test_cases)
    for r in results:
        print(
            f"\n  query: {r.query[:40]}..."
        )
        print(
            f"  scores: relevance={r.relevance:.2f} completeness={r.completeness:.2f} "
            f"accuracy={r.accuracy:.2f}"
        )
        print(f"  overall: {r.overall:.2f} | execution_time: {r.execution_time:.1f}s")

    print("\n  summary:")
    summary = evaluator.summary(results)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.3f}")
        else:
            print(f"    {k}: {v}")


# ============================================================
#  7. Central router
# ============================================================
def demo_central():
    banner("7. Central router (full pipeline)")
    from orchestrator import get_router

    router = get_router()

    queries = [
        ("What is the square root of 144?", "calculation"),
        ("Generate 3 UUIDs for me.", "tool_use"),
        ("Explain what an agent framework is.", "knowledge_query"),
        ("Hello!", "chitchat"),
    ]

    for q, expected_intent in queries:
        print(f"\n>>> {q}")
        print(f"    expected intent: {expected_intent}")
        result = router.route(q, mode="auto")
        print(
            f"\nintent: {result.intent.intent} | mode: {result.used_paradigm} | "
            f"score: {result.reflection.score:.2f}"
        )
        print(f"elapsed: {result.elapsed_time:.2f}s")
        ans = (result.final_answer or "").strip().replace("\n\n", "\n")
        print(f"answer: {ans[:300]}{'...' if len(ans) > 300 else ''}")


# ============================================================
#  Entry point
# ============================================================
def main():
    args = sys.argv[1:]
    full = "--full" in args
    only_central = "--central" in args
    only_memory = "--memory" in args
    only_rag = "--rag" in args
    only_eval = "--eval" in args

    print("=" * 70)
    print("  PolyAgent — full demo")
    print("=" * 70)

    if only_eval:
        demo_evaluation()
        return
    if only_central:
        demo_central()
        return
    if only_memory:
        demo_memory()
        return
    if only_rag:
        demo_rag()
        return

    # Default: tools (no LLM) + optional LLM demos
    demo_tools()

    if not full:
        print("\n" + "=" * 70)
        print("  Current mode: tools only (no LLM API needed)")
        print("  For the full tour run:  python demo.py --full")
        print("  Or one section:        --central / --memory / --rag / --eval")
        print("=" * 70)
        return

    demo_memory()
    demo_rag()
    demo_context()
    demo_agents()
    demo_evaluation()
    demo_central()

    print("\n" + "=" * 70)
    print("  All demos complete.")
    print("  Start the server:  python main.py")
    print("  Browse the docs:   http://localhost:8000/docs")
    print("  SSE stream:        http://localhost:8000/api/agent/stream/chat?query=...")
    print("=" * 70)


if __name__ == "__main__":
    main()
