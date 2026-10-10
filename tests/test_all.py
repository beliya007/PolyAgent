"""
Unit tests covering all core modules.

Run:
  cd polyagent
  python -m pytest tests/test_all.py -v
  python -m pytest tests/test_all.py::TestMemory -v
  python -m pytest tests/test_all.py::TestRAG -v
  python -m pytest tests/test_all.py::TestTools -v
  python -m pytest tests/test_all.py::TestContext -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# Make `polyagent` importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def _estimate_tokens_local(text: str) -> int:
    if not text:
        return 0
    cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cn
    return int(cn / 1.5 + other / 4) + 1


# ============================================================
#  fixtures
# ============================================================
@pytest.fixture(scope="session")
def llm():
    from llm_client import build_llm
    return build_llm()


@pytest.fixture
def memory_manager():
    from memory import MemoryManager
    return MemoryManager(user_id="test_user")


@pytest.fixture
def memory_persistent():
    from memory import MemoryManager
    mm = MemoryManager(
        user_id="test_persist",
        enable_persistence=True,
        persist_dir="./data/test_memory",
    )
    yield mm
    mm.clear_all()


@pytest.fixture
def rag_tool(llm):
    from rag import RAGTool
    rag = RAGTool(llm, namespace="test_kb")
    rag.ingest_text(
        "# Deployment principles\nScale resources with workload.\n"
        "Hot-spot regions double the baseline.",
        "test_principles",
    )
    return rag


@pytest.fixture
def tool_registry():
    from tools import create_default_tool_registry
    return create_default_tool_registry()


# ============================================================
#  TestTools
# ============================================================
class TestTools:
    def test_default_registry_has_core_tools(self, tool_registry):
        names = tool_registry.list_tools()
        # The shipped defaults must include the generic helpers
        for expected in ("calculator", "text_stats", "uuid_generator", "json_format"):
            assert expected in names, f"missing default tool: {expected}"

    def test_calculator(self, tool_registry):
        result = tool_registry.execute_tool("calculator", {"expression": "(8 + 4) * 2"})
        data = json.loads(result)
        assert data["result"] == 24

    def test_text_stats(self, tool_registry):
        result = tool_registry.execute_tool(
            "text_stats", {"text": "Hello world!\nSecond line."}
        )
        data = json.loads(result)
        assert data["lines"] == 2
        assert data["words"] >= 3

    def test_uuid_generator(self, tool_registry):
        result = tool_registry.execute_tool("uuid_generator", {"count": 3})
        data = json.loads(result)
        assert data["count"] == 3
        assert len(data["uuids"]) == 3

    def test_json_format(self, tool_registry):
        result = tool_registry.execute_tool(
            "json_format", {"json_string": '{"a":1,"b":[2,3]}'}
        )
        assert '"a"' in result and '2' in result


# ============================================================
#  TestMemory
# ============================================================
class TestMemory:
    def test_memory_4_types(self, memory_manager):
        memory_manager.add_working("working test", importance=0.8)
        memory_manager.add_episodic("episodic test", importance=0.7, event_type="test")
        memory_manager.add_semantic("semantic test", importance=0.9, concept="test")
        memory_manager.add_perceptual("perceptual test", importance=0.6, modality="text")
        stats = memory_manager.stats()
        assert stats["working"]["count"] == 1
        assert stats["episodic"]["count"] == 1
        assert stats["semantic"]["count"] == 1
        assert stats["perceptual"]["count"] == 1

    def test_memory_search(self, memory_manager):
        memory_manager.add_semantic("RAG knowledge base indexing", importance=0.9)
        results = memory_manager.search("knowledge")
        assert len(results) >= 1
        assert "knowledge" in results[0].content.lower()

    def test_memory_consolidate(self, memory_manager):
        memory_manager.add_working("important working item", importance=0.8)
        memory_manager.add_episodic("ordinary episodic item", importance=0.3)
        before = memory_manager.stats()
        memory_manager.consolidate(
            from_type="working", to_type="episodic", importance_threshold=0.6
        )
        after = memory_manager.stats()
        assert after["working"]["count"] < before["working"]["count"]

    def test_memory_persistence(self, memory_persistent):
        memory_persistent.add_semantic("persistence round-trip", importance=0.9)
        memory_persistent.save_all()
        from memory import MemoryManager
        mm2 = MemoryManager(
            user_id="test_persist",
            enable_persistence=True,
            persist_dir="./data/test_memory",
        )
        assert mm2.stats()["semantic"]["count"] >= 1
        mm2.clear_all()


# ============================================================
#  TestRAG
# ============================================================
class TestRAG:
    def test_rag_ingest_and_retrieve(self, rag_tool):
        results = rag_tool.retrieve("deployment")
        assert len(results) >= 1
        assert results[0]["score"] > 0

    def test_rag_build_context(self, rag_tool):
        ctx = rag_tool.build_context("scaling")
        assert "knowledge base" in ctx.lower()

    def test_rag_stats(self, rag_tool):
        stats = rag_tool.stats()
        assert stats["namespace"] == "test_kb"
        assert stats["vector_count"] >= 1


# ============================================================
#  TestAgents
# ============================================================
class TestAgents:
    @pytest.mark.skipif(
        not Path("./polyagent/.env").exists() and not Path("./.env").exists(),
        reason="LLM_API_KEY required",
    )
    def test_simple_agent(self, llm, tool_registry):
        from agents import SimpleAgent
        agent = SimpleAgent(llm=llm, tool_registry=tool_registry)
        result = agent.run("Hello!")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(
        not Path("./polyagent/.env").exists() and not Path("./.env").exists(),
        reason="LLM_API_KEY required",
    )
    def test_intent_recognition(self, llm):
        from orchestrator import IntentRecognizer
        recognizer = IntentRecognizer(llm)
        result = recognizer.recognize("Generate 3 UUIDs for me.")
        assert result.intent in {
            "task_planning",
            "data_query",
            "calculation",
            "knowledge_query",
            "tool_use",
            "chitchat",
            "unknown",
        }
        assert 0.0 <= result.confidence <= 1.0


# ============================================================
#  TestContext
# ============================================================
class TestContext:
    def test_context_builder(self, memory_manager, rag_tool):
        from context import ContextBuilder, ContextConfig
        from core.message import Message

        builder = ContextBuilder(
            config=ContextConfig(max_tokens=1000),
            memory_manager=memory_manager,
            rag_tool=rag_tool,
        )
        ctx = builder.build("scaling deployment")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_context_compressor(self, llm):
        from context import ContextCompressor
        compressor = ContextCompressor(llm, max_tokens=200)
        long_history = [
            {"role": "system", "content": "system prompt"},
            *[
                {"role": "user", "content": f"message {i}: " + "x" * 200}
                for i in range(10)
            ],
        ]
        original = sum(_estimate_tokens_local(m["content"]) for m in long_history)
        compressed = compressor.compress(long_history)
        compressed_t = sum(_estimate_tokens_local(m.get("content", "")) for m in compressed)
        assert compressed_t < original * 0.7, (
            f"compression not effective: {compressed_t} vs {original}"
        )


# ============================================================
#  TestLLMClient
# ============================================================
class TestLLMClient:
    @pytest.mark.skipif(
        not Path("./polyagent/.env").exists() and not Path("./.env").exists(),
        reason="LLM_API_KEY required",
    )
    def test_llm_invoke(self, llm):
        result = llm.invoke([{"role": "user", "content": "What is 1+1? Reply in one word."}])
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(
        not Path("./polyagent/.env").exists() and not Path("./.env").exists(),
        reason="LLM_API_KEY required",
    )
    def test_llm_embed(self, llm):
        vecs = llm.embed(["sample text"])
        assert len(vecs) == 1
        assert len(vecs[0]) > 0

    @pytest.mark.skipif(
        not Path("./polyagent/.env").exists() and not Path("./.env").exists(),
        reason="LLM_API_KEY required",
    )
    def test_llm_stats(self, llm):
        stats = llm.stats()
        assert "model" in stats
        assert "call_count" in stats


# ============================================================
#  TestEvaluation
# ============================================================
class TestEvaluation:
    @pytest.mark.skipif(
        not Path("./polyagent/.env").exists() and not Path("./.env").exists(),
        reason="LLM_API_KEY required",
    )
    def test_evaluator(self, llm):
        from evaluation import Evaluator
        evaluator = Evaluator(llm)
        result = evaluator.evaluate(
            query="What is 12 * 8?",
            answer="12 * 8 = 96",
            intent="calculation",
            paradigm="react",
            execution_time=3.0,
        )
        assert 0.0 <= result.overall <= 1.0
        assert result.execution_time == 3.0


# ============================================================
#  TestTracing
# ============================================================
class TestTracing:
    def test_tracer_record(self):
        from observability import Tracer
        tracer = Tracer(persist_path="./data/test_traces/trace.jsonl")
        tracer.start_trace("test-001")
        tracer.llm_call(model="test", prompt_preview="test", duration_ms=100.0)
        tracer.tool_call(tool_name="test_tool", args={}, result="ok", success=True)
        tracer.end_trace()
        s = tracer.stats()
        assert s["llm_calls"] >= 1
        assert s["tool_calls"] >= 1
        tracer.flush()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
