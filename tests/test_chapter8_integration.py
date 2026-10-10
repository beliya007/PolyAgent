"""
Chapter 8 integration test — multi-modal document loading + Markdown chunking
+ multi-backend storage. Mirrors the polyagent tutorial.

Run:
  cd polyagent
  python tests/test_chapter8_integration.py
"""
from __future__ import annotations

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Path stability
HERE = Path(__file__).resolve().parent
AGENT_ROOT = HERE.parent
sys.path.insert(0, str(AGENT_ROOT))


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main() -> None:
    section("1. Converter registry — auto-detect available backends")
    from rag.loader import (
        ConverterRegistry, TextConverter, MarkdownConverter, CodeConverter,
        CSVConverter, JSONConverter, HTMLConverter, XMLConverter,
        DocxConverter, PptxConverter, XlsxConverter, PdfConverter,
        ImageConverter, AudioConverter,
    )
    reg = ConverterRegistry()
    supported = reg.list_supported()
    print(f"+ {len(reg._converters)} converters registered")
    print(f"+ {len(supported)} extensions supported: {supported[:8]}...")
    must_have = {"md", "txt", "csv", "json", "html", "docx", "pptx",
                 "xlsx", "pdf", "png", "jpg", "mp3", "py", "yaml"}
    missing = must_have - set(supported)
    assert not missing, f"missing key formats: {missing}"
    print(f"+ all key formats covered (including {sorted(must_have)})")

    section("2. convert_to_markdown — unified entry point")
    from rag.loader import convert_to_markdown

    # MD
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("# Test\n\n**bold** and `code`.")
        md_path = f.name
    md = convert_to_markdown(md_path)
    assert "# Test" in md and "**bold**" in md
    print(f"+ MD -> Markdown: {md.strip()[:60]}")
    os.unlink(md_path)

    # CSV -> Markdown table
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("name,age\nAlice,30\nBob,25\n")
        csv_path = f.name
    csv_md = convert_to_markdown(csv_path)
    assert "| name |" in csv_md and "| Alice |" in csv_md
    print(f"+ CSV -> Markdown table:")
    for line in csv_md.splitlines():
        print(f"    {line}")
    os.unlink(csv_path)

    # JSON -> code block
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write('{"key": "value"}')
        json_path = f.name
    json_md = convert_to_markdown(json_path)
    assert "```json" in json_md
    print(f"+ JSON -> fenced code block (first line: {json_md.splitlines()[0]})")
    os.unlink(json_path)

    # HTML -> heading + paragraph
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write("<html><body><h1>Title</h1><p>Hello <strong>world</strong></p></body></html>")
        html_path = f.name
    html_md = convert_to_markdown(html_path)
    assert "# Title" in html_md and "**world**" in html_md
    print(f"+ HTML -> Markdown heading/paragraph")
    os.unlink(html_path)

    # PY -> code block
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write('def hello():\n    print("hi")\n')
        py_path = f.name
    py_md = convert_to_markdown(py_path)
    assert "```python" in py_md
    print(f"+ PY -> fenced code block")
    os.unlink(py_path)

    section("3. DocumentLoader — load + cache + modality inference")
    from rag.loader import DocumentLoader, Document

    kb_dir = tempfile.mkdtemp(prefix="kb_")
    md_file = Path(kb_dir) / "guide.md"
    md_file.write_text(
        "# Deployment\n\n## Nightly strategy\n\nBackground jobs as the main pattern.\n\n## Daytime strategy\n\nDedicated nodes as the main pattern.",
        encoding="utf-8",
    )
    loader = DocumentLoader(kb_dir)
    doc = loader.load_file(md_file)
    assert isinstance(doc, Document)
    assert doc.format == "md"
    assert doc.modality == "text"
    print(f"+ load_file: id={doc.id} modality={doc.modality} len={len(doc.content)}")

    cases = {
        "a.pdf": "document", "a.docx": "document",
        "a.xlsx": "spreadsheet", "a.pptx": "presentation",
        "a.png": "image", "a.mp3": "audio",
        "a.py": "code", "a.md": "text",
    }
    for ext, expected_mod in cases.items():
        actual = DocumentLoader.infer_modality(ext)
        assert actual == expected_mod, f"{ext}: expected {expected_mod}, got {actual}"
    print(f"+ infer_modality covers 8 common formats")

    doc_again = loader.load_file(md_file)
    assert doc_again.content == doc.content
    cache_files = list(loader.cache_dir.glob("*.json"))
    print(f"+ cache hit: {len(cache_files)} cache files")
    loader.clear_cache()

    section("4. MarkdownChunker — heading-aware + token-controlled")
    from rag.chunker import MarkdownChunker, TextChunker, _approx_token_len

    long_md = """# Engineering handbook

## Chapter 1: Capacity planning

Capacity planning should follow a baseline ratio determined per workload.

### 1.1 Steady-state load

Steady-state load should keep headroom around thirty percent.

### 1.2 Burst handling

Burst handling may relax the ratio when traffic spikes briefly.

## Chapter 2: Time-of-day strategy

### 2.1 Business hours

Business hours rely on dedicated nodes, supported by elastic workers.

### 2.2 Off hours

Off hours rely on background tasks, with on-demand nodes in critical queues.

#### 2.2.1 Worker pool

Worker pool must cover all critical queues.

#### 2.2.2 Incident response

Incident response time should be kept under five minutes.

## Chapter 3: Special scenarios

### 3.1 Release day

Release day requires thirty percent more capacity than normal traffic.

### 3.2 Severe weather

Severe weather triggers the contingency failover plan.
"""
    mc = MarkdownChunker(chunk_tokens=80, overlap_tokens=20)
    chunks = mc.split(long_md, doc_id="engineering")
    assert len(chunks) >= 2, "should produce at least 2 chunks"
    for c in chunks:
        assert c.metadata.get("heading_path"), f"chunk {c.chunk_index} missing heading_path"
        assert c.metadata.get("token_len", 0) <= 80 + 30
    print(f"+ MarkdownChunker: {len(chunks)} chunks, every chunk has heading_path")
    for c in chunks:
        print(f"    [{c.chunk_index:2d}] tokens={c.metadata['token_len']:3d} heading={c.metadata['heading_path']}")

    cjk_text = "扩容方案" * 10  # 40 CJK chars
    eng_text = "capacity planning strategy"  # 3 words
    assert _approx_token_len(cjk_text) == 40
    assert _approx_token_len(eng_text) == 3
    print(f"+ CJK/EN token estimate: 40 chars -> {_approx_token_len(cjk_text)}, 3 words -> {_approx_token_len(eng_text)}")

    tc = TextChunker(chunk_size=200, chunk_overlap=30)
    tc_chunks = tc.split(long_md, doc_id="text")
    assert len(tc_chunks) >= 1
    print(f"+ TextChunker (basic char split): {len(tc_chunks)} chunks")

    section("5. RAGTool — 摄取 + 检索 + 高级检索")
    from rag.rag_tool import RAGTool

    class MockLLM:
        def embed(self, texts):
            # 返回固定维度零向量,模拟 embedding 不可用 → 触发 BM25 降级
            return [[0.0] * 16 for _ in texts]
        def invoke(self, messages=None, **kw):
            return "Mock LLM response."
        def chat(self, messages, **kw):
            return "Mock LLM response."
    llm = MockLLM()
    rag = RAGTool(
        llm=llm,
        knowledge_base_path=kb_dir,
        namespace="test",
        chunk_size=80,
        chunk_overlap=20,
        enable_markdown_chunking=True,
        enable_hybrid=True,
        enable_mqe=False,
        enable_hyde=False,
    )
    n = rag.ingest_file(md_file)
    assert n >= 1
    print(f"✓ 摄取: {n} chunks (MarkdownChunker)")
    # stats
    s = rag.stats()
    assert s["chunker"] == "MarkdownChunker"
    assert s["vector_count"] == n
    assert s["hybrid_enabled"]
    print(f"✓ stats: {s['chunker']}, vectors={s['vector_count']}, hybrid={s['hybrid_enabled']}")

    # 检索(BM25 降级,因为 embed 返回 0 向量)
    results = rag.retrieve("nightly", top_k=2)
    assert len(results) >= 1
    print(f"+ retrieve('nightly'): {len(results)} hits")
    for r in results:
        print(f"    score={r['score']:.3f} content={r['content'][:50].strip()}...")

    # build_context
    ctx = rag.build_context("nightly strategy")
    assert "[知识库检索结果]" in ctx
    print(f"+ build_context injected context (length {len(ctx)})")

    # stats 包含 advanced search 配置
    assert "advanced_search" in s
    assert s["advanced_search"]["enable_mqe"] is False
    print(f"✓ advanced_search 配置: {s['advanced_search']}")

    shutil.rmtree(kb_dir)

    section("6. 存储后端 — memory / sqlite / qdrant(降级) / neo4j(降级)")
    from memory.storage.memory_store import (
        MemoryVectorStore, MemoryGraphStore, MemoryDocumentStore,
    )
    from memory.storage.sqlite_store import SQLiteDocumentStore
    from memory.storage import QdrantVectorStore, Neo4jGraphStore
    from memory.storage.base import (
        VectorRecord, GraphNode, GraphEdge, DocumentRecord,
    )

    # 内存向量
    mv = MemoryVectorStore()
    mv.add_vectors(["a", "b"], [[1, 0, 0], [0, 1, 0]], contents=["x", "y"])
    hits = mv.search_similar([1, 0, 0], top_k=2)
    assert len(hits) == 2 and hits[0][0].id == "a"
    print(f"✓ MemoryVectorStore: cosine 检索 {len(hits)} hits, top={hits[0][0].id}")

    # 内存图
    mg = MemoryGraphStore()
    mg.upsert_node(GraphNode(id="Alice", label="Person"))
    mg.upsert_node(GraphNode(id="Station", label="Org"))
    mg.upsert_edge(GraphEdge("Alice", "Station", "WORKS_AT"))
    nbrs = mg.neighbors("Alice")
    assert len(nbrs) == 1
    print(f"✓ MemoryGraphStore: 邻居查询 {len(nbrs)} ({nbrs[0].id})")

    # 内存文档
    md_store = MemoryDocumentStore()
    md_store.upsert(DocumentRecord(id="d1", content="c1", metadata={"k": "v"}))
    md_store.upsert(DocumentRecord(id="d2", content="c2", metadata={"k": "w"}))
    assert md_store.count() == 2
    q = md_store.query({"k": "v"})
    assert len(q) == 1
    print(f"✓ MemoryDocumentStore: count={md_store.count()}, query by meta")

    # SQLite
    sq = SQLiteDocumentStore(":memory:")
    sq.upsert(DocumentRecord(id="d1", content="sqlite doc", metadata={"user": "u1"}))
    assert sq.count() == 1
    sq.upsert(DocumentRecord(id="d1", content="sqlite doc v2", metadata={"user": "u1"}))  # 覆盖
    assert sq.count() == 1
    rec = sq.get("d1")
    assert rec.content == "sqlite doc v2"
    print(f"✓ SQLiteDocumentStore: upsert 覆盖, get 返回最新")

    # Qdrant 降级
    qv = QdrantVectorStore()
    assert qv.is_remote is False  # 未配置 URL,降级
    print(f"✓ QdrantVectorStore: 缺失 URL 自动降级为内存")

    # Neo4j 降级
    nv = Neo4jGraphStore()
    assert nv.is_remote is False
    print(f"✓ Neo4jGraphStore: 缺失 URI 自动降级为内存")

    section("7. MemoryManager — 4 种记忆 + 3 类存储")
    from memory import MemoryManager, MemoryConfig

    cfg = MemoryConfig(embedding_prefer="tfidf")
    mm = MemoryManager(user_id="chapter8", config=cfg)
    assert set(mm.memory_types.keys()) == {"working", "episodic", "semantic", "perceptual"}
    print(f"✓ 4 种记忆类型已激活: {list(mm.memory_types.keys())}")

    mm.add_working("用户在咖啡馆", importance=0.6, session_id="S1")
    mm.add_episodic("用户完成了需求评审:5 个功能点", importance=0.8, event_type="review")
    mm.add_semantic(
        "用户偏好简短回复",
        importance=0.9, domain="preference", entities=["咖啡馆"],
    )
    mm.add_perceptual("会议截图", importance=0.7, modality="image", source="cam-01")

    stats = mm.stats()
    # 各记忆类型都有 backend = memory
    for t in ["working", "episodic", "semantic", "perceptual"]:
        assert stats[t]["backend"] == "memory"
    # 三类存储都有
    assert stats["backends"]["vector"]["backend"] == "memory"
    assert stats["backends"]["graph"]["backend"] == "memory"
    assert stats["backends"]["document"]["backend"] == "memory"
    print(f"✓ 所有后端 = memory, vector/graph/document 三类齐备")

    # 检索
    res = mm.search("咖啡", limit=5)
    assert len(res) >= 2
    print(f"✓ search('咖啡') 返回 {len(res)} 条 (跨类型)")

    # 整合
    promoted = mm.consolidate(from_type="working", to_type="episodic", importance_threshold=0.5)
    print(f"✓ consolidate(working→episodic, 阈值=0.5): 提升 {promoted} 条")

    # 遗忘
    forgotten = mm.forget(strategy="importance_based", threshold=0.5)
    print(f"✓ forget(importance_based, 阈值=0.5): 遗忘 {forgotten} 条")

    section("✅ Chapter 8 集成测试全部通过")


if __name__ == "__main__":
    main()