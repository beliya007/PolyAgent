"""
RAG 工具 (RAGTool)

polyagent 第八章 RAG 设计:
- 文档摄取(Loader → Chunker → Embedder → VectorStore)
- 检索(query → Embedder → top_k)
- 自动降级:Embedding 失败时回退 BM25
- 可选: 启用 hybrid (向量 + BM25 融合) 和 reranker
- 高级检索: MQE (Multi-Query Expansion) + HyDE (Hypothetical Document Embeddings)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import get_config
from llm_client import PolyAgentLLM
from rag.advanced_search import search_with_expansion
from rag.chunker import Chunk, MarkdownChunker, TextChunker
from rag.embedder import Embedder
from rag.hybrid_search import HybridSearch
from rag.loader import Document, DocumentLoader, convert_to_markdown
from rag.reranker import LLMReranker
from rag.vector_store import VectorRecord, VectorStore


class RAGTool:
    """RAG 检索增强工具"""

    def __init__(
        self,
        llm: PolyAgentLLM,
        knowledge_base_path: Optional[str] = None,
        namespace: str = "default",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        enable_hybrid: bool = True,
        enable_reranker: bool = False,
        enable_markdown_chunking: bool = True,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        mqe_expansions: int = 2,
    ):
        cfg = get_config().business
        self.kb_path = Path(knowledge_base_path or cfg.knowledge_base_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace

        # 持久化路径
        persist = self.kb_path / f"{namespace}.vectors.json"
        self.vector_store = VectorStore(persist_path=str(persist))
        self.loader = DocumentLoader(self.kb_path)
        self.llm = llm

        # 分块器:默认用 Markdown 感知分块,纯文本用基础分块
        if enable_markdown_chunking:
            self.chunker = MarkdownChunker(
                chunk_tokens=chunk_size or cfg.rag_chunk_size,
                overlap_tokens=chunk_overlap or cfg.rag_chunk_overlap,
            )
            self.use_markdown_chunker = True
        else:
            self.chunker = TextChunker(
                chunk_size=chunk_size or cfg.rag_chunk_size,
                chunk_overlap=chunk_overlap or cfg.rag_chunk_overlap,
            )
            self.use_markdown_chunker = False

        self.embedder = Embedder(llm)
        self.use_vector = True

        # 混合检索 + reranker
        self.hybrid = HybridSearch(self.vector_store, self.embedder) if enable_hybrid else None
        self.reranker = LLMReranker(llm, enabled=enable_reranker)

        # 高级检索
        self.enable_mqe = enable_mqe
        self.enable_hyde = enable_hyde
        self.mqe_expansions = mqe_expansions

    # ============================================================
    #  摄取
    # ============================================================
    def ingest_text(
        self,
        content: str,
        doc_id: str = "raw_doc",
        metadata: Optional[Dict] = None,
    ) -> int:
        """摄取一段文本(Markdown 优先)"""
        chunks = self.chunker.split(content, doc_id=doc_id)
        return self._index_chunks(chunks, source=doc_id, extra_meta=metadata or {})

    def ingest_file(self, path: Union[str, Path]) -> int:
        """摄取单个文件(MarkItDown 风格:任意格式 → Markdown → 分块)"""
        doc = self.loader.load_file(path)
        return self.ingest_document(doc)

    def ingest_document(self, doc: Document) -> int:
        """摄取已加载的 Document 对象"""
        chunks = self.chunker.split(doc.content, doc_id=doc.id)
        meta = {
            "source": doc.source,
            "format": doc.format,
            "modality": doc.modality,
            "filename": doc.metadata.get("filename", doc.id),
        }
        return self._index_chunks(chunks, source=doc.source, extra_meta=meta)

    def ingest_directory(self, pattern: str = "**/*.*") -> int:
        """摄取整个知识库目录"""
        docs = self.loader.load_directory(pattern=pattern)
        total = 0
        for d in docs:
            total += self.ingest_document(d)
        return total

    def _index_chunks(
        self,
        chunks: List[Chunk],
        source: str,
        extra_meta: Optional[Dict] = None,
    ) -> int:
        if not chunks:
            return 0
        texts = [c.content for c in chunks]
        vectors: List[List[float]] = []
        if self.use_vector:
            try:
                vectors = self.embedder.embed(texts)
                if not vectors or all(
                    not v or all(x == 0.0 for x in v) for v in vectors
                ):
                    self.use_vector = False
                    vectors = []
            except Exception:
                self.use_vector = False
                vectors = []

        records: List[VectorRecord] = []
        for i, c in enumerate(chunks):
            base_meta = {
                "source": source,
                "chunk_index": c.chunk_index,
                "doc_id": c.doc_id,
                **c.metadata,
                **(extra_meta or {}),
            }
            records.append(
                VectorRecord(
                    id=c.id,
                    content=c.content,
                    vector=vectors[i] if i < len(vectors) and vectors[i] else [],
                    metadata=base_meta,
                )
            )
        self.vector_store.add_batch(records)
        self.vector_store.save()
        return len(records)

    # ============================================================
    #  检索
    # ============================================================
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = False,
        enable_advanced_search: bool = False,
    ) -> List[Dict[str, Any]]:
        """检索 top_k 文档块

        enable_advanced_search=True 时启用 MQE + HyDE(8.3.5 节)
        """
        top_k = top_k or get_config().business.rag_top_k

        if enable_advanced_search:
            return self._retrieve_advanced(query, top_k=top_k, use_rerank=use_rerank)

        results: List[Tuple[VectorRecord, float]] = []

        if self.hybrid is not None:
            try:
                results = self.hybrid.search(
                    query, top_k=top_k, use_vector=self.use_vector
                )
            except Exception:
                results = self.vector_store.search_by_text(query, top_k=top_k)
        elif self.use_vector and self.vector_store.size() > 0:
            try:
                q_vec = self.embedder.embed_one(query)
                if q_vec and any(q_vec):
                    results = self.vector_store.search_by_vector(q_vec, top_k=top_k)
            except Exception:
                self.use_vector = False

        if not results:
            results = self.vector_store.search_by_text(query, top_k=top_k)

        out = [
            {
                "id": r.id,
                "content": r.content,
                "score": float(score),
                "source": r.metadata.get("source", ""),
                "metadata": dict(r.metadata),
            }
            for r, score in results
        ]

        if use_rerank and self.reranker.enabled and len(out) > 1:
            out = self.reranker.rerank(query, out, top_k=top_k)

        return out

    def _retrieve_advanced(
        self,
        query: str,
        top_k: int,
        use_rerank: bool = False,
    ) -> List[Dict[str, Any]]:
        """高级检索:MQE + HyDE"""
        def _retrieve_fn(q: str, k: int) -> List[Dict[str, Any]]:
            if self.hybrid is not None:
                results = self.hybrid.search(q, top_k=k, use_vector=self.use_vector)
            elif self.use_vector and self.vector_store.size() > 0:
                q_vec = self.embedder.embed_one(q)
                if q_vec and any(q_vec):
                    results = self.vector_store.search_by_vector(q_vec, top_k=k)
                else:
                    results = []
            else:
                results = []
            if not results:
                results = self.vector_store.search_by_text(q, top_k=k)
            return [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": float(score),
                    "source": r.metadata.get("source", ""),
                }
                for r, score in results
            ]

        merged = search_with_expansion(
            retrieve_fn=_retrieve_fn,
            query=query,
            top_k=top_k,
            enable_mqe=self.enable_mqe,
            mqe_expansions=self.mqe_expansions,
            enable_hyde=self.enable_hyde,
            candidate_pool_multiplier=4,
            llm=self.llm,
        )

        if use_rerank and self.reranker.enabled and len(merged) > 1:
            merged = self.reranker.rerank(query, merged, top_k=top_k)
        return merged

    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: bool = False,
        enable_advanced_search: bool = False,
    ) -> str:
        """把检索结果拼成上下文片段(注入 prompt)"""
        results = self.retrieve(
            query,
            top_k=top_k,
            use_rerank=use_rerank,
            enable_advanced_search=enable_advanced_search,
        )
        if not results:
            return ""
        lines = ["[知识库检索结果]"]
        for i, r in enumerate(results, 1):
            lines.append(f"({i}) [相关度 {r['score']:.2f}] {r['content']}")
            if r.get("source"):
                lines.append(f"    来源: {r['source']}")
        return "\n".join(lines)

    # ============================================================
    #  智能问答(组合 RAG + LLM)
    # ============================================================
    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        enable_advanced_search: bool = False,
        use_rerank: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        """RAG 增强问答:检索 + LLM 合成"""
        context = self.build_context(
            question,
            top_k=top_k,
            use_rerank=use_rerank,
            enable_advanced_search=enable_advanced_search,
        )
        if not context:
            return "⚠️ 知识库中未检索到相关内容。"

        sys_prompt = system_prompt or (
            "你是知识库助手。请严格依据下方检索到的资料回答用户问题。"
            "回答时:(a) 优先使用资料原文;(b) 标注引用编号 (1)(2)...;"
            "(c) 若资料不足以回答,直接说明。"
        )
        user_msg = f"问题:{question}\n\n检索资料:\n{context}"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
        try:
            return self.llm.invoke(messages) or ""
        except Exception as e:
            return f"❌ LLM 调用失败: {e}"

    # ============================================================
    #  统计 / 配置
    # ============================================================
    def stats(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "kb_path": str(self.kb_path),
            "vector_count": self.vector_store.size(),
            "use_vector": self.use_vector,
            "chunk_size": getattr(self.chunker, "chunk_size", None) or getattr(self.chunker, "chunk_tokens", None),
            "chunk_overlap": getattr(self.chunker, "chunk_overlap", None) or getattr(self.chunker, "overlap_tokens", None),
            "chunker": "MarkdownChunker" if self.use_markdown_chunker else "TextChunker",
            "hybrid_enabled": self.hybrid is not None,
            "reranker_enabled": self.reranker.enabled,
            "advanced_search": {
                "enable_mqe": self.enable_mqe,
                "enable_hyde": self.enable_hyde,
                "mqe_expansions": self.mqe_expansions,
            },
            "loader_supported": self.loader.supported_extensions(),
        }