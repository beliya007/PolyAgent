"""
混合检索 (Hybrid Search)

polyagent 第八章:
- 向量检索 + BM25 关键词检索加权融合
- 提高召回率(向量)与精确度(BM25)
- 加权公式: score = α * vector_sim + (1-α) * bm25_score
"""
from __future__ import annotations

import math
from typing import Dict, Optional

from rag.vector_store import VectorRecord, VectorStore


class HybridSearch:
    """向量 + BM25 混合检索"""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder,                    # Embedder 实例
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.alpha = vector_weight
        self.beta = bm25_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_vector: bool = True,
    ) -> list[tuple[VectorRecord, float]]:
        """
        混合检索。
        1) 如果 use_vector=True 且 Embedder 成功,做向量检索
        2) 同时做 BM25 检索
        3) 融合分数,排序
        """
        vector_scores: Dict[str, float] = {}
        bm25_scores: Dict[str, float] = {}

        # 1) 向量检索
        if use_vector and self.embedder and self.vector_store.size() > 0:
            try:
                q_vec = self.embedder.embed_one(query)
                if q_vec and any(q_vec):
                    for rec, sim in self.vector_store.search_by_vector(q_vec, top_k=top_k * 2):
                        vector_scores[rec.id] = float(sim)
            except Exception:
                pass

        # 2) BM25 检索
        for rec, score in self.vector_store.search_by_text(query, top_k=top_k * 2):
            bm25_scores[rec.id] = float(score)

        # 3) 融合
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        fused: Dict[str, float] = {}
        for rid in all_ids:
            v = vector_scores.get(rid, 0.0)
            b = bm25_scores.get(rid, 0.0)
            fused[rid] = self.alpha * v + self.beta * b

        # 排序
        sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:top_k]
        records_by_id = {r.id: r for r in self.vector_store.records.values()}
        return [(records_by_id[rid], fused[rid]) for rid in sorted_ids if rid in records_by_id]