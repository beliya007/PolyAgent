"""
情景记忆 (Episodic Memory)
- 记录具体事件
- 时间序列 + 会话级关联
- 存储:DocumentStore(结构化) + VectorStore(向量检索)
- 评分:(向量相似度 × 0.8 + 时间近因性 × 0.2) × 重要性权重
"""
from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory.base import BaseMemory, MemoryItem
from memory.persistence import JSONLPersistence
from memory.storage.base import BaseDocumentStore, BaseVectorStore


def _recency_score(ts: float, now: Optional[float] = None) -> float:
    """时间近因性:24 小时内基本保持,之后指数衰减"""
    now = now or time.time()
    age_h = max(0.0, (now - ts) / 3600)
    return max(0.1, math.exp(-0.1 * age_h / 24))


class EpisodicMemory(BaseMemory):
    memory_type = "episodic"

    def __init__(
        self,
        capacity: int = 500,
        persist: Optional[JSONLPersistence] = None,
        document_store: Optional[BaseDocumentStore] = None,
        vector_store: Optional[BaseVectorStore] = None,
        embedder=None,
    ):
        super().__init__(capacity=capacity, persist=persist, storage_backend=document_store, embedder=embedder)
        self.doc_store = document_store
        self.vector_store = vector_store
        self._has_embedder = embedder is not None

    def add(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
        event_type: str = "interaction",
        session_id: Optional[str] = None,
    ) -> str:
        meta = metadata or {}
        meta.setdefault("event_type", event_type)
        if session_id:
            meta["session_id"] = session_id
        item = MemoryItem(
            memory_type="episodic",
            content=content,
            importance=importance,
            metadata=meta,
        )
        self.items.append(item)
        self._evict()

        # 外部存储:结构化文档
        if self.doc_store is not None:
            try:
                from memory.storage.base import DocumentRecord
                self.doc_store.upsert(
                    DocumentRecord(
                        id=item.id,
                        content=item.content,
                        metadata={
                            **{k: v for k, v in item.metadata.items()
                               if isinstance(v, (str, int, float, bool))},
                            "memory_type": item.memory_type,
                            "importance": float(item.importance),
                            "timestamp": float(item.timestamp),
                        },
                    )
                )
            except Exception as e:
                print(f"[EpisodicMemory] doc_store.upsert 失败: {e}")

        # 外部存储:向量
        if self.vector_store is not None and self.embedder is not None:
            try:
                vec = self.embedder.encode_one(content)
                if vec and any(vec):
                    self.vector_store.add_vectors(
                        ids=[item.id], vectors=[vec], contents=[content],
                        metadatas=[{
                            "memory_type": item.memory_type,
                            "importance": float(item.importance),
                            "timestamp": float(item.timestamp),
                            **{k: v for k, v in item.metadata.items()
                               if isinstance(v, (str, int, float, bool))},
                        }],
                    )
            except Exception as e:
                print(f"[EpisodicMemory] vector_store.add 失败: {e}")

        self._persist_item(item)
        return item.id

    def search(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[MemoryItem]:
        # 1) 结构化预过滤
        candidates: List[MemoryItem] = self.items
        if event_type:
            candidates = [i for i in candidates if i.metadata.get("event_type") == event_type]
        if session_id:
            candidates = [i for i in candidates if i.metadata.get("session_id") == session_id]
        candidates = [i for i in candidates if i.importance >= min_importance]
        if not candidates:
            return []

        # 2) 向量检索(若有)
        vec_scores: Dict[str, float] = {}
        if self.vector_store is not None and self.embedder is not None:
            try:
                qv = self.embedder.encode_one(query)
                if qv and any(qv):
                    hits = self.vector_store.search_similar(
                        query_vector=qv,
                        top_k=max(limit * 5, 20),
                        where={"memory_type": self.memory_type},
                    )
                    for rec, score in hits:
                        vec_scores[rec.id] = float(score)
            except Exception as e:
                print(f"[EpisodicMemory] vector_search 失败: {e}")

        # 3) 综合评分
        scored: List[tuple] = []
        for item in candidates:
            vec_score = vec_scores.get(item.id, self._bm25_score(query, item.content))
            recency = _recency_score(item.timestamp)
            # (向量相似度 × 0.8 + 时间近因性 × 0.2) × 重要性权重
            base = vec_score * 0.8 + recency * 0.2
            weight = 0.8 + item.importance * 0.4
            final = base * weight
            scored.append((item, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [i for i, _ in scored[:limit]]
        for r in results:
            r.touch()
        return results

    def by_session(self, session_id: str, limit: int = 50) -> List[MemoryItem]:
        items = [i for i in self.items if i.metadata.get("session_id") == session_id]
        items.sort(key=lambda x: x.timestamp)
        return items[:limit]
