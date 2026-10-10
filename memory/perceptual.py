"""
感知记忆 (Perceptual Memory)
- 多模态数据(文本 / 图片 / 音频 ...)
- 按模态分向量集合,跨模态通过 text 锚点对齐
- 存储:DocumentStore(原始路径/标签) + VectorStore(按模态)
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from memory.base import BaseMemory, MemoryItem
from memory.persistence import JSONLPersistence
from memory.storage.base import (
    BaseDocumentStore,
    BaseVectorStore,
    DocumentRecord,
)


_MODALITY_COLLECTION_SUFFIX = {
    "text": "_perceptual_text",
    "image": "_perceptual_image",
    "audio": "_perceptual_audio",
    "video": "_perceptual_video",
}


def _recency_score(ts: float, now: Optional[float] = None) -> float:
    now = now or time.time()
    age_h = max(0.0, (now - ts) / 3600)
    return max(0.1, math.exp(-0.1 * age_h / 24))


class PerceptualMemory(BaseMemory):
    memory_type = "perceptual"

    def __init__(
        self,
        capacity: int = 300,
        persist: Optional[JSONLPersistence] = None,
        document_store: Optional[BaseDocumentStore] = None,
        vector_stores: Optional[Dict[str, BaseVectorStore]] = None,
        embedder=None,
    ):
        super().__init__(capacity=capacity, persist=persist, storage_backend=document_store, embedder=embedder)
        self.doc_store = document_store
        # 缺省:所有模态共用一个 vector_store(用 modality 字段区分)
        self.vector_stores: Dict[str, BaseVectorStore] = vector_stores or {}

    def _pick_store(self, modality: str) -> Optional[BaseVectorStore]:
        """按模态挑一个向量存储(未配置则共用默认)"""
        if not self.vector_stores:
            return None
        return self.vector_stores.get(modality) or self.vector_stores.get("text")

    def add(
        self,
        content: str,
        importance: float = 0.6,
        metadata: Optional[Dict] = None,
        modality: str = "text",
        source: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> str:
        meta = metadata or {}
        meta["modality"] = modality
        if source:
            meta["source"] = source
        item = MemoryItem(
            memory_type="perceptual",
            content=content,
            importance=importance,
            metadata=meta,
        )
        self.items.append(item)
        self._evict()

        # 文档存储:保留文件路径/标签
        if self.doc_store is not None:
            try:
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
                print(f"[PerceptualMemory] doc_store 失败: {e}")

        # 向量存储
        store = self._pick_store(modality)
        if store is not None:
            try:
                if vector is None and self.embedder is not None and modality == "text":
                    vector = self.embedder.encode_one(content)
                if vector and any(vector):
                    store.add_vectors(
                        ids=[item.id], vectors=[vector], contents=[content],
                        metadatas=[{
                            "memory_type": item.memory_type,
                            "importance": float(item.importance),
                            "timestamp": float(item.timestamp),
                            "modality": modality,
                            "source": source or "",
                        }],
                    )
            except Exception as e:
                print(f"[PerceptualMemory] vector_store.add 失败: {e}")

        self._persist_item(item)
        return item.id

    def search(
        self,
        query: str,
        limit: int = 5,
        modality: Optional[str] = None,
        min_importance: float = 0.0,
        query_vector: Optional[List[float]] = None,
    ) -> List[MemoryItem]:
        candidates: List[MemoryItem] = self.items
        if modality:
            candidates = [i for i in candidates if i.metadata.get("modality") == modality]
        candidates = [i for i in candidates if i.importance >= min_importance]
        if not candidates:
            return []

        # 1) 向量检索
        vec_scores: Dict[str, float] = {}
        target_modality = modality or "text"
        store = self._pick_store(target_modality)
        if store is not None:
            try:
                if query_vector is None and self.embedder is not None:
                    query_vector = self.embedder.encode_one(query)
                if query_vector and any(query_vector):
                    where = {"memory_type": self.memory_type}
                    if modality:
                        where["modality"] = modality
                    hits = store.search_similar(
                        query_vector=query_vector,
                        top_k=max(limit * 5, 20),
                        where=where,
                    )
                    for rec, score in hits:
                        vec_scores[rec.id] = float(score)
            except Exception as e:
                print(f"[PerceptualMemory] vector_search 失败: {e}")

        # 2) 融合评分
        scored: List[tuple] = []
        for item in candidates:
            vs = vec_scores.get(item.id, self._bm25_score(query, item.content))
            recency = _recency_score(item.timestamp)
            base = vs * 0.8 + recency * 0.2
            weight = 0.8 + item.importance * 0.4
            final = base * weight
            scored.append((item, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [i for i, _ in scored[:limit]]
        for r in results:
            r.touch()
        return results
