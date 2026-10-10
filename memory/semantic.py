"""
语义记忆 (Semantic Memory)
- 抽象知识 / 概念 / 关系
- 存储:VectorStore(语义检索) + GraphStore(实体关系)
- 评分:(向量相似度 × 0.7 + 图相似度 × 0.3) × 重要性权重
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from memory.base import BaseMemory, MemoryItem
from memory.persistence import JSONLPersistence
from memory.storage.base import BaseGraphStore, BaseVectorStore, GraphEdge, GraphNode


# ============================================================
#  简易中文/英文实体抽取(无需 spaCy)
# ============================================================
_CN_STOP = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "里", "为", "与", "及", "或", "等", "之", "于",
}
_EN_STOP = {"the", "a", "an", "of", "in", "on", "at", "and", "or", "is", "are", "was", "be"}


def _is_cjk(ch: str) -> bool:
    c = ord(ch)
    return 0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF


def _extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """极简关键词抽取:CJK 2~6 字 ngram + 英文词,过滤停用词"""
    cjk_ngrams: List[str] = []
    seq: List[str] = []
    for ch in text:
        if _is_cjk(ch):
            seq.append(ch)
        else:
            if 2 <= len(seq) <= 6:
                cjk_ngrams.append("".join(seq))
            elif len(seq) > 6:
                for n in range(2, 7):
                    for i in range(0, len(seq) - n + 1):
                        cjk_ngrams.append("".join(seq[i:i + n]))
            seq = []
    if 2 <= len(seq) <= 6:
        cjk_ngrams.append("".join(seq))

    en_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", text)
    out: List[str] = []
    seen = set()
    for t in cjk_ngrams + en_tokens:
        t_low = t.lower()
        if t_low in _CN_STOP or t_low in _EN_STOP or t_low in seen or len(t_low) < 2:
            continue
        seen.add(t_low)
        out.append(t)
        if len(out) >= top_k:
            break
    return out


# ============================================================
#  语义记忆类
# ============================================================
class SemanticMemory(BaseMemory):
    memory_type = "semantic"

    def __init__(
        self,
        capacity: int = 1000,
        persist: Optional[JSONLPersistence] = None,
        vector_store: Optional[BaseVectorStore] = None,
        graph_store: Optional[BaseGraphStore] = None,
        embedder=None,
    ):
        super().__init__(capacity=capacity, persist=persist, storage_backend=vector_store, embedder=embedder)
        self.vector_store = vector_store
        self.graph_store = graph_store
        self._entity_index: Dict[str, Dict[str, Any]] = {}     # entity_id -> {label, count}

    def add(
        self,
        content: str,
        importance: float = 0.7,
        metadata: Optional[Dict] = None,
        concept: Optional[str] = None,
        domain: Optional[str] = None,
        entities: Optional[List[str]] = None,
        relations: Optional[List[Tuple[str, str, str]]] = None,    # (src, rel, dst)
    ) -> str:
        meta = metadata or {}
        if concept:
            meta["concept"] = concept
        if domain:
            meta["domain"] = domain
        item = MemoryItem(
            memory_type="semantic",
            content=content,
            importance=importance,
            metadata=meta,
        )
        self.items.append(item)
        self._evict()

        # 自动抽取实体
        extracted = entities or _extract_keywords(content, top_k=5)
        if extracted:
            meta["entities"] = extracted
            item.metadata = meta
            # 更新图谱
            if self.graph_store is not None:
                try:
                    for ent in extracted:
                        node = GraphNode(
                            id=ent,
                            label=domain or "Entity",
                            properties={"text": ent, "memory_id": item.id},
                        )
                        self.graph_store.upsert_node(node)
                        self._entity_index[ent] = {"label": node.label, "count": self._entity_index.get(ent, {}).get("count", 0) + 1}
                    # 两两相连(CO_OCCURS_WITH)
                    if len(extracted) >= 2:
                        for i in range(len(extracted)):
                            for j in range(i + 1, len(extracted)):
                                self.graph_store.upsert_edge(
                                    GraphEdge(
                                        source_id=extracted[i],
                                        target_id=extracted[j],
                                        relation="CO_OCCURS_WITH",
                                        properties={"memory_id": item.id, "weight": 1.0},
                                    )
                                )
                    # 用户显式关系
                    if relations:
                        for src, rel, dst in relations:
                            self.graph_store.upsert_edge(
                                GraphEdge(
                                    source_id=src, target_id=dst, relation=rel,
                                    properties={"memory_id": item.id},
                                )
                            )
                except Exception as e:
                    print(f"[SemanticMemory] graph_store 写入失败: {e}")

        # 向量存储
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
                            "domain": domain or "",
                            "concept": concept or "",
                            "entities": ",".join(extracted) if extracted else "",
                        }],
                    )
            except Exception as e:
                print(f"[SemanticMemory] vector_store.add 失败: {e}")

        self._persist_item(item)
        return item.id

    def search(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0,
        domain: Optional[str] = None,
        concept: Optional[str] = None,
    ) -> List[MemoryItem]:
        candidates: List[MemoryItem] = self.items
        if domain:
            candidates = [i for i in candidates if i.metadata.get("domain") == domain]
        if concept:
            candidates = [i for i in candidates if i.metadata.get("concept") == concept]
        candidates = [i for i in candidates if i.importance >= min_importance]
        if not candidates:
            return []

        # 1) 向量检索
        vec_scores: Dict[str, float] = {}
        if self.vector_store is not None and self.embedder is not None:
            try:
                qv = self.embedder.encode_one(query)
                if qv and any(qv):
                    hits = self.vector_store.search_similar(
                        query_vector=qv,
                        top_k=max(limit * 4, 20),
                        where={"memory_type": self.memory_type},
                    )
                    for rec, score in hits:
                        vec_scores[rec.id] = float(score)
            except Exception as e:
                print(f"[SemanticMemory] vector_search 失败: {e}")

        # 2) 图检索:用抽取的关键词命中相邻节点,找到关联记忆
        graph_scores: Dict[str, float] = {}
        if self.graph_store is not None:
            try:
                keywords = _extract_keywords(query, top_k=3)
                for kw in keywords:
                    neighbors = self.graph_store.neighbors(kw, limit=5)
                    for nb in neighbors:
                        mid = nb.properties.get("memory_id", "")
                        if mid:
                            graph_scores[mid] = max(graph_scores.get(mid, 0.0), 0.6)
            except Exception as e:
                print(f"[SemanticMemory] graph_search 失败: {e}")

        # 3) 混合排序
        scored: List[tuple] = []
        for item in candidates:
            vs = vec_scores.get(item.id, self._bm25_score(query, item.content))
            gs = graph_scores.get(item.id, 0.0)
            base = vs * 0.7 + gs * 0.3
            weight = 0.8 + item.importance * 0.4
            final = base * weight
            scored.append((item, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [i for i, _ in scored[:limit]]
        for r in results:
            r.touch()
        return results

    def by_concept(self, concept: str) -> List[MemoryItem]:
        return [i for i in self.items if i.metadata.get("concept") == concept]

    def related_entities(self, entity: str, limit: int = 10) -> List[str]:
        """返回与 entity 在图谱中相邻的实体 ID"""
        if self.graph_store is None:
            return []
        try:
            return [nb.id for nb in self.graph_store.neighbors(entity, limit=limit)]
        except Exception:
            return []
