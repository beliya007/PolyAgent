"""
向量存储 (VectorStore)

轻量级实现:基于 NumPy 的内存向量数据库。
- 支持向量余弦相似度检索
- 支持持久化到 JSON
- 当 Embedding 失败时降级为 BM25 检索
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class VectorRecord:
    id: str
    content: str
    vector: List[float] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class VectorStore:
    """简易向量数据库"""

    def __init__(self, persist_path: Optional[str] = None):
        self.records: Dict[str, VectorRecord] = {}
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()

    # ---------------- 写入 ----------------

    def add(self, record: VectorRecord) -> None:
        self.records[record.id] = record

    def add_batch(self, records: List[VectorRecord]) -> None:
        for r in records:
            self.records[r.id] = r

    # ---------------- 检索 ----------------

    def search_by_vector(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[VectorRecord, float]]:
        if not self.records:
            return []
        scored = []
        for rec in self.records.values():
            if not rec.vector:
                continue
            sim = self._cosine(query_vec, rec.vector)
            scored.append((rec, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_by_text(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[VectorRecord, float]]:
        """BM25 降级检索(无向量时使用)"""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        scored = []
        for rec in self.records.values():
            c_tokens = self._tokenize(rec.content)
            if not c_tokens:
                continue
            overlap = sum(1 for q in q_tokens if q in c_tokens)
            score = overlap / len(q_tokens)
            if score >= min_score:
                scored.append((rec, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def size(self) -> int:
        return len(self.records)

    # ---------------- 持久化 ----------------

    def save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            rid: {
                "id": r.id,
                "content": r.content,
                "vector": r.vector,
                "metadata": r.metadata,
            }
            for rid, r in self.records.items()
        }
        self.persist_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"💾 已保存 {len(self.records)} 条向量到 {self.persist_path}")

    def _load(self) -> None:
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
            for rid, r in data.items():
                self.records[rid] = VectorRecord(
                    id=r["id"], content=r["content"], vector=r.get("vector", []), metadata=r.get("metadata", {})
                )
            print(f"📦 已从磁盘加载 {len(self.records)} 条向量")
        except Exception as e:
            print(f"⚠️ 向量加载失败: {e}")

    # ---------------- 内部 ----------------

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        s = text.lower()
        for ch in "，。！？、,!?;:":
            s = s.replace(ch, " ")
        return [w for w in s.split() if w]
