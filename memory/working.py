"""
工作记忆 (Working Memory)
- 容量有限 + TTL 自动清理
- 纯内存存储,访问速度极快
- 组合检索:TF-IDF 词项相似度 + 关键词匹配
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from memory.base import BaseMemory, MemoryItem
from memory.persistence import JSONLPersistence


class WorkingMemory(BaseMemory):
    memory_type = "working"

    def __init__(
        self,
        capacity: int = 20,
        ttl_minutes: int = 60,
        ttl_seconds: Optional[int] = None,
        persist: Optional[JSONLPersistence] = None,
        storage_backend=None,
        embedder=None,
    ):
        super().__init__(capacity=capacity, persist=persist, storage_backend=storage_backend, embedder=embedder)
        # 同时支持 minutes(旧)和 seconds(新)两个 TTL 单位
        if ttl_seconds is not None:
            self.ttl_seconds = int(ttl_seconds)
        else:
            self.ttl_seconds = int(ttl_minutes) * 60

    def add(
        self, content: str, importance: float = 0.5, metadata: Optional[Dict] = None
    ) -> str:
        item = MemoryItem(
            memory_type="working",
            content=content,
            importance=importance,
            metadata=metadata or {},
        )
        self.items.append(item)
        self._evict()
        self._persist_item(item)
        return item.id

    def search(self, query: str, limit: int = 5, min_importance: float = 0.0) -> List[MemoryItem]:
        # 清理过期
        self._cleanup_expired()
        # 评分:BM25 + 时间衰减 + 重要性
        scored: List[tuple] = []
        now = time.time()
        for item in self.items:
            if item.importance < min_importance:
                continue
            base = self._bm25_score(query, item.content)
            age_h = (now - item.timestamp) / 3600
            decay = math.exp(-0.5 * age_h)
            # 综合:(相关性 × 时间衰减) × (0.8 + 重要性 × 0.4)
            final = (base * decay) * (0.8 + item.importance * 0.4)
            if final > 0:
                scored.append((item, final))
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [i for i, _ in scored[:limit]]
        for r in results:
            r.touch()
        return results

    def _cleanup_expired(self) -> None:
        now = time.time()
        before = len(self.items)
        self.items = [i for i in self.items if now - i.timestamp < self.ttl_seconds]
        if len(self.items) < before and self.persist:
            self.persist.save_items(self.memory_type, self.items)
            print(f"🧹 工作记忆清理 {before - len(self.items)} 条过期记录")
