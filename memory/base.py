"""
记忆抽象基类

polyagent 第八章:
- 四种记忆类型(working / episodic / semantic / perceptual)
- 统一接口:add / search / clear / stats
- 支持时间衰减、BM25 检索
- 可选:挂载外部存储后端(Qdrant / SQLite / Neo4j / Memory*)
"""
from __future__ import annotations

import math
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    memory_type: str = "working"
    content: str = ""
    importance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_type": self.memory_type,
            "content": self.content,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryItem":
        return cls(
            id=d.get("id") or str(uuid.uuid4())[:8],
            memory_type=d.get("memory_type", "working"),
            content=d.get("content", ""),
            importance=float(d.get("importance", 0.5)),
            timestamp=float(d.get("timestamp", time.time())),
            metadata=d.get("metadata", {}) or {},
            access_count=int(d.get("access_count", 0)),
        )

    def decay(self) -> None:
        """时间衰减(重要性随时间下降)"""
        age_min = (time.time() - self.timestamp) / 60
        self.importance *= math.exp(-0.01 * age_min)

    def age_min(self) -> float:
        return (time.time() - self.timestamp) / 60

    def touch(self) -> None:
        """命中时调用,刷新访问计数"""
        self.access_count += 1


class BaseMemory(ABC):
    memory_type: str = "base"

    def __init__(
        self,
        capacity: int = 100,
        persist: Optional["JSONLPersistence"] = None,
        storage_backend: Optional[Any] = None,
        embedder: Optional[Any] = None,
    ):
        self.capacity = capacity
        self.items: List[MemoryItem] = []
        self.persist = persist
        # 可选外部存储后端(由子类决定如何使用)
        self.storage = storage_backend
        self.embedder = embedder
        # 启动时从 JSONL 加载(若启用)
        if persist is not None:
            self.items = persist.load_items(self.memory_type)
            if self.items:
                print(f"📦 [{self.memory_type}] 从 JSONL 恢复 {len(self.items)} 条记忆")

    @abstractmethod
    def add(self, content: str, importance: float = 0.5, metadata: Optional[Dict] = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, limit: int = 5, min_importance: float = 0.0) -> List[MemoryItem]:
        raise NotImplementedError

    def clear(self) -> int:
        n = len(self.items)
        self.items.clear()
        # 同时清理外部存储
        if self.storage is not None and hasattr(self.storage, "delete"):
            try:
                self.storage.delete([it.id for it in self.items])
            except Exception:
                pass
        if self.persist is not None:
            self.persist.save_items(self.memory_type, [])
        return n

    def stats(self) -> Dict[str, Any]:
        return {
            "type": self.memory_type,
            "count": len(self.items),
            "capacity": self.capacity,
            "avg_importance": (
                sum(i.importance for i in self.items) / len(self.items) if self.items else 0
            ),
            "backend": getattr(self.storage, "backend_name", None) if self.storage else "memory",
        }

    def _evict(self) -> None:
        """容量超出时淘汰最不重要的"""
        if len(self.items) < self.capacity:
            return
        self.items.sort(key=lambda x: x.importance, reverse=True)
        self.items = self.items[: self.capacity]

    def _persist_item(self, item: MemoryItem) -> None:
        if self.persist is not None:
            self.persist.append_item(item)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        s = text.lower()
        for ch in "，。！？、,!?;:":
            s = s.replace(ch, " ")
        return [w for w in s.split() if w]

    def _bm25_score(self, query: str, content: str) -> float:
        """简易 BM25 检索分数(单文档 vs 单查询)"""
        q_tokens = self._tokenize(query)
        c_tokens = self._tokenize(content)
        if not q_tokens or not c_tokens:
            return 0.0
        c_set = set(c_tokens)
        overlap = sum(1 for q in q_tokens if q in c_set)
        return overlap / len(q_tokens)


# 前向声明(避免循环导入)
from memory.persistence import JSONLPersistence  # noqa: E402
