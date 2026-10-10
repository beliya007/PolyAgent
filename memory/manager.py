"""
记忆管理器 (MemoryManager)

hello-agents 第八章组合模式 + 多后端存储:
- 统一管理 4 种记忆 + 三类存储(向量/图/文档)
- 提供 search_all / add / consolidate / forget 等高级操作
- 支持选择性启用记忆类型
- 支持 JSONL 持久化(向后兼容)
- 根据 MemoryConfig 自动构建:内存 / SQLite / Qdrant / Neo4j 后端
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from memory.base import MemoryItem
from memory.config import MemoryConfig, get_memory_config
from memory.embedding import BaseEmbedding, get_text_embedder
from memory.episodic import EpisodicMemory
from memory.persistence import JSONLPersistence
from memory.perceptual import PerceptualMemory
from memory.semantic import SemanticMemory
from memory.storage import (
    MemoryDocumentStore,
    MemoryGraphStore,
    MemoryVectorStore,
    Neo4jGraphStore,
    QdrantVectorStore,
    SQLiteDocumentStore,
)
from memory.working import WorkingMemory


class MemoryManager:
    """组合模式:管理多种记忆 + 多种存储后端"""

    def __init__(
        self,
        user_id: str = "default",
        config: Optional[MemoryConfig] = None,
        enabled_types: Optional[List[str]] = None,
        working_capacity: Optional[int] = None,
        working_ttl_min: Optional[int] = None,
        working_ttl_seconds: Optional[int] = None,
        persist_dir: Optional[str] = None,
        enable_persistence: Optional[bool] = None,
        embedder: Optional[BaseEmbedding] = None,
    ):
        self.user_id = user_id
        cfg = config or get_memory_config()
        self.config = cfg
        enabled = set(enabled_types or cfg.enabled_types)

        # -------- 持久化(可选,兼容旧版 JSONL) --------
        self.persist: Optional[JSONLPersistence] = None
        enable_jsonl = enable_persistence if enable_persistence is not None else cfg.enable_jsonl_persistence
        if enable_jsonl:
            self.persist = JSONLPersistence(persist_dir or cfg.persist_dir, user_id=user_id)

        # -------- 嵌入器(全局单例,可被覆盖) --------
        self.embedder = embedder or get_text_embedder(prefer=cfg.embedding.prefer)

        # -------- 构建存储后端 --------
        self.vector_store = self._build_vector_store(cfg)
        self.graph_store = self._build_graph_store(cfg)
        self.doc_store = self._build_doc_store(cfg)

        # 感知记忆按模态独立向量集合
        self.perceptual_vector_stores = self._build_perceptual_vector_stores(cfg)

        # -------- 构建记忆类型 --------
        self.memory_types: Dict[str, Any] = {}
        if "working" in enabled:
            self.memory_types["working"] = WorkingMemory(
                capacity=working_capacity or cfg.working_memory_capacity,
                ttl_minutes=working_ttl_min or cfg.working_memory_ttl_min,
                ttl_seconds=working_ttl_seconds or cfg.working_memory_ttl_seconds,
                persist=self.persist,
            )
        if "episodic" in enabled:
            self.memory_types["episodic"] = EpisodicMemory(
                capacity=cfg.episodic_capacity,
                persist=self.persist,
                document_store=self.doc_store,
                vector_store=self.vector_store,
                embedder=self.embedder,
            )
        if "semantic" in enabled:
            self.memory_types["semantic"] = SemanticMemory(
                capacity=cfg.semantic_capacity,
                persist=self.persist,
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                embedder=self.embedder,
            )
        if "perceptual" in enabled:
            self.memory_types["perceptual"] = PerceptualMemory(
                capacity=cfg.perceptual_capacity,
                persist=self.persist,
                document_store=self.doc_store,
                vector_stores=self.perceptual_vector_stores,
                embedder=self.embedder,
            )

        print(
            f"🧠 MemoryManager 初始化完成 "
            f"(用户:{user_id}, 类型:{list(self.memory_types)}, "
            f"后端:vector={self.vector_store.backend_name}, "
            f"graph={self.graph_store.backend_name}, "
            f"doc={self.doc_store.backend_name}, "
            f"embedder={self.embedder.name})"
        )

    # ============================================================
    #  后端构建
    # ============================================================
    def _build_vector_store(self, cfg: MemoryConfig):
        backend = cfg.vector.backend.lower()
        if backend == "qdrant" and QdrantVectorStore is not None:
            try:
                return QdrantVectorStore(
                    url=cfg.vector.qdrant_url,
                    api_key=cfg.vector.qdrant_api_key,
                    collection=cfg.vector.collection,
                    vector_size=cfg.vector.vector_size,
                    distance=cfg.vector.distance,
                    timeout=cfg.vector.timeout,
                )
            except Exception as e:
                print(f"[MemoryManager] QdrantVectorStore 构建失败,回落内存: {e}")
        return MemoryVectorStore()

    def _build_graph_store(self, cfg: MemoryConfig):
        backend = cfg.graph.backend.lower()
        if backend == "neo4j" and Neo4jGraphStore is not None:
            try:
                return Neo4jGraphStore(
                    uri=cfg.graph.neo4j_uri,
                    username=cfg.graph.neo4j_username,
                    password=cfg.graph.neo4j_password,
                    database=cfg.graph.database,
                )
            except Exception as e:
                print(f"[MemoryManager] Neo4jGraphStore 构建失败,回落内存: {e}")
        return MemoryGraphStore()

    def _build_doc_store(self, cfg: MemoryConfig):
        backend = cfg.document.backend.lower()
        if backend == "sqlite":
            try:
                return SQLiteDocumentStore(cfg.document.sqlite_path)
            except Exception as e:
                print(f"[MemoryManager] SQLiteDocumentStore 构建失败,回落内存: {e}")
        return MemoryDocumentStore()

    def _build_perceptual_vector_stores(self, cfg: MemoryConfig) -> Dict[str, Any]:
        """感知记忆按模态分集合。缺省:不创建(共用默认向量库)"""
        # 为简化:共用 self.vector_store,通过 metadata.modality 区分
        # 若想分集合,用户可在外部配置后再 set
        return {}

    def attach_perceptual_vector_store(self, modality: str, store) -> None:
        """业务侧为某模态单独挂载一个向量存储"""
        if "perceptual" in self.memory_types:
            self.memory_types["perceptual"].vector_stores[modality] = store

    # ============================================================
    #  写入
    # ============================================================
    def add(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[str]:
        mem = self.memory_types.get(memory_type)
        if not mem:
            return None
        meta = metadata or {}
        meta.setdefault("user_id", self.user_id)
        return mem.add(content=content, importance=importance, metadata=meta, **kwargs)

    def add_working(
        self,
        content: str,
        importance: float = 0.5,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        meta = kwargs.pop("metadata", {}) or {}
        if session_id:
            meta["session_id"] = session_id
        return self.add(content, "working", importance, metadata=meta if meta else None, **kwargs)

    def add_episodic(
        self,
        content: str,
        importance: float = 0.6,
        event_type: str = "interaction",
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[str]:
        meta = metadata or {}
        if session_id:
            meta["session_id"] = session_id
        return self.add(
            content, "episodic", importance, metadata=meta if meta else None, event_type=event_type
        )

    def add_semantic(
        self,
        content: str,
        importance: float = 0.7,
        concept: Optional[str] = None,
        domain: Optional[str] = None,
        entities: Optional[List[str]] = None,
        relations: Optional[List[tuple]] = None,
    ) -> Optional[str]:
        return self.add(
            content, "semantic", importance, concept=concept, domain=domain,
            entities=entities, relations=relations,
        )

    def add_perceptual(
        self,
        content: str,
        importance: float = 0.6,
        modality: str = "text",
        source: Optional[str] = None,
        vector: Optional[List[float]] = None,
    ) -> Optional[str]:
        return self.add(
            content, "perceptual", importance,
            modality=modality, source=source, vector=vector,
        )

    # ============================================================
    #  检索
    # ============================================================
    def search(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        if memory_type:
            mem = self.memory_types.get(memory_type)
            return mem.search(query, limit=limit, min_importance=min_importance) if mem else []
        # 跨类型搜索,合并按分数排序(无法统一分数时用重要性 × 类型权重近似)
        collected: List[tuple] = []
        for t, mem in self.memory_types.items():
            for item in mem.search(query, limit=limit, min_importance=min_importance):
                weight = {"semantic": 1.2, "episodic": 1.1, "working": 0.9, "perceptual": 1.0}.get(t, 1.0)
                collected.append((item, item.importance * weight))
        collected.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in collected[:limit]]

    def search_all(self, query: str, limit_per_type: int = 3) -> Dict[str, List[MemoryItem]]:
        return {
            t: mem.search(query, limit=limit_per_type)
            for t, mem in self.memory_types.items()
        }

    # ============================================================
    #  整合 / 遗忘
    # ============================================================
    def consolidate(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.6,
    ) -> int:
        """把重要的工作记忆提升为长期记忆"""
        from_mem = self.memory_types.get(from_type)
        to_mem = self.memory_types.get(to_type)
        if not from_mem or not to_mem:
            return 0
        promoted = 0
        for item in list(from_mem.items):
            if item.importance >= importance_threshold:
                to_mem.items.append(item)
                # 同步到目标类型的外部存储(尽力而为)
                try:
                    if hasattr(to_mem, "_persist_item"):
                        to_mem._persist_item(item)
                except Exception:
                    pass
                from_mem.items.remove(item)
                promoted += 1
        if promoted:
            if self.persist:
                from_mem.persist.save_items(from_type, from_mem.items)
            print(f"🔄 整合 {promoted} 条 {from_type}→{to_type}")
        return promoted

    def forget(self, strategy: str = "importance_based", threshold: float = 0.2, max_age_days: int = 30) -> int:
        n = 0
        import time
        now = time.time()
        max_age_sec = max_age_days * 86400
        for t, mem in self.memory_types.items():
            before = len(mem.items)
            kept = []
            for it in mem.items:
                # 三种策略任选其一
                if strategy == "importance_based" and it.importance < threshold:
                    continue
                if strategy == "time_based" and (now - it.timestamp) > max_age_sec:
                    continue
                if strategy == "capacity_based":
                    # 容量策略在 _evict 已处理,这里跳过过滤
                    pass
                kept.append(it)
            mem.items = kept
            diff = before - len(mem.items)
            if diff > 0 and self.persist:
                self.persist.save_items(t, mem.items)
            n += diff
        if n:
            print(f"🧹 遗忘 {n} 条记忆 (策略:{strategy})")
        return n

    def clear_all(self) -> Dict[str, int]:
        result = {t: mem.clear() for t, mem in self.memory_types.items()}
        # 同时清空外部存储
        for store in (self.vector_store, self.graph_store, self.doc_store):
            try:
                ids = list(store.records.keys()) if hasattr(store, "records") else []
                if ids:
                    store.delete(ids)
            except Exception:
                pass
        if self.persist:
            self.persist.clear()
        return result

    def save_all(self) -> Dict[str, int]:
        if not self.persist:
            return {}
        out: Dict[str, int] = {}
        for t, mem in self.memory_types.items():
            self.persist.save_items(t, mem.items)
            out[t] = len(mem.items)
        return out

    # ============================================================
    #  统计
    # ============================================================
    def stats(self) -> Dict[str, Any]:
        s = {
            "user_id": self.user_id,
            **{t: mem.stats() for t, mem in self.memory_types.items()},
            "backends": {
                "vector": self.vector_store.stats() if self.vector_store else None,
                "graph": self.graph_store.stats() if self.graph_store else None,
                "document": self.doc_store.stats() if self.doc_store else None,
                "embedder": self.embedder.name,
                "embedder_dim": getattr(self.embedder, "dimension", 0),
            },
        }
        if self.persist:
            s["persist_stats"] = self.persist.stats()
        return s

    # ============================================================
    #  摘要(给 LLM 用)
    # ============================================================
    def summary(self, query: Optional[str] = None, limit_per_type: int = 3) -> str:
        """生成记忆摘要,便于塞进 LLM 上下文"""
        if query:
            items_by_type = self.search_all(query, limit_per_type=limit_per_type)
        else:
            items_by_type = {t: mem.items[-limit_per_type:] for t, mem in self.memory_types.items()}
        lines: List[str] = []
        type_label = {
            "working": "工作记忆", "episodic": "情景记忆",
            "semantic": "语义记忆", "perceptual": "感知记忆",
        }
        for t, items in items_by_type.items():
            if not items:
                continue
            lines.append(f"## {type_label.get(t, t)} ({len(items)} 条)")
            for i, it in enumerate(items, 1):
                preview = it.content[:80] + ("..." if len(it.content) > 80 else "")
                lines.append(f"  ({i}) [imp={it.importance:.2f}] {preview}")
        return "\n".join(lines) if lines else "(无记忆)"
