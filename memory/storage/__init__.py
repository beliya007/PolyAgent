"""storage 包 — 多后端存储层"""
from memory.storage.base import (
    BaseDocumentStore,
    BaseGraphStore,
    BaseVectorStore,
    DocumentRecord,
    GraphEdge,
    GraphNode,
    VectorRecord,
)
from memory.storage.memory_store import (
    MemoryDocumentStore,
    MemoryGraphStore,
    MemoryVectorStore,
    bm25_search,
)
from memory.storage.sqlite_store import SQLiteDocumentStore

# 可选后端(导入失败时不抛错,只是不可用)
try:
    from memory.storage.qdrant_store import QdrantVectorStore  # noqa: F401
except Exception:  # pragma: no cover
    QdrantVectorStore = None  # type: ignore

try:
    from memory.storage.neo4j_store import Neo4jGraphStore     # noqa: F401
except Exception:  # pragma: no cover
    Neo4jGraphStore = None  # type: ignore


__all__ = [
    # 抽象
    "BaseVectorStore", "BaseGraphStore", "BaseDocumentStore",
    "VectorRecord", "GraphNode", "GraphEdge", "DocumentRecord",
    # 内存实现(零依赖,始终可用)
    "MemoryVectorStore", "MemoryGraphStore", "MemoryDocumentStore", "bm25_search",
    # SQLite(标准库,始终可用)
    "SQLiteDocumentStore",
    # 可选外部实现
    "QdrantVectorStore", "Neo4jGraphStore",
]
