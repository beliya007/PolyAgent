"""memory 包 — 4 种记忆 + MemoryManager + 持久化 + 多后端存储"""
from memory.base import BaseMemory, MemoryItem
from memory.config import MemoryConfig, get_memory_config
from memory.embedding import (
    BaseEmbedding,
    DashScopeEmbedding,
    FallbackEmbedding,
    LocalEmbedding,
    TFIDFEmbedding,
    get_text_embedder,
)
from memory.episodic import EpisodicMemory
from memory.manager import MemoryManager
from memory.perceptual import PerceptualMemory
from memory.persistence import JSONLPersistence
from memory.semantic import SemanticMemory
from memory.storage import (
    # 抽象
    BaseDocumentStore,
    BaseGraphStore,
    BaseVectorStore,
    DocumentRecord,
    GraphEdge,
    GraphNode,
    VectorRecord,
    # 内存实现
    MemoryDocumentStore,
    MemoryGraphStore,
    MemoryVectorStore,
    bm25_search,
    # SQLite
    SQLiteDocumentStore,
    # 可选外部实现(可能为 None)
    Neo4jGraphStore,
    QdrantVectorStore,
)
from memory.working import WorkingMemory

__all__ = [
    # 基础
    "BaseMemory",
    "MemoryItem",
    # 配置
    "MemoryConfig",
    "get_memory_config",
    # 嵌入
    "BaseEmbedding",
    "DashScopeEmbedding",
    "LocalEmbedding",
    "TFIDFEmbedding",
    "FallbackEmbedding",
    "get_text_embedder",
    # 四种记忆
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "PerceptualMemory",
    # 管理器
    "MemoryManager",
    # 持久化(JSONL, 向后兼容)
    "JSONLPersistence",
    # 存储抽象
    "BaseVectorStore", "BaseGraphStore", "BaseDocumentStore",
    "VectorRecord", "GraphNode", "GraphEdge", "DocumentRecord",
    # 内存实现
    "MemoryVectorStore", "MemoryGraphStore", "MemoryDocumentStore", "bm25_search",
    # SQLite
    "SQLiteDocumentStore",
    # 可选外部
    "QdrantVectorStore", "Neo4jGraphStore",
]
