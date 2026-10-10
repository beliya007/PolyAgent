"""
记忆系统配置 (MemoryConfig)

集中管理所有存储后端的开关和连接参数,
业务侧可以通过 .env 或直接传参覆盖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key, default)
    return v.strip() if isinstance(v, str) else v


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


# ============================================================
#  各后端配置
# ============================================================
@dataclass
class VectorBackendConfig:
    """向量存储后端配置"""
    backend: str = "memory"                                  # memory | qdrant
    url: Optional[str] = None
    api_key: Optional[str] = None
    collection: str = "hello_agents_vectors"
    vector_size: int = 1024
    distance: str = "cosine"
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "VectorBackendConfig":
        return cls(
            backend=_env("VECTOR_BACKEND", "memory") or "memory",
            url=_env("QDRANT_URL"),
            api_key=_env("QDRANT_API_KEY"),
            collection=_env("QDRANT_COLLECTION", "hello_agents_vectors") or "hello_agents_vectors",
            vector_size=_env_int("QDRANT_VECTOR_SIZE", 1024),
            distance=(_env("QDRANT_DISTANCE", "cosine") or "cosine").lower(),
            timeout=_env_int("QDRANT_TIMEOUT", 30),
        )


@dataclass
class GraphBackendConfig:
    """图存储后端配置"""
    backend: str = "memory"                                   # memory | neo4j
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: str = "neo4j"

    @classmethod
    def from_env(cls) -> "GraphBackendConfig":
        return cls(
            backend=_env("GRAPH_BACKEND", "memory") or "memory",
            uri=_env("NEO4J_URI"),
            username=_env("NEO4J_USERNAME", "neo4j"),
            password=_env("NEO4J_PASSWORD"),
            database=_env("NEO4J_DATABASE", "neo4j") or "neo4j",
        )


@dataclass
class DocumentBackendConfig:
    """结构化文档存储后端配置"""
    backend: str = "memory"                                   # memory | sqlite
    sqlite_path: str = "./data/memory/memory.db"

    @classmethod
    def from_env(cls) -> "DocumentBackendConfig":
        default_path = "./data/memory/memory.db"
        return cls(
            backend=_env("DOC_BACKEND", "memory") or "memory",
            sqlite_path=_env("MEMORY_SQLITE_PATH", default_path) or default_path,
        )


@dataclass
class EmbeddingConfig:
    """嵌入服务配置"""
    prefer: str = "auto"                                      # auto | dashscope | local | tfidf
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    dimension: int = 1024

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            prefer=(_env("EMBED_PREFER", "auto") or "auto").lower(),
            model=_env("EMBED_MODEL_NAME"),
            base_url=_env("EMBED_BASE_URL"),
            api_key=_env("EMBED_API_KEY"),
            dimension=_env_int("EMBED_DIM", 1024),
        )


# ============================================================
#  顶层配置
# ============================================================
@dataclass
class MemoryConfig:
    """记忆系统统一配置"""

    # 各后端开关
    vector_backend: str = "memory"
    graph_backend: str = "memory"
    document_backend: str = "memory"
    embedding_prefer: str = "auto"

    vector: VectorBackendConfig = field(default_factory=VectorBackendConfig.from_env)
    graph: GraphBackendConfig = field(default_factory=GraphBackendConfig.from_env)
    document: DocumentBackendConfig = field(default_factory=DocumentBackendConfig.from_env)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig.from_env)

    # 各类记忆容量 / TTL
    working_memory_capacity: int = 50
    working_memory_ttl_min: int = 60
    episodic_capacity: int = 5000
    semantic_capacity: int = 5000
    perceptual_capacity: int = 1000
    working_memory_ttl_seconds: int = 3600

    # 检索参数
    search_default_limit: int = 5
    search_score_threshold: float = 0.0

    # 启用哪些记忆类型
    enabled_types: List[str] = field(default_factory=lambda: ["working", "episodic", "semantic", "perceptual"])

    # JSONL 持久化(向后兼容旧版)
    enable_jsonl_persistence: bool = False
    persist_dir: str = "./data/memory"

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        return cls(
            vector_backend=_env("VECTOR_BACKEND", "memory") or "memory",
            graph_backend=_env("GRAPH_BACKEND", "memory") or "memory",
            document_backend=_env("DOC_BACKEND", "memory") or "memory",
            embedding_prefer=(_env("EMBED_PREFER", "auto") or "auto").lower(),
            working_memory_capacity=_env_int("WORKING_MEMORY_CAPACITY", 50),
            working_memory_ttl_min=_env_int("WORKING_MEMORY_TTL", 60),
            episodic_capacity=_env_int("EPISODIC_CAPACITY", 5000),
            semantic_capacity=_env_int("SEMANTIC_CAPACITY", 5000),
            perceptual_capacity=_env_int("PERCEPTUAL_CAPACITY", 1000),
            working_memory_ttl_seconds=_env_int("WORKING_MEMORY_TTL_SECONDS", 3600),
            search_default_limit=_env_int("MEMORY_SEARCH_LIMIT", 5),
            search_score_threshold=float(_env("MEMORY_SCORE_THRESHOLD", "0.0") or 0.0),
            enable_jsonl_persistence=_env_bool("MEMORY_ENABLE_JSONL", False),
            persist_dir=_env("MEMORY_STORAGE_PATH", "./data/memory") or "./data/memory",
        )

    def backend_for(self, role: str) -> str:
        return {
            "vector": self.vector_backend,
            "graph": self.graph_backend,
            "document": self.document_backend,
        }.get(role, "memory")


# 单例缓存
_config: Optional[MemoryConfig] = None


def get_memory_config(reload: bool = False) -> MemoryConfig:
    global _config
    if _config is None or reload:
        _config = MemoryConfig.from_env()
    return _config
