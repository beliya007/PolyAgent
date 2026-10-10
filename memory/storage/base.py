"""存储后端抽象层 (Storage Backend Abstraction)

设计参考 hello-agents 第八章:
- 三类存储:VectorStore(向量) / GraphStore(图) / DocumentStore(文档)
- 统一接口:add / search / delete / stats
- 支持内存/SQLite/Qdrant/Neo4j 等多种实现,可按需切换或组合
- 所有外部依赖(qdrant-client / neo4j / sqlite)均为可选,
  缺失时自动回落到内存实现,保证项目始终可运行
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
#  通用数据容器
# ============================================================
@dataclass
class VectorRecord:
    """向量存储中的一条记录"""
    id: str
    vector: List[float] = field(default_factory=list)
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """图节点"""
    id: str
    label: str = "Entity"            # 节点类型(Entity / Memory / Concept ...)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """图边"""
    source_id: str
    target_id: str
    relation: str                     # 关系类型(RELATED_TO / HAS_PROPERTY ...)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentRecord:
    """结构化文档记录(用于 SQLite 等关系型存储)"""
    id: str
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
#  向量存储抽象
# ============================================================
class BaseVectorStore(ABC):
    """向量存储统一接口

    add_vectors / search_similar / delete / stats 是所有向量后端的最小公约数。
    """

    backend_name: str = "base"

    @abstractmethod
    def add_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        contents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """批量添加向量,返回成功条数"""
        raise NotImplementedError

    @abstractmethod
    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[VectorRecord, float]]:
        """相似度检索,返回 (record, score) 列表,按 score 倒序"""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "count": self.count()}

    # ----- 文本检索(可选,默认 None 表示不支持) -----
    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        embed_fn=None,                              # callable: str -> List[float]
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[VectorRecord, float]]:
        """先用 embed_fn 把文本转成向量,再 search_similar"""
        if embed_fn is None:
            raise NotImplementedError("此后端不支持文本检索,需提供 embed_fn")
        qv = embed_fn(query)
        return self.search_similar(qv, top_k=top_k, where=where)


# ============================================================
#  图存储抽象
# ============================================================
class BaseGraphStore(ABC):
    """图存储统一接口(用于实体/关系/知识图谱)"""

    backend_name: str = "base"

    @abstractmethod
    def upsert_node(self, node: GraphNode) -> None: ...

    @abstractmethod
    def upsert_edge(self, edge: GraphEdge) -> None: ...

    @abstractmethod
    def find_node(self, node_id: str) -> Optional[GraphNode]: ...

    @abstractmethod
    def neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        limit: int = 20,
    ) -> List[GraphNode]: ...

    @abstractmethod
    def query(self, cypher_like: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> int: ...

    def stats(self) -> Dict[str, Any]:
        return {"backend": self.backend_name}


# ============================================================
#  文档存储抽象
# ============================================================
class BaseDocumentStore(ABC):
    """结构化文档存储(SQLite / JSONL / 等)"""

    backend_name: str = "base"

    @abstractmethod
    def upsert(self, doc: DocumentRecord) -> None: ...

    @abstractmethod
    def get(self, doc_id: str) -> Optional[DocumentRecord]: ...

    @abstractmethod
    def delete(self, ids: List[str]) -> int: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def query(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[DocumentRecord]: ...

    def stats(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "count": self.count()}
