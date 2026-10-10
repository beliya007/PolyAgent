"""
Neo4j 图存储 (Neo4jGraphStore)

polyagent 第八章设计:
- 知识图谱:节点 = Entity,边 = 关系
- 支持 Cypher 查询
- 缺失 neo4j 驱动时,降级为 MemoryGraphStore,保证代码可运行

环境变量:
  NEO4J_URI       e.g. neo4j+s://xxx.databases.neo4j.io  或  bolt://localhost:7687
  NEO4J_USERNAME  neo4j
  NEO4J_PASSWORD  ********
  NEO4J_DATABASE  默认 neo4j
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from memory.storage.base import BaseGraphStore, GraphEdge, GraphNode
from memory.storage.memory_store import MemoryGraphStore


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j 实现,失败时无缝降级为内存实现"""

    backend_name = "neo4j"

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_timeout: int = 60,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = None
        self._fallback: Optional[MemoryGraphStore] = None
        self._init_driver()

    def _init_driver(self) -> None:
        if not self.uri:
            self._switch_to_fallback("未配置 NEO4J_URI")
            return
        try:
            from neo4j import GraphDatabase  # type: ignore
        except Exception as e:
            self._switch_to_fallback(f"未安装 neo4j: {e}")
            return
        try:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=self._env_int(
                    "NEO4J_MAX_CONNECTION_LIFETIME", 3600
                ),
                max_connection_pool_size=self._env_int(
                    "NEO4J_MAX_CONNECTION_POOL_SIZE", 50
                ),
                connection_timeout=self._env_int(
                    "NEO4J_CONNECTION_TIMEOUT", 60
                ),
            )
            self._driver.verify_connectivity()
            # 索引(可选)
            with self._driver.session(database=self.database) as s:
                s.run("CREATE INDEX entity_id IF NOT EXISTS FOR (n:Entity) ON (n.id)")
            print(f"[Neo4j] ✅ 已连接 {self.uri} | database={self.database}")
        except Exception as e:
            self._switch_to_fallback(f"连接失败: {e}")

    def _env_int(self, key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except Exception:
            return default

    def _switch_to_fallback(self, reason: str) -> None:
        if self._fallback is None:
            self._fallback = MemoryGraphStore()
        print(f"[Neo4j] ⚠️ 降级为内存实现 ({reason})")

    @property
    def is_remote(self) -> bool:
        return self._driver is not None

    # ---------------- 写入 ----------------
    def upsert_node(self, node: GraphNode) -> None:
        if self._driver is None:
            assert self._fallback is not None
            self._fallback.upsert_node(node)
            return
        try:
            with self._driver.session(database=self.database) as s:
                s.run(
                    f"""
                    MERGE (n:`{node.label}` {{id: $id}})
                    SET n += $props
                    """,
                    id=node.id,
                    props=_stringify_props(node.properties),
                )
        except Exception as e:
            print(f"[Neo4j] upsert_node 失败: {e}")

    def upsert_edge(self, edge: GraphEdge) -> None:
        if self._driver is None:
            assert self._fallback is not None
            self._fallback.upsert_edge(edge)
            return
        try:
            with self._driver.session(database=self.database) as s:
                s.run(
                    f"""
                    MATCH (a {{id: $src}})
                    MATCH (b {{id: $dst}})
                    MERGE (a)-[r:`{edge.relation}`]->(b)
                    SET r += $props
                    """,
                    src=edge.source_id,
                    dst=edge.target_id,
                    props=_stringify_props(edge.properties),
                )
        except Exception as e:
            print(f"[Neo4j] upsert_edge 失败: {e}")

    # ---------------- 查询 ----------------
    def find_node(self, node_id: str) -> Optional[GraphNode]:
        if self._driver is None:
            assert self._fallback is not None
            return self._fallback.find_node(node_id)
        try:
            with self._driver.session(database=self.database) as s:
                rec = s.run(
                    "MATCH (n {id:$id}) RETURN labels(n) AS labels, properties(n) AS props LIMIT 1",
                    id=node_id,
                ).single()
            if rec is None:
                return None
            labels = rec["labels"] or ["Entity"]
            return GraphNode(id=node_id, label=str(labels[0]), properties=dict(rec["props"]))
        except Exception as e:
            print(f"[Neo4j] find_node 失败: {e}")
            return None

    def neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        limit: int = 20,
    ) -> List[GraphNode]:
        if self._driver is None:
            assert self._fallback is not None
            return self._fallback.neighbors(node_id, relation, limit)
        try:
            with self._driver.session(database=self.database) as s:
                rel_clause = f":`{relation}`" if relation else ""
                rec = s.run(
                    f"""
                    MATCH (n {{id:$id}})-[r{rel_clause}]->(m)
                    RETURN labels(m) AS labels, properties(m) AS props
                    LIMIT $limit
                    """,
                    id=node_id, limit=limit,
                )
                out: List[GraphNode] = []
                for row in rec:
                    labels = row["labels"] or ["Entity"]
                    out.append(
                        GraphNode(
                            id=str(row["props"].get("id", "")),
                            label=str(labels[0]),
                            properties=dict(row["props"]),
                        )
                    )
                return out
        except Exception as e:
            print(f"[Neo4j] neighbors 失败: {e}")
            return []

    def query(self, cypher_like: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """直接执行一段 Cypher,返回记录列表。"""
        if self._driver is None:
            # 内存版只支持子集语义
            assert self._fallback is not None
            return self._fallback.query(cypher_like, params)
        try:
            with self._driver.session(database=self.database) as s:
                rec = s.run(cypher_like, **(params or {}))
                return [dict(row) for row in rec]
        except Exception as e:
            print(f"[Neo4j] query 失败: {e}")
            return []

    def delete(self, ids: List[str]) -> int:
        if self._driver is None:
            assert self._fallback is not None
            return self._fallback.delete(ids)
        try:
            with self._driver.session(database=self.database) as s:
                rec = s.run(
                    "MATCH (n) WHERE n.id IN $ids DETACH DELETE n RETURN count(n) AS c",
                    ids=ids,
                ).single()
                return int(rec["c"]) if rec else 0
        except Exception as e:
            print(f"[Neo4j] delete 失败: {e}")
            return 0

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        base.update({
            "uri": self.uri,
            "database": self.database,
            "remote": self.is_remote,
        })
        return base


def _stringify_props(d: Dict[str, Any]) -> Dict[str, Any]:
    """Neo4j 属性必须是基本类型"""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = [str(x) for x in v]
        elif v is None:
            continue
        else:
            out[k] = str(v)
    return out
