"""
内存存储实现 (In-Memory Storage)

- 默认实现,零外部依赖
- 向量检索使用 NumPy 加速(可选,纯 Python 兜底)
- 文档存储用 dict
- 图存储用邻接表
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from memory.storage.base import (
    BaseDocumentStore,
    BaseGraphStore,
    BaseVectorStore,
    DocumentRecord,
    GraphEdge,
    GraphNode,
    VectorRecord,
)


# ============================================================
#  内存向量存储
# ============================================================
class MemoryVectorStore(BaseVectorStore):
    """基于 NumPy 的内存向量库

    - 余弦相似度
    - metadata 支持精确字段过滤
    - 内容保留,可直接读取原文
    """

    backend_name = "memory"

    def __init__(self) -> None:
        self.records: Dict[str, VectorRecord] = {}
        self._np = None
        try:
            import numpy as np  # type: ignore
            self._np = np
        except Exception:
            pass

    # ---------- 写入 ----------
    def add_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        contents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        n = 0
        for i, vid in enumerate(ids):
            content = (contents[i] if contents and i < len(contents) else "") or ""
            meta = (metadatas[i] if metadatas and i < len(metadatas) else {}) or {}
            self.records[vid] = VectorRecord(
                id=vid,
                vector=list(vectors[i]) if i < len(vectors) else [],
                content=content,
                metadata=dict(meta),
            )
            n += 1
        return n

    # ---------- 检索 ----------
    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[VectorRecord, float]]:
        candidates = self._filter(self.records.values(), where)
        if not candidates:
            return []
        scored: List[Tuple[VectorRecord, float]] = []
        for rec in candidates:
            if not rec.vector:
                continue
            sim = self._cosine(query_vector, rec.vector)
            if score_threshold is not None and sim < score_threshold:
                continue
            scored.append((rec, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ---------- 删除 ----------
    def delete(self, ids: List[str]) -> int:
        n = 0
        for vid in ids:
            if vid in self.records:
                self.records.pop(vid, None)
                n += 1
        return n

    def count(self) -> int:
        return len(self.records)

    def clear(self) -> None:
        self.records.clear()

    # ---------- 工具 ----------
    @staticmethod
    def _filter(records, where: Optional[Dict[str, Any]]):
        if not where:
            return list(records)
        return [r for r in records if all(r.metadata.get(k) == v for k, v in where.items())]

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


# ============================================================
#  内存图存储
# ============================================================
class MemoryGraphStore(BaseGraphStore):
    """邻接表实现的轻量级图存储"""

    backend_name = "memory"

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        # out_edges: src -> list[GraphEdge]
        self.out_edges: Dict[str, List[GraphEdge]] = defaultdict(list)
        # in_edges:  dst -> list[GraphEdge]  (便于反向查找)
        self.in_edges: Dict[str, List[GraphEdge]] = defaultdict(list)

    def upsert_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def upsert_edge(self, edge: GraphEdge) -> None:
        self.out_edges[edge.source_id].append(edge)
        self.in_edges[edge.target_id].append(edge)

    def find_node(self, node_id: str) -> Optional[GraphNode]:
        return self.nodes.get(node_id)

    def neighbors(
        self,
        node_id: str,
        relation: Optional[str] = None,
        limit: int = 20,
    ) -> List[GraphNode]:
        out: List[GraphNode] = []
        for edge in self.out_edges.get(node_id, []):
            if relation and edge.relation != relation:
                continue
            nb = self.nodes.get(edge.target_id)
            if nb is not None:
                out.append(nb)
            if len(out) >= limit:
                break
        return out

    def query(self, cypher_like: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """极简子集解析:支持 MATCH (n)-[r]->(m) WHERE n.label=$label

        仅用于语义记忆的实体关系查询,够用即可。
        """
        params = params or {}
        results: List[Dict[str, Any]] = []
        # 简化:把所有 (src, edge, dst) 三元组展开
        triples: List[Tuple[GraphNode, GraphEdge, GraphNode]] = []
        for src_id, edges in self.out_edges.items():
            src_node = self.nodes.get(src_id)
            if src_node is None:
                continue
            for edge in edges:
                dst_node = self.nodes.get(edge.target_id)
                if dst_node is None:
                    continue
                triples.append((src_node, edge, dst_node))

        for src, edge, dst in triples:
            if params.get("src_label") and src.label != params["src_label"]:
                continue
            if params.get("dst_label") and dst.label != params["dst_label"]:
                continue
            if params.get("relation") and edge.relation != params["relation"]:
                continue
            results.append(
                {
                    "src": {"id": src.id, "label": src.label, "props": src.properties},
                    "edge": {"relation": edge.relation, "props": edge.properties},
                    "dst": {"id": dst.id, "label": dst.label, "props": dst.properties},
                }
            )
            if len(results) >= params.get("limit", 50):
                break
        return results

    def delete(self, ids: List[str]) -> int:
        n = 0
        for nid in ids:
            if nid in self.nodes:
                self.nodes.pop(nid, None)
                self.out_edges.pop(nid, None)
                self.in_edges.pop(nid, None)
                n += 1
        # 清理涉及这些 id 的边
        for src in list(self.out_edges.keys()):
            self.out_edges[src] = [e for e in self.out_edges[src] if e.target_id not in ids]
        for dst in list(self.in_edges.keys()):
            self.in_edges[dst] = [e for e in self.in_edges[dst] if e.source_id not in ids]
        return n

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "nodes": len(self.nodes),
            "edges": sum(len(v) for v in self.out_edges.values()),
        }


# ============================================================
#  内存文档存储
# ============================================================
class MemoryDocumentStore(BaseDocumentStore):
    """基于 dict 的结构化文档存储"""

    backend_name = "memory"

    def __init__(self) -> None:
        self.docs: Dict[str, DocumentRecord] = {}

    def upsert(self, doc: DocumentRecord) -> None:
        self.docs[doc.id] = doc

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        return self.docs.get(doc_id)

    def delete(self, ids: List[str]) -> int:
        n = 0
        for did in ids:
            if did in self.docs:
                self.docs.pop(did, None)
                n += 1
        return n

    def count(self) -> int:
        return len(self.docs)

    def query(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[DocumentRecord]:
        out: List[DocumentRecord] = []
        for d in self.docs.values():
            if where and not all(d.metadata.get(k) == v for k, v in where.items()):
                continue
            out.append(d)
            if len(out) >= limit:
                break
        return out

    def stats(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "count": self.count()}


# ============================================================
#  文本检索工具(BM25 简易实现,供图/文档存储的文本搜索使用)
# ============================================================
_BM25_SPLIT_RE = re.compile(r"[\s,。!?;:,\.\!\?;:]+")


def bm25_search(query: str, docs: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
    """简易 BM25(单语料快速过滤)

    返回 [(doc_idx, score)] 倒序。
    """
    if not query or not docs:
        return []
    q_tokens = [t for t in _BM25_SPLIT_RE.split(query.lower()) if t]
    if not q_tokens:
        return []

    tokenized_docs = [[t for t in _BM25_SPLIT_RE.split(d.lower()) if t] for d in docs]
    df = defaultdict(int)
    for toks in tokenized_docs:
        for t in set(toks):
            df[t] += 1
    N = len(tokenized_docs)
    avgdl = sum(len(t) for t in tokenized_docs) / max(N, 1)
    k1, b = 1.5, 0.75

    scored: List[Tuple[int, float]] = []
    for idx, toks in enumerate(tokenized_docs):
        if not toks:
            continue
        score = 0.0
        dl = len(toks)
        for qt in q_tokens:
            if qt not in toks:
                continue
            f = toks.count(qt)
            idf = math.log(1 + (N - df[qt] + 0.5) / (df[qt] + 0.5))
            denom = f + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * (f * (k1 + 1)) / max(denom, 1e-9)
        if score > 0:
            scored.append((idx, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
