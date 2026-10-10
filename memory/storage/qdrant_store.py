"""
Qdrant 向量存储 (QdrantVectorStore)

hello-agents 第八章:
- 高性能向量检索(基于 Qdrant Cloud 或本地 Docker)
- 支持 metadata 过滤 + 余弦/欧氏距离
- 缺失 qdrant-client 时,自动回落到 MemoryVectorStore,保证代码可运行

环境变量:
  QDRANT_URL              e.g. https://xxx.qdrant.tech:6333
  QDRANT_API_KEY          可选
  QDRANT_COLLECTION       集合名,默认 hello_agents_vectors
  QDRANT_VECTOR_SIZE      向量维度,默认 1024
  QDRANT_DISTANCE         cosine / dot / euclid
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from memory.storage.base import BaseVectorStore, VectorRecord
from memory.storage.memory_store import MemoryVectorStore


class QdrantVectorStore(BaseVectorStore):
    """Qdrant 实现,失败时无缝降级为内存实现"""

    backend_name = "qdrant"

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: Optional[str] = None,
        vector_size: Optional[int] = None,
        distance: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.url = url or os.getenv("QDRANT_URL", "")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", "")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "hello_agents_vectors")
        self.vector_size = int(vector_size or os.getenv("QDRANT_VECTOR_SIZE", "1024"))
        self.distance = (distance or os.getenv("QDRANT_DISTANCE", "Cosine")).lower()
        self.timeout = int(os.getenv("QDRANT_TIMEOUT", str(timeout)))

        self._client = None
        self._fallback: Optional[MemoryVectorStore] = None
        self._init_client()

    # ---------------- 初始化 ----------------
    def _init_client(self) -> None:
        if not self.url:
            self._switch_to_fallback("未配置 QDRANT_URL")
            return
        try:
            from qdrant_client import QdrantClient        # type: ignore
            from qdrant_client.http import models as qm   # type: ignore
        except Exception as e:
            self._switch_to_fallback(f"未安装 qdrant-client: {e}")
            return

        try:
            self._client = QdrantClient(
                url=self.url, api_key=self.api_key or None, timeout=self.timeout
            )
            # 确保集合存在
            existing = [c.name for c in self._client.get_collections().collections]
            if self.collection not in existing:
                dist_map = {
                    "cosine": qm.Distance.COSINE,
                    "dot": qm.Distance.DOT,
                    "euclid": qm.Distance.EUCLID,
                }
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qm.VectorParams(
                        size=self.vector_size, distance=dist_map.get(self.distance, qm.Distance.COSINE)
                    ),
                )
                print(f"[Qdrant] 创建集合 {self.collection} (dim={self.vector_size})")
            print(f"[Qdrant] ✅ 已连接 {self.url} | collection={self.collection}")
        except Exception as e:
            self._switch_to_fallback(f"连接失败: {e}")

    def _switch_to_fallback(self, reason: str) -> None:
        if self._fallback is None:
            self._fallback = MemoryVectorStore()
        print(f"[Qdrant] ⚠️ 降级为内存实现 ({reason})")

    @property
    def is_remote(self) -> bool:
        return self._client is not None

    # ---------------- 写入 ----------------
    def add_vectors(
        self,
        ids: List[str],
        vectors: List[List[float]],
        contents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        if self._client is None:
            assert self._fallback is not None
            return self._fallback.add_vectors(ids, vectors, contents, metadatas)
        try:
            from qdrant_client.http import models as qm  # type: ignore

            points = []
            for i, vid in enumerate(ids):
                payload: Dict[str, Any] = {"content": (contents[i] if contents and i < len(contents) else "") or ""}
                if metadatas and i < len(metadatas) and metadatas[i]:
                    # qdrant 要求 metadata 值是基本类型或 list[int]/str/float/bool
                    payload.update(self._normalize_meta(metadatas[i]))
                points.append(qm.PointStruct(id=vid, vector=list(vectors[i]), payload=payload))
            self._client.upsert(collection_name=self.collection, points=points)
            return len(points)
        except Exception as e:
            print(f"[Qdrant] upsert 失败,降级内存: {e}")
            assert self._fallback is not None
            return self._fallback.add_vectors(ids, vectors, contents, metadatas)

    # ---------------- 检索 ----------------
    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[VectorRecord, float]]:
        if self._client is None:
            assert self._fallback is not None
            return self._fallback.search_similar(query_vector, top_k, where, score_threshold)
        try:
            from qdrant_client.http import models as qm  # type: ignore

            flt = None
            if where:
                flt = qm.Filter(must=[qm.FieldCondition(key=k, match=qm.MatchValue(value=v))
                                       for k, v in where.items()])
            resp = self._client.search(
                collection_name=self.collection,
                query_vector=list(query_vector),
                limit=top_k,
                query_filter=flt,
                score_threshold=score_threshold,
                with_payloads=True,
            )
            out: List[Tuple[VectorRecord, float]] = []
            for hit in resp:
                payload = dict(hit.payload or {})
                content = payload.pop("content", "") if "content" in payload else ""
                out.append(
                    (
                        VectorRecord(
                            id=str(hit.id),
                            vector=list(query_vector),  # Qdrant 不返回向量,占位
                            content=content,
                            metadata=payload,
                        ),
                        float(hit.score),
                    )
                )
            return out
        except Exception as e:
            print(f"[Qdrant] search 失败,降级内存: {e}")
            assert self._fallback is not None
            return self._fallback.search_similar(query_vector, top_k, where, score_threshold)

    # ---------------- 删除 ----------------
    def delete(self, ids: List[str]) -> int:
        if self._client is None:
            assert self._fallback is not None
            return self._fallback.delete(ids)
        try:
            from qdrant_client.http import models as qm  # type: ignore

            self._client.delete(
                collection_name=self.collection,
                points_selector=qm.PointIdsList(points=ids),
            )
            return len(ids)
        except Exception as e:
            print(f"[Qdrant] delete 失败: {e}")
            return 0

    def count(self) -> int:
        if self._client is None:
            assert self._fallback is not None
            return self._fallback.count()
        try:
            info = self._client.get_collection(self.collection)
            return int(getattr(info, "vectors_count", 0) or 0)
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        base = super().stats()
        base.update({
            "url": self.url,
            "collection": self.collection,
            "vector_size": self.vector_size,
            "remote": self.is_remote,
        })
        return base

    @staticmethod
    def _normalize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Qdrant payload 字段必须是基本类型,递归转换"""
        out: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, list):
                out[k] = [str(x) if not isinstance(x, (int, float, str, bool)) else x for x in v]
            elif v is None:
                continue
            else:
                out[k] = str(v)
        return out
