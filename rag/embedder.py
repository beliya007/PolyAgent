"""
向量化 (Embedder)

通过 LLM 客户端获取文本向量(调用 Embedding API)。
底层失败时自动降级为 BM25 文本检索。
"""
from __future__ import annotations

from typing import List

from llm_client import PolyAgentLLM


class Embedder:
    """Embedding 客户端,封装 PolyAgentLLM 的 embed 接口"""

    def __init__(self, llm: PolyAgentLLM):
        self.llm = llm
        self.dimension = 1024  # text-embedding-v3 默认

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.llm.embed(texts)
        if vectors and vectors[0]:
            self.dimension = len(vectors[0])
        return vectors

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
