"""
统一嵌入服务 (Embedding Service)

polyagent 第八章设计:
- 首选 DashScope(通义千问)云端 API
- 降级到本地 sentence-transformers
- 再次降级到 TF-IDF 轻量级兜底

设计要点:
- 单一公开接口: get_text_embedder() 返回一个符合 BaseEmbedding 协议的对象
- 自动 fallback 链,无需业务方感知底层用了哪个实现
- TF-IDF 维度自适应(默认 128),无需预训练模型
- 所有失败都通过 stdout 警告而非抛错,保持系统可用
"""
from __future__ import annotations

import math
import os
import re
import threading
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

# ============================================================
#  抽象
# ============================================================
class BaseEmbedding:
    """嵌入服务统一接口"""
    name: str = "base"
    dimension: int = 0

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def encode_one(self, text: str) -> List[float]:
        vecs = self.encode([text])
        return vecs[0] if vecs else []

    # TF-IDF 等确定性实现可以缓存
    def warmup(self, corpus: Optional[Sequence[str]] = None) -> None:
        """预热(可选):为 TF-IDF 等模型准备词表"""
        return None


# ============================================================
#  1) DashScope (阿里云百炼 / 通义千问) 嵌入
# ============================================================
class DashScopeEmbedding(BaseEmbedding):
    """阿里云百炼 / OpenAI 兼容接口的 embedding 实现

    兼容的 base_url: dashscope.aliyuncs.com / OpenAI / 任意 compatible-mode
    """
    name = "dashscope"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        from config import get_config
        cfg = get_config().llm
        self.api_key = api_key or os.getenv("EMBED_API_KEY") or cfg.api_key
        self.base_url = (
            base_url
            or os.getenv("EMBED_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or os.getenv("EMBED_MODEL_NAME") or cfg.embedding_model or "text-embedding-v3"
        self.timeout = timeout
        self.dimension = int(os.getenv("EMBED_DIM", "1024"))
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.api_key:
            print("[Embedding: DashScope] ⚠️ 未配置 API Key,稍后将自动 fallback")
            return
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            print(f"[Embedding: DashScope] ✅ 已连接 {self.base_url} | model={self.model}")
        except Exception as e:
            print(f"[Embedding: DashScope] ⚠️ 初始化失败: {e}")
            self._client = None

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self._client is None or not texts:
            return []
        try:
            # DashScope text-embedding-v3 / v4 支持 dimensions 参数
            kwargs: Dict[str, Any] = {"model": self.model, "input": list(texts)}
            try:
                resp = self._client.embeddings.create(**kwargs, dimensions=self.dimension)
            except TypeError:
                resp = self._client.embeddings.create(**kwargs)
            vecs: List[List[float]] = [list(d.embedding) for d in resp.data]
            if vecs and vecs[0]:
                self.dimension = len(vecs[0])
            return vecs
        except Exception as e:
            print(f"[Embedding: DashScope] encode 失败: {e}")
            return []


# ============================================================
#  2) 本地 sentence-transformers
# ============================================================
class LocalEmbedding(BaseEmbedding):
    """本地 sentence-transformers 嵌入(离线可用)

    默认模型: all-MiniLM-L6-v2 (384 维,英文友好)
    """
    name = "local"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name or os.getenv(
            "EMBED_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = device
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
            print(f"[Embedding: Local] ✅ loaded {self.model_name} (dim={self.dimension})")
        except Exception as e:
            print(f"[Embedding: Local] ⚠️ 加载失败: {e}")
            self._model = None

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self._model is None or not texts:
            return []
        try:
            arr = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
            return [list(map(float, v)) for v in arr]
        except Exception as e:
            print(f"[Embedding: Local] encode 失败: {e}")
            return []


# ============================================================
#  3) TF-IDF 兜底
# ============================================================
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SPLIT_RE = re.compile(r"[\s,;.!?。，；！？、]+")


def _tokenize_for_tfidf(text: str) -> List[str]:
    """中英混合分词:CJK 按字符,英文按词"""
    text = text.lower()
    tokens: List[str] = []
    buf: List[str] = []
    for ch in text:
        if _CJK_RE.match(ch):
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
    if buf:
        tokens.append("".join(buf))
    return [t for t in tokens if t]


class TFIDFEmbedding(BaseEmbedding):
    """TF-IDF 嵌入(轻量级、无需模型)

    - 自动构建词表,基于给定语料计算 IDF
    - 维度 = min(target_dim, vocab_size)
    - 余弦归一化
    """
    name = "tfidf"

    def __init__(self, target_dim: int = 128) -> None:
        self.target_dim = target_dim
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._fitted = False
        self._lock = threading.Lock()
        self.dimension = target_dim

    def fit(self, corpus: Sequence[str]) -> None:
        """基于语料训练词表与 IDF"""
        with self._lock:
            docs_tokens = [_tokenize_for_tfidf(d) for d in corpus]
            df: Counter = Counter()
            for toks in docs_tokens:
                for t in set(toks):
                    df[t] += 1
            n = max(len(docs_tokens), 1)
            # 选频次最高的 target_dim 个词作为词表
            top_terms = [t for t, _ in df.most_common(self.target_dim)]
            self._vocab = {t: i for i, t in enumerate(top_terms)}
            self._idf = {
                t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items() if t in self._vocab
            }
            self._fitted = bool(self._vocab)
            self.dimension = max(len(self._vocab), 1)
        if self._fitted:
            print(f"[Embedding: TFIDF] ✅ 词表大小 {len(self._vocab)},dim={self.dimension}")

    def warmup(self, corpus: Optional[Sequence[str]] = None) -> None:
        if corpus:
            self.fit(corpus)

    def _encode_one_locked(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        toks = _tokenize_for_tfidf(text)
        if not toks or not self._vocab:
            return vec
        tf = Counter(toks)
        for t, freq in tf.items():
            idx = self._vocab.get(t)
            if idx is None:
                continue
            vec[idx] = float(freq) * float(self._idf.get(t, 1.0))
        # 归一化
        n = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / n for x in vec]

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        with self._lock:
            if not self._fitted:
                self.fit(texts)
            return [self._encode_one_locked(t) for t in texts]


# ============================================================
#  4) Fallback Embedder 链
# ============================================================
class FallbackEmbedding(BaseEmbedding):
    """按顺序尝试多个 embedder,直到某个能产出非空向量

    用法:
        embedder = FallbackEmbedding([DashScopeEmbedding(), LocalEmbedding(), TFIDFEmbedding()])
        vecs = embedder.encode(["你好", "hello"])
    """
    name = "fallback"

    def __init__(self, embedders: List[BaseEmbedding], min_dim: int = 16) -> None:
        self.embedders = [e for e in embedders if e is not None]
        self.min_dim = min_dim
        self.dimension = max((e.dimension for e in self.embedders if e.dimension > 0), default=min_dim)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        texts = list(texts)
        if not texts:
            return []
        results: List[Optional[List[float]]] = [None] * len(texts)
        for emb in self.embedders:
            try:
                vecs = emb.encode(texts)
            except Exception as e:
                print(f"[Embedding: {emb.name}] 失败,继续 fallback: {e}")
                vecs = []
            if not vecs or len(vecs) != len(texts):
                continue
            ok = True
            for i, v in enumerate(vecs):
                if results[i] is not None:
                    continue
                if v and len(v) >= self.min_dim and any(x != 0.0 for x in v):
                    results[i] = v
            if all(r is not None for r in results):
                break
        # 任意没成功的,用最后一个 embedder 再来一遍(避免返回全零)
        if any(r is None for r in results):
            for emb in self.embedders[::-1]:
                try:
                    fill = emb.encode([t for i, t in enumerate(texts) if results[i] is None])
                except Exception:
                    fill = []
                idx = 0
                for i in range(len(results)):
                    if results[i] is None and idx < len(fill):
                        results[i] = fill[idx]
                        idx += 1
        # 最终兜底:全零
        for i in range(len(results)):
            if results[i] is None:
                results[i] = [0.0] * max(self.dimension, self.min_dim)
        # 更新 dimension
        if results and results[0]:
            self.dimension = len(results[0])
        return [r if r is not None else [0.0] * self.dimension for r in results]

    def warmup(self, corpus: Optional[Sequence[str]] = None) -> None:
        for e in self.embedders:
            try:
                e.warmup(corpus)
            except Exception:
                pass


# ============================================================
#  5) 单例工厂
# ============================================================
_embedder_singleton: Optional[BaseEmbedding] = None
_embedder_lock = threading.Lock()


def get_text_embedder(
    prefer: str = "auto",
    corpus: Optional[Sequence[str]] = None,
    force_new: bool = False,
) -> BaseEmbedding:
    """获取全局嵌入器(单例)

    prefer:
      - "auto"   按 DashScope → Local → TFIDF 自动尝试
      - "dashscope" / "local" / "tfidf" 强制指定
    """
    global _embedder_singleton
    with _embedder_lock:
        if _embedder_singleton is not None and not force_new:
            return _embedder_singleton

        chain: List[BaseEmbedding] = []
        if prefer == "auto":
            chain.append(DashScopeEmbedding())
            chain.append(LocalEmbedding())
            chain.append(TFIDFEmbedding())
        elif prefer == "dashscope":
            chain.append(DashScopeEmbedding())
        elif prefer == "local":
            chain.append(LocalEmbedding())
        elif prefer == "tfidf":
            chain.append(TFIDFEmbedding())

        # FallbackEmbedding 自动按顺序选择
        embedder = FallbackEmbedding(chain)
        if corpus:
            embedder.warmup(corpus)
        _embedder_singleton = embedder
        return embedder


def reset_embedder() -> None:
    """重置单例(测试用)"""
    global _embedder_singleton
    _embedder_singleton = None
