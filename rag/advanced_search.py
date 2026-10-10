"""
高级检索策略 (Advanced Search Strategies)

polyagent 第八章完整实现:
- MQE (Multi-Query Expansion):用 LLM 生成多样化查询
- HyDE (Hypothetical Document Embeddings):用答案找答案
- 统一的扩展检索框架:expansion → search → dedup → rank
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from llm_client import PolyAgentLLM


# ============================================================
#  MQE - 多查询扩展
# ============================================================
_MQE_SYSTEM = (
    "你是检索查询扩展助手。生成语义等价或互补的多样化查询。"
    "使用中文,简短,避免标点,每行一个查询。"
)
_MQE_USER_TPL = "原始查询:{query}\n请给出{n}个不同表述的查询,每行一个。"


def _prompt_mqe(query: str, n: int = 2, llm: Optional[PolyAgentLLM] = None) -> List[str]:
    """用 LLM 生成 {n} 个语义等价的查询"""
    if not query:
        return []
    llm = llm or _try_llm()
    if llm is None:
        return []
    try:
        text = llm.invoke([
            {"role": "system", "content": _MQE_SYSTEM},
            {"role": "user", "content": _MQE_USER_TPL.format(query=query, n=n)},
        ]) or ""
        lines: List[str] = []
        for ln in text.splitlines():
            ln = re.sub(r"^[\-\*\d\.\s]+", "", ln).strip()
            if ln and ln != query:
                lines.append(ln)
        return lines[:n] or [query]
    except Exception:
        return [query]


# ============================================================
#  HyDE - 假设文档嵌入
# ============================================================
_HYDE_SYSTEM = (
    "根据用户问题,先写一段可能的答案性段落,用于向量检索的查询文档。"
    "不要分析过程,直接写正文。"
)
_HYDE_USER_TPL = "问题:{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"


def _prompt_hyde(query: str, llm: Optional[PolyAgentLLM] = None) -> Optional[str]:
    """让 LLM 写一段假设性答案,用于缩小 q↔d 语义鸿沟"""
    if not query:
        return None
    llm = llm or _try_llm()
    if llm is None:
        return None
    try:
        text = llm.invoke([
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": _HYDE_USER_TPL.format(query=query)},
        ])
        text = (text or "").strip()
        return text or None
    except Exception:
        return None


# ============================================================
#  统一扩展检索
# ============================================================
def search_with_expansion(
    retrieve_fn: Callable[[str, int], List[Dict[str, Any]]],
    query: str,
    top_k: int = 5,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
    llm: Optional[PolyAgentLLM] = None,
) -> List[Dict[str, Any]]:
    """统一的扩展检索入口(用于任意 retrieve_fn)

    retrieve_fn: callable(query, top_k) -> [{"id", "content", "score", ...}, ...]
    """
    if not query:
        return []

    expansions: List[str] = [query]
    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(query, mqe_expansions, llm))
    if enable_hyde:
        hyde_text = _prompt_hyde(query, llm)
        if hyde_text:
            expansions.append(hyde_text)

    # 去重
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq

    # 候选池
    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))

    agg: Dict[str, Dict[str, Any]] = {}
    for q in expansions:
        try:
            hits = retrieve_fn(q, per)
        except Exception:
            hits = []
        for h in hits:
            hid = h.get("id") or h.get("memory_id") or h.get("content", "")[:64]
            score = float(h.get("score", 0.0))
            if hid not in agg or score > float(agg[hid].get("score", 0.0)):
                agg[hid] = dict(h)
                agg[hid]["score"] = score
                agg[hid]["matched_query"] = q

    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]


# ============================================================
#  辅助:安全获取 LLM 实例
# ============================================================
def _try_llm() -> Optional[PolyAgentLLM]:
    try:
        from llm_client import PolyAgentLLM  # type: ignore
        return PolyAgentLLM()
    except Exception:
        return None


__all__ = [
    "_prompt_mqe",
    "_prompt_hyde",
    "search_with_expansion",
]